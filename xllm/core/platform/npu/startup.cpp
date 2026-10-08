/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/platform/npu/startup.h"

#include <absl/strings/str_join.h>
#include <absl/strings/str_split.h>
#include <acl/acl.h>
#include <fcntl.h>
#include <glog/logging.h>
#include <pybind11/embed.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/xattr.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/config/distributed_config.h"
#include "core/framework/config/model_config.h"
#include "core/platform/device_name_utils.h"

namespace py = pybind11;

namespace xllm {
namespace {
// Initialize Python interpreter and torch_npu runtime early, before any NPU
// tensor allocation. torch_npu (post4+) calls PyGILState_Ensure inside
// empty_with_format(), so Python must be alive before the first NPU op.
// All NPU processes go through this path for consistency — the build system
// links against the pip-installed torch_npu .so directly.
// The caller owns the interpreter and holds the GIL during initialization.
void init_npu_python_runtime() {
  // Select the same logical device this process's worker will run on. Multi-
  // process single-card serving lets every process see all TP cards and picks
  // its own via node_rank (see Master ctor). Using .front() here would pin an
  // extra context on logical device 0 for every node_rank != 0 process, piling
  // small allocations onto die0. Mirror master.cpp's get_device_idx instead.
  const auto& distributed_config = DistributedConfig::get_instance();
  const int32_t visible_device_count =
      DeviceNameUtils::parse_devices("auto").size();
  const int32_t device_index =
      DeviceNameUtils::get_device_idx(distributed_config.node_rank(),
                                      distributed_config.nnodes(),
                                      visible_device_count);

  // Register fla_npu's embedded AscendC opapi (libcust_opapi) + OPP vendor
  // dir before aclInit. The KDA forward op (aclnnChunkKdaFwd) and its kernels
  // live in fla_npu's embedded OPP; without this registration the opapi cannot
  // locate the kernel binary and aclnnChunkKdaFwdGetWorkspaceSize returns
  // ACLNN_ERR_INNER_NULLPTR (561103), breaking GLM-5.3-Flash's KDA layers.
  // Guarded so non-KDA builds without fla_npu stay unaffected.
  {
    py::gil_scoped_acquire gil;
    py::exec(
        "try:\n"
        "    import fla_npu\n"
        "except ModuleNotFoundError as error:\n"
        "    if error.name != 'fla_npu':\n"
        "        raise\n"
        "else:\n"
        "    fla_npu.load_ascendc_opapi_libraries()\n");
  }

  const aclError acl_ret = aclInit(nullptr);
  CHECK(acl_ret == ACL_SUCCESS || acl_ret == ACL_ERROR_INTERNAL_ERROR)
      << "aclInit failed with error " << acl_ret;

  // Bind the ACL context before torch_npu creates its default runtime state.
  // Without this, every rank can briefly allocate on logical device 0 before
  // the worker selects its own card, leaving stray per-rank processes there.
  const aclError set_device_ret = aclrtSetDevice(device_index);
  CHECK_EQ(set_device_ret, ACL_SUCCESS)
      << "aclrtSetDevice failed for device " << device_index << " with error "
      << set_device_ret;

  // We own ACL initialization, so torch_npu will skip aclFinalize. Register
  // before importing torch_npu: Python runs exit hooks in reverse order,
  // releasing its groups, streams and devices before we finalize CANN.
  py::module_::import("atexit").attr("register")(py::cpp_function(
      []() {
        const aclError status = aclFinalize();
        if (status != ACL_SUCCESS) {
          LOG(ERROR) << "aclFinalize failed with error " << status;
        }
      },
      py::call_guard<py::gil_scoped_release>()));

  {
    py::gil_scoped_acquire gil;
    py::exec(
        "import os, sys\n"
        "os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'\n"
        "import torch\n"
        "orig = torch._C._get_accelerator\n"
        "try:\n"
        "    torch._C._get_accelerator = lambda: torch.device('cpu')\n"
        "    import torch_npu\n"
        "finally:\n"
        "    torch._C._get_accelerator = orig\n"
        "import torch_npu.npu as _npu_mod\n"
        "try:\n"
        "    torch_npu._C._npu_init()\n"
        "except RuntimeError as e:\n"
        "    if 'already initialized' not in str(e).lower():\n"
        "        raise\n"
        "_npu_mod._initialized = True\n"
        "_npu_mod._original_pid = os.getpid()\n"
        "torch_npu._C._npu_setDevice(" +
        std::to_string(device_index) + ")\n");
  }
}

std::string build_priority_line(const std::vector<std::string>& desired_order,
                                const std::vector<std::string>& existing) {
  std::vector<std::string> vendors = desired_order;
  for (const std::string& v : existing) {
    if (v.empty()) {
      continue;
    }
    if (std::find(vendors.begin(), vendors.end(), v) == vendors.end()) {
      vendors.push_back(v);
    }
  }
  return "load_priority=" + absl::StrJoin(vendors, ",");
}

// Preserve the access ACL strictly; other xattrs are best-effort.
bool copy_xattrs_to_fd(const std::filesystem::path& src, int dst_fd) {
  static constexpr const char* kAcl = "system.posix_acl_access";

  ssize_t acl_sz = ::getxattr(src.c_str(), kAcl, nullptr, 0);
  if (acl_sz < 0) {
    if (errno == ENODATA || errno == ENOENT || errno == ENOTSUP) {
      return true;
    }
    LOG(WARNING) << "getxattr ACL " << src << " failed ("
                 << std::strerror(errno) << "); config.ini left unchanged";
    return false;
  }
  if (acl_sz == 0) {
    return true;
  }
  std::vector<char> acl(static_cast<size_t>(acl_sz));
  ssize_t got = ::getxattr(src.c_str(), kAcl, acl.data(), acl.size());
  if (got <= 0) {
    LOG(WARNING) << "read ACL " << src << " failed (" << std::strerror(errno)
                 << "); config.ini left unchanged";
    return false;
  }
  if (::fsetxattr(dst_fd, kAcl, acl.data(), static_cast<size_t>(got), 0) != 0) {
    LOG(WARNING) << "fsetxattr ACL on temp failed (" << std::strerror(errno)
                 << "); config.ini left unchanged";
    return false;
  }

  ssize_t list_sz = ::listxattr(src.c_str(), nullptr, 0);
  if (list_sz <= 0) {
    return true;
  }
  std::vector<char> names(static_cast<size_t>(list_sz));
  ssize_t actual = ::listxattr(src.c_str(), names.data(), names.size());
  if (actual <= 0) {
    return true;
  }
  size_t off = 0;
  const size_t end = static_cast<size_t>(actual);
  while (off < end) {
    const char* name = names.data() + off;
    const size_t name_len = std::strlen(name);
    off += name_len + 1;
    if (std::strcmp(name, kAcl) == 0) {
      continue;
    }
    ssize_t val_sz = ::getxattr(src.c_str(), name, nullptr, 0);
    if (val_sz <= 0) {
      continue;
    }
    std::vector<char> val(static_cast<size_t>(val_sz));
    if (::getxattr(src.c_str(), name, val.data(), val.size()) <= 0) {
      continue;
    }
    ::fsetxattr(dst_fd, name, val.data(), val.size(), 0);
  }
  return true;
}

// Atomic replacement must preserve ownership, permissions and the access ACL.
bool publish_config_ini(const std::filesystem::path& target,
                        const std::string& content) {
  namespace fs = std::filesystem;
  const fs::path dir = target.parent_path();
  std::string tmpl = (dir / "config.ini.xllm.tmp.XXXXXX").string();
  const int fd = ::mkstemp(tmpl.data());
  if (fd < 0) {
    LOG(WARNING) << "mkstemp in " << dir << " failed (" << std::strerror(errno)
                 << "); config.ini left unchanged";
    return false;
  }
  const fs::path tmp = tmpl;
  class PublishGuard {
   public:
    int fd;
    fs::path path;
    bool dismissed = false;
    ~PublishGuard() {
      if (fd >= 0) {
        ::close(fd);
      }
      if (!dismissed) {
        std::error_code ec;
        fs::remove(path, ec);
      }
    }
  } guard{fd, tmp, false};

  size_t off = 0;
  while (off < content.size()) {
    const ssize_t n = ::write(fd, content.data() + off, content.size() - off);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      LOG(WARNING) << "write " << tmp << " failed (" << std::strerror(errno)
                   << "); config.ini left unchanged";
      return false;
    }
    off += static_cast<size_t>(n);
  }
  if (::fsync(fd) != 0) {
    LOG(WARNING) << "fsync " << tmp << " failed (" << std::strerror(errno)
                 << "); config.ini left unchanged";
    return false;
  }
  struct stat st;
  mode_t mode = 0644;
  uid_t uid = static_cast<uid_t>(-1);
  gid_t gid = static_cast<gid_t>(-1);
  if (::stat(target.c_str(), &st) == 0) {
    mode = st.st_mode & 0777;
    uid = st.st_uid;
    gid = st.st_gid;
  } else if (errno != ENOENT) {
    LOG(WARNING) << "stat " << target << " failed (" << std::strerror(errno)
                 << "); config.ini left unchanged";
    return false;
  }
  if (::fchmod(fd, mode) != 0) {
    LOG(WARNING) << "fchmod " << tmp << " failed (" << std::strerror(errno)
                 << "); config.ini left unchanged";
    return false;
  }
  if ((uid != static_cast<uid_t>(-1) || gid != static_cast<gid_t>(-1)) &&
      ::fchown(fd, uid, gid) != 0) {
    LOG(WARNING) << "fchown " << tmp << " failed (" << std::strerror(errno)
                 << "); cannot preserve ownership, config.ini left unchanged";
    return false;
  }

  if (!copy_xattrs_to_fd(target, fd)) {
    return false;
  }
  const int close_ret = ::close(fd);
  guard.fd = -1;
  if (close_ret != 0) {
    LOG(WARNING) << "close " << tmp << " failed (" << std::strerror(errno)
                 << "); config.ini left unchanged";
    return false;
  }
  std::error_code rename_ec;
  fs::rename(tmp, target, rename_ec);
  if (rename_ec) {
    LOG(WARNING) << "rename " << tmp << " -> " << target << " failed ("
                 << rename_ec.message() << "); config.ini left unchanged";
    return false;
  }
  guard.dismissed = true;
  return true;
}

struct VendorConfigRead {
  std::vector<std::string> lines;
  int32_t priority_index = -1;
  std::vector<std::string> existing_vendors;
  bool bad = false;
  bool exists_unreadable = false;
};

bool priority_matches(const VendorConfigRead& config,
                      const std::string& desired_line) {
  // Cosmetic differences must not force a potentially failing rewrite.
  return config.priority_index >= 0 &&
         build_priority_line({}, config.existing_vendors) == desired_line;
}

VendorConfigRead read_vendor_config(const std::filesystem::path& target) {
  VendorConfigRead r;
  const std::string prefix = "load_priority=";
  std::ifstream ifs(target);
  if (!ifs) {
    std::error_code ec;
    if (std::filesystem::exists(target, ec)) {
      r.exists_unreadable = true;
    }
    return r;
  }
  std::string line;
  while (std::getline(ifs, line)) {
    if (line.rfind(prefix, 0) == 0) {
      r.priority_index = static_cast<int32_t>(r.lines.size());
      r.existing_vendors = std::vector<std::string>(
          absl::StrSplit(line.substr(prefix.size()), ','));
    }
    r.lines.push_back(line);
  }
  if (ifs.bad()) {
    r.bad = true;
  }
  return r;
}

bool ensure_priority_published(const std::filesystem::path& target,
                               const std::vector<std::string>& desired_order,
                               const VendorConfigRead& r,
                               bool& wrote) {
  wrote = false;
  const std::string new_priority =
      build_priority_line(desired_order, r.existing_vendors);

  if (priority_matches(r, new_priority)) {
    return true;
  }
  std::string content;
  for (size_t i = 0; i < r.lines.size(); ++i) {
    content +=
        i == static_cast<size_t>(r.priority_index) ? new_priority : r.lines[i];
    content += '\n';
  }
  if (r.priority_index < 0) {
    content += new_priority;
    content += '\n';
  }
  if (publish_config_ini(target, content)) {
    wrote = true;
    return true;
  }
  return false;
}

// Hold a shared lock through aclInit to prevent vendor changes during its read.
// Writers publish under an exclusive lock, then recheck after reacquiring
// shared.
class VendorConfigLock {
 public:
  VendorConfigLock() {
    const char* disable = std::getenv("XLLM_DISABLE_VENDOR_CONFIG_INI_WRITE");
    if (disable != nullptr && std::string(disable) == "1") {
      return;
    }
    const char* opp_path_env = std::getenv("ASCEND_OPP_PATH");
    if (opp_path_env == nullptr || std::string(opp_path_env).empty()) {
      LOG(WARNING) << "ASCEND_OPP_PATH is not set; skip vendor config lock";
      return;
    }
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path config_file =
        fs::canonical(fs::path(opp_path_env) / "vendors" / "config.ini", ec);
    target_ =
        ec ? fs::path(opp_path_env) / "vendors" / "config.ini" : config_file;
    compute_desired();
    if (desired_order_.empty()) {
      LOG(WARNING) << "no target vendor for " << backend_label_
                   << " backend is installed under " << target_.parent_path()
                   << "; vendor config hook is a no-op (proceeding with the "
                      "existing config, no lock, no write)";
      return;
    }
    const fs::path lockfile = target_.parent_path() / ".xllm_vendor.flock";

    fd_ = ::open(lockfile.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0666);
    if (fd_ >= 0) {
      ::fchmod(fd_, 0666);
    } else if (errno == EACCES) {
      // flock also permits a read-only fd for a lockfile owned by another UID.
      fd_ = ::open(lockfile.c_str(), O_RDONLY | O_CLOEXEC);
    }
    if (fd_ < 0) {
      LOG(WARNING) << "open lockfile " << lockfile << " failed ("
                   << std::strerror(errno)
                   << "); vendor lock unavailable, skip write (leave "
                      "config.ini as-is)";
      return;
    }

    acquire_with_retry();
  }
  ~VendorConfigLock() {
    if (fd_ >= 0) {
      ::flock(fd_, LOCK_UN);
      ::close(fd_);
      fd_ = -1;
    }
  }
  VendorConfigLock(const VendorConfigLock&) = delete;
  VendorConfigLock& operator=(const VendorConfigLock&) = delete;

  bool unsettled() const { return state_ == State::kUnsettled; }

 private:
  enum class State : int8_t { kIdle, kSettled, kUnsettled };
  State state_ = State::kIdle;
  int fd_ = -1;
  std::filesystem::path target_;
  std::vector<std::string> desired_order_;
  std::string backend_label_;

  static std::vector<std::string> desired_order_for(bool is_python) {
    return is_python ? std::vector<std::string>{"glm_next_transformer",
                                                "custom_transformer",
                                                "custom_xllm_math"}
                     : std::vector<std::string>{"custom_xllm_math",
                                                "custom_transformer",
                                                "glm_next_transformer"};
  }

  void compute_desired() {
    const bool is_python = ModelConfig::is_python_model_impl(
        ModelConfig::get_instance().model_impl());
    desired_order_ = desired_order_for(is_python);
    backend_label_ = is_python ? "python" : "atb";

    // Missing vendor libraries stop CANN's search, hiding later usable vendors.
    namespace fs = std::filesystem;
    const fs::path vendors_dir = target_.parent_path();
    std::error_code ec;
    std::vector<std::string> installed;
    installed.reserve(desired_order_.size());
    for (const std::string& v : desired_order_) {
      const fs::path so =
          vendors_dir / v / "op_api" / "lib" / "libcust_opapi.so";
      if (fs::exists(so, ec)) {
        installed.push_back(v);
      } else {
        LOG(WARNING) << "vendor " << v << " has no libcust_opapi.so under "
                     << vendors_dir << "; dropping from " << backend_label_
                     << " load_priority to avoid breaking the vendor search";
      }
    }
    desired_order_ = std::move(installed);
  }

  enum class PublishResult : int8_t {
    kPublished,
    kMatched,

    kPublishFailedUnreadable,  // Leave an unknown order unchanged.

    kPublishFailedStale,  // Refuse aclInit on a known conflicting order.
  };

  enum class CheckResult : int8_t {
    kCheckSettled,
    kCheckMismatch,
    kCheckAbort,
  };

  // Caller holds LOCK_EX.
  PublishResult publish_under_exclusive() {
    VendorConfigRead w = read_vendor_config(target_);
    if (w.bad || w.exists_unreadable) {
      return PublishResult::kPublishFailedUnreadable;
    }
    bool wrote = false;
    if (!ensure_priority_published(target_, desired_order_, w, wrote)) {
      return PublishResult::kPublishFailedStale;
    }
    if (wrote) {
      LOG(INFO) << "wrote " << target_ << " "
                << build_priority_line(desired_order_, w.existing_vendors)
                << " for " << backend_label_ << " backend";
      return PublishResult::kPublished;
    }
    return PublishResult::kMatched;
  }

  CheckResult check_match_under_shared() {
    if (::flock(fd_, LOCK_SH) != 0) {
      LOG(WARNING) << "flock LOCK_SH failed (" << std::strerror(errno) << ")";
      return CheckResult::kCheckAbort;
    }
    VendorConfigRead r = read_vendor_config(target_);
    if (r.bad) {
      LOG(WARNING) << "read " << target_ << " failed mid-stream";
      ::flock(fd_, LOCK_UN);
      return CheckResult::kCheckAbort;
    }
    if (r.exists_unreadable) {
      LOG(WARNING) << target_ << " exists but cannot be read; skip lock";
      ::flock(fd_, LOCK_UN);
      return CheckResult::kCheckAbort;
    }

    const std::string want =
        build_priority_line(desired_order_, r.existing_vendors);
    if (priority_matches(r, want)) {
      LOG(INFO) << "vendors/config.ini load_priority matches (" << want
                << "); holding shared lock through aclInit for "
                << backend_label_ << " backend";
      state_ = State::kSettled;
      return CheckResult::kCheckSettled;
    }
    ::flock(fd_, LOCK_UN);
    return CheckResult::kCheckMismatch;
  }

  // Probe first so same-backend readers can initialize concurrently.
  void acquire_with_retry() {
    for (int32_t round = 0; round < 4; ++round) {
      PublishResult pr = PublishResult::kMatched;
      bool did_publish = false;
      for (int32_t probe = 0; probe < 200; ++probe) {
        const CheckResult c = check_match_under_shared();
        if (c == CheckResult::kCheckSettled) {
          return;
        }
        if (c == CheckResult::kCheckAbort) {
          return;
        }

        if (::flock(fd_, LOCK_EX | LOCK_NB) == 0) {
          pr = publish_under_exclusive();
          ::flock(fd_, LOCK_UN);
          did_publish = true;
          break;
        }
        if (errno != EWOULDBLOCK && errno != EAGAIN) {
          LOG(WARNING) << "flock LOCK_EX|NB failed (" << std::strerror(errno)
                       << ")";
          return;
        }

        ::usleep(1000);
      }

      if (!did_publish) {
        if (::flock(fd_, LOCK_EX) != 0) {
          LOG(WARNING) << "flock LOCK_EX failed (" << std::strerror(errno)
                       << ")";
          return;
        }
        pr = publish_under_exclusive();
        ::flock(fd_, LOCK_UN);
      }

      if (pr == PublishResult::kPublishFailedUnreadable) {
        LOG(WARNING) << "vendor config unreadable; config.ini left as-is, "
                     << "proceeding without a lock";
        return;
      }
      if (pr == PublishResult::kPublishFailedStale) {
        LOG(ERROR) << "config.ini is at a different-backend order and we "
                      "cannot rewrite it (permission/IO); refusing to aclInit "
                      "on a wrong order";
        state_ = State::kUnsettled;
        return;
      }

      // Another writer may win the exclusive-to-shared transition.
      const CheckResult c = check_match_under_shared();
      if (c == CheckResult::kCheckSettled) {
        return;
      }
      if (c == CheckResult::kCheckAbort) {
        return;
      }
    }

    LOG(ERROR) << "could not settle vendor config order to " << backend_label_
               << " backend under cross-backend contention; refusing to "
                  "aclInit on a different-backend order";
    state_ = State::kUnsettled;
  }
};
}  // namespace

bool initialize_npu_startup() {
  VendorConfigLock vendor_lock;
  if (vendor_lock.unsettled()) {
    LOG(ERROR) << "refusing to start: vendor config rewrite failed or "
                  "backends are contending; see earlier errors. Check write "
                  "permissions or avoid concurrent launches of different "
                  "backends. Set XLLM_DISABLE_VENDOR_CONFIG_INI_WRITE=1 to "
                  "skip automatic vendor configuration";
    return false;
  }
  init_npu_python_runtime();
  return true;
}

}  // namespace xllm

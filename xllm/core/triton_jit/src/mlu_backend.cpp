/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "backends/mlu_backend.h"

#include <framework/core/MLUStream.h>
#include <framework/core/device.h>
#include <glog/logging.h>

#include <cstdint>
#include <fstream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace xllm::triton_jit {

namespace {

inline std::string cnbin_path(const std::string& dir, const std::string& name) {
  return dir + "/" + name + ".cnbin";
}

inline std::string meta_path(const std::string& dir, const std::string& name) {
  return dir + "/" + name + ".json";
}

inline void check_cn(CNresult code, const std::string& ctx) {
  if (code != CN_SUCCESS) {
    const char* s = nullptr;
    cnGetErrorString(code, &s);
    if (s == nullptr) {
      s = "unknown cndrv error";
    }
    LOG(FATAL) << "triton_jit MLU: " << ctx << ": " << s;
  }
}

inline void check_cn_recoverable(CNresult code, const std::string& ctx) {
  if (code != CN_SUCCESS) {
    const char* s = nullptr;
    cnGetErrorString(code, &s);
    if (s == nullptr) {
      s = "unknown cndrv error";
    }
    throw std::runtime_error("triton_jit MLU: " + ctx + ": " + s);
  }
}

MluMeta load_meta(const std::string& dir, const std::string& name) {
  MluMeta m;
  std::ifstream f(meta_path(dir, name));
  if (!f.is_open()) {
    return m;
  }
  nlohmann::json j =
      nlohmann::json::parse(f, /*cb=*/nullptr, /*allow_exceptions=*/false);
  if (!j.is_object()) {
    return m;
  }
  m.shared = j.value("shared", 0u);
  m.num_warps = j.value("num_warps", 1);
  m.promote_shared = j.value("promote_shared", false);
  if (j.contains("target") && j["target"].is_object() &&
      j["target"].contains("arch")) {
    m.arch = j["target"]["arch"].get<uint32_t>();
  }
  return m;
}

void ensure_context() {
  CNcontext ctx = nullptr;
  if (cnCtxGetCurrent(&ctx) == CN_SUCCESS && ctx != nullptr) {
    return;
  }
  CNdev dev = 0;
  check_cn(cnDeviceGet(&dev, /*ordinal=*/0), "cnDeviceGet");
  CNcontext new_ctx = nullptr;
  check_cn(cnCtxCreate(&new_ctx, /*flags=*/0, dev), "cnCtxCreate");
  check_cn(cnCtxSetCurrent(new_ctx), "cnCtxSetCurrent");
}

int32_t current_dev() {
  return static_cast<int32_t>(torch_mlu::current_device());
}

}  // namespace

MluCompiledKernel::MluCompiledKernel(std::string dir, std::string name)
    : dir_(std::move(dir)), name_(std::move(name)) {}

MluCompiledKernel::~MluCompiledKernel() = default;

void MluCompiledKernel::lazy_load() {
  std::call_once(load_once_, [this]() {
    ensure_context();

    meta_ = load_meta(dir_, name_);

    // Arch compatibility: relax when the metadata lacks an arch (only verify
    // when both sides are known), to stay robust across triton-mlu versions.
    if (meta_.arch != 0) {
      torch_mlu::DeviceProp* prop =
          torch_mlu::getDeviceProperties(current_dev());
      if (prop != nullptr && prop->isa_version != -1 &&
          static_cast<uint32_t>(prop->isa_version) != meta_.arch) {
        LOG(FATAL) << "triton_jit MLU: ISA mismatch, kernel arch=" << meta_.arch
                   << " device arch=" << prop->isa_version;
      }
    }

    check_cn_recoverable(
        cnModuleLoad(cnbin_path(dir_, name_).c_str(), &module_),
        "cnModuleLoad");
    check_cn_recoverable(cnModuleGetKernel(module_, name_.c_str(), &kernel_),
                         "cnModuleGetKernel");

    torch_mlu::DeviceProp* prop = torch_mlu::getDeviceProperties(current_dev());
    if (prop != nullptr) {
      if (prop->core_num_per_cluster > 0) {
        core_per_cluster_ = prop->core_num_per_cluster;
      }
      if (prop->cluster_count > 0) {
        cluster_count_ = prop->cluster_count;
      }
    }
  });
}

void MluCompiledKernel::launch(void* stream, Grid g, const void* const* args) {
  lazy_load();

  uint32_t gx = g.x * static_cast<uint32_t>(meta_.num_warps);
  uint32_t gy = g.y;
  uint32_t gz = g.z;
  uint64_t total = static_cast<uint64_t>(gx) * gy * gz;

  uint64_t func_type = static_cast<uint64_t>(meta_.num_warps);
  bool tiles_evenly = (core_per_cluster_ > 0) && (gx % core_per_cluster_ == 0);
  bool fits_cluster =
      total <= static_cast<uint64_t>(cluster_count_) * core_per_cluster_;
  if (meta_.num_warps == 1 && tiles_evenly &&
      (meta_.promote_shared || fits_cluster)) {
    func_type = static_cast<uint64_t>(core_per_cluster_);
  }

  CNqueue queue = reinterpret_cast<CNqueue>(stream);
  CNresult r = cnInvokeKernel(kernel_,
                              gx,
                              gy,
                              gz,
                              static_cast<KernelClass>(func_type),
                              /*extra=*/0,
                              queue,
                              const_cast<void**>(args),
                              /*extra2=*/nullptr);
  check_cn_recoverable(r, "cnInvokeKernel");
}

int32_t MluBackend::current_device() const { return current_dev(); }

std::string MluBackend::compile_options_json() const {
  // MLU-specific triton.compile hints; merged on top of {num_warps,num_stages}
  // by the Python compile script.
  return R"({"is_linear_hint":true,"restrict_ptr_hint":true})";
}

std::unique_ptr<CompiledKernel> MluBackend::load_kernel(
    const std::string& cache_dir,
    const std::string& name) const {
  return std::make_unique<MluCompiledKernel>(cache_dir, name);
}

void MluBackend::sync(void* /*stream*/) const { torch_mlu::synchronize(); }

bool MluBackend::is_capturing(void* stream) const {
  cnrtQueueCaptureStatus status{};
  unsigned long capture_id = 0;
  cnrtQueue_t queue = reinterpret_cast<cnrtQueue_t>(stream);
  auto r = cnrtQueueGetCaptureInfo(queue,
                                   &status,
                                   &capture_id,
                                   /*graph_id=*/nullptr,
                                   /*flags=*/nullptr,
                                   /*dependency=*/nullptr);
  if (r != cnrtSuccess) {
    return false;
  }
  return status == cnrtQueueCaptureStatus::cnrtQueueCaptureStatusActive;
}

}  // namespace xllm::triton_jit

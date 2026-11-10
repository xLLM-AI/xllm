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

#include "core/common/native_library.h"

#include <dlfcn.h>
#include <glog/logging.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>

extern "C" __attribute__((visibility("default"))) void
xllm_native_library_host_anchor() {}

namespace xllm {
namespace {

void host_location() {}

void export_host_symbols() {
  Dl_info host_info{};
  CHECK_NE(dladdr(reinterpret_cast<void*>(&host_location), &host_info), 0)
      << "Cannot locate the xLLM native library host.";

  // RTLD_DEFAULT can include the caller's local lookup scope on glibc. An
  // explicit process handle only searches symbols visible to other modules.
  void* process_handle = dlopen(nullptr, RTLD_NOW);
  CHECK_NE(process_handle, nullptr)
      << "Cannot inspect xLLM process symbols: " << dlerror();
  void* anchor = dlsym(process_handle, "xllm_native_library_host_anchor");
  if (anchor == nullptr) {
    // Python normally imports extension modules with RTLD_LOCAL. Promote the
    // existing module, rather than loading a second copy of its registries.
    void* handle =
        dlopen(host_info.dli_fname, RTLD_NOW | RTLD_NOLOAD | RTLD_GLOBAL);
    CHECK_NE(handle, nullptr)
        << "Cannot export xLLM host symbols: " << dlerror();
    anchor = dlsym(process_handle, "xllm_native_library_host_anchor");
  }
  CHECK_NE(anchor, nullptr)
      << "xLLM host registration symbols are not exported.";
  Dl_info anchor_info{};
  CHECK_NE(dladdr(anchor, &anchor_info), 0);
  CHECK_EQ(anchor_info.dli_fbase, host_info.dli_fbase)
      << "Multiple xLLM hosts cannot share native extension registries.";
  dlclose(process_handle);
}

void check_string(const char* field,
                  const char* expected,
                  const char* actual,
                  const std::filesystem::path& path) {
  CHECK_NE(actual, nullptr)
      << "Native library " << path << " has no " << field << ".";
  CHECK_EQ(std::strcmp(expected, actual), 0)
      << "Native library " << path << " has incompatible " << field
      << ": expected " << expected << ", got " << actual << ".";
}

void check_build_info(void* handle, const std::filesystem::path& path) {
  using BuildInfoGetter = const NativeLibraryBuildInfo* (*)();
  dlerror();
  auto getter = reinterpret_cast<BuildInfoGetter>(
      dlsym(handle, "xllm_native_library_build_info"));
  const char* error = dlerror();
  CHECK_EQ(error, nullptr) << "Native library " << path
                           << " has no build descriptor: " << error;
  CHECK_NE(getter, nullptr);
  Dl_info getter_info{};
  CHECK_NE(dladdr(reinterpret_cast<void*>(getter), &getter_info), 0)
      << "Cannot locate the build descriptor for native library " << path
      << ".";
  std::error_code getter_path_error;
  const std::filesystem::path getter_path =
      std::filesystem::canonical(getter_info.dli_fname, getter_path_error);
  CHECK(!getter_path_error)
      << "Cannot resolve build descriptor owner for native library " << path
      << ": " << getter_path_error.message();
  CHECK_EQ(getter_path, path) << "Native library " << path
                              << " does not define its own build descriptor.";
  const NativeLibraryBuildInfo* actual = getter();
  CHECK_NE(actual, nullptr)
      << "Native library " << path << " returned no build descriptor.";
  static constexpr NativeLibraryBuildInfo kExpected =
      detail::native_library_build_info();
  CHECK_EQ(actual->version, kExpected.version)
      << "Native library " << path
      << " has an incompatible descriptor version.";
  CHECK_EQ(actual->size, kExpected.size)
      << "Native library " << path << " has an incompatible descriptor size.";
  CHECK_NE(std::strcmp(kExpected.revision, "unknown"), 0)
      << "Native libraries require an xLLM build with a known source revision.";
  check_string("source revision", kExpected.revision, actual->revision, path);
  check_string(
      "xLLM version", kExpected.xllm_version, actual->xllm_version, path);
  check_string(
      "Torch version", kExpected.torch_version, actual->torch_version, path);
  check_string("compiler version",
               kExpected.compiler_version,
               actual->compiler_version,
               path);
  check_string(
      "architecture", kExpected.architecture, actual->architecture, path);
  CHECK_EQ(actual->cxx_standard, kExpected.cxx_standard)
      << "Native library " << path << " uses an incompatible C++ standard.";
  CHECK_EQ(actual->standard_library_version, kExpected.standard_library_version)
      << "Native library " << path << " uses an incompatible standard library.";
  CHECK_EQ(actual->cxx11_abi, kExpected.cxx11_abi)
      << "Native library " << path << " uses an incompatible C++ ABI.";
  CHECK_EQ(actual->device_flags, kExpected.device_flags)
      << "Native library " << path << " uses incompatible device features.";
  CHECK_EQ(actual->pointer_size, kExpected.pointer_size)
      << "Native library " << path << " uses an incompatible pointer size.";
}

}  // namespace

void load_native_libraries() {
  static std::once_flag initialized;
  std::call_once(initialized, [] {
    const char* libraries = std::getenv("XLLM_NATIVE_LIBRARIES");
    if (libraries == nullptr || libraries[0] == '\0') {
      return;
    }
    export_host_symbols();
    std::unordered_set<std::string> loaded;
    std::string_view remaining(libraries);
    while (!remaining.empty()) {
      const size_t separator = remaining.find(':');
      const std::filesystem::path requested(remaining.substr(0, separator));
      CHECK(!requested.empty() && requested.is_absolute())
          << "XLLM_NATIVE_LIBRARIES requires absolute, nonempty paths.";
      std::error_code error;
      const std::filesystem::path path =
          std::filesystem::canonical(requested, error);
      CHECK(!error) << "Cannot resolve native library " << requested << ": "
                    << error.message();
      if (loaded.insert(path.string()).second) {
        void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        CHECK_NE(handle, nullptr)
            << "Cannot load native library " << path << ": " << dlerror();
        check_build_info(handle, path);
        // Intentionally retain the dlopen reference: registries own callbacks,
        // vtables, and objects whose code remains in this library.
        LOG(INFO) << "Loaded native library " << path;
      }
      if (separator == std::string_view::npos) {
        break;
      }
      remaining.remove_prefix(separator + 1);
      CHECK(!remaining.empty())
          << "XLLM_NATIVE_LIBRARIES has an empty trailing path.";
    }
  });
}

}  // namespace xllm

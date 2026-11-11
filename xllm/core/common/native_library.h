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

#pragma once

#include <torch/version.h>

#include <cstdint>
#include <string>

#include "core/common/native_library_build_info.h"

namespace xllm {

// Native extensions use the host's C++ types and must match its exact build.
// This descriptor detects incompatible builds; it is not a security boundary.
struct NativeLibraryBuildInfo {
  uint32_t version;
  uint32_t size;
  const char* revision;
  const char* xllm_version;
  const char* torch_version;
  const char* compiler_version;
  const char* architecture;
  uint64_t cxx_standard;
  uint64_t standard_library_version;
  int32_t cxx11_abi;
  uint32_t device_flags;
  uint32_t pointer_size;
};

namespace detail {

constexpr NativeLibraryBuildInfo native_library_build_info() {
  uint32_t device_flags = 0;
#if defined(USE_NPU)
  device_flags |= 1U << 0;
#endif
#if defined(USE_CUDA)
  device_flags |= 1U << 1;
#endif
#if defined(USE_MLU)
  device_flags |= 1U << 2;
#endif
#if defined(USE_MUSA)
  device_flags |= 1U << 3;
#endif
#if defined(USE_DCU)
  device_flags |= 1U << 4;
#endif
#if defined(USE_ILU)
  device_flags |= 1U << 5;
#endif
#if defined(USE_MACA)
  device_flags |= 1U << 6;
#endif

#if defined(_GLIBCXX_USE_CXX11_ABI)
  constexpr int32_t kCxx11Abi = _GLIBCXX_USE_CXX11_ABI;
#else
  constexpr int32_t kCxx11Abi = -1;
#endif
#if defined(__GLIBCXX__)
  constexpr uint64_t kStandardLibraryVersion = __GLIBCXX__;
#elif defined(_LIBCPP_VERSION)
  constexpr uint64_t kStandardLibraryVersion = _LIBCPP_VERSION;
#else
  constexpr uint64_t kStandardLibraryVersion = 0;
#endif

  return {3,
          sizeof(NativeLibraryBuildInfo),
          NATIVE_LIBRARY_BUILD_REVISION,
          NATIVE_LIBRARY_BUILD_VERSION,
          TORCH_VERSION,
          __VERSION__,
          NATIVE_LIBRARY_BUILD_ARCHITECTURE,
          __cplusplus,
          kStandardLibraryVersion,
          kCxx11Abi,
          device_flags,
          sizeof(void*)};
}

}  // namespace detail

// Loads XLLM_NATIVE_LIBRARIES once for the current process. Each executable or
// host module that can query registries must call this from its startup entry
// point before querying registries or creating objects. Registration may call
// existing register_* APIs, but must not query factories. Loaded libraries stay
// resident until process exit; unloading is unsupported.
void load_native_libraries();

}  // namespace xllm

// Include this once in each extension shared library, alongside normal REGISTER
// macros. Extensions resolve registry symbols from the running xLLM host.
#define DEFINE_NATIVE_LIBRARY()                                  \
  extern "C" __attribute__((visibility("default")))              \
  const ::xllm::NativeLibraryBuildInfo*                          \
  xllm_native_library_build_info() {                             \
    static constexpr ::xllm::NativeLibraryBuildInfo kBuildInfo = \
        ::xllm::detail::native_library_build_info();             \
    return &kBuildInfo;                                          \
  }

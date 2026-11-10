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

#if defined(TEST_NATIVE_MISMATCH) || defined(TEST_NATIVE_VERSION_MISMATCH)
extern "C" __attribute__((visibility("default")))
const xllm::NativeLibraryBuildInfo*
xllm_native_library_build_info() {
  static constexpr xllm::NativeLibraryBuildInfo kBuildInfo = [] {
    xllm::NativeLibraryBuildInfo info =
        xllm::detail::native_library_build_info();
#if defined(TEST_NATIVE_MISMATCH)
    info.revision = "incompatible-source-revision";
#else
    --info.version;
#endif
    return info;
  }();
  return &kBuildInfo;
}
#elif !defined(TEST_NATIVE_NO_DESCRIPTOR)
DEFINE_NATIVE_LIBRARY()
#endif

#if defined(TEST_NATIVE_NO_DESCRIPTOR)
extern "C" int32_t xllm_native_library_valid_dependency_value();

extern "C" __attribute__((visibility("default"))) int32_t
xllm_native_library_force_dependency() {
  return xllm_native_library_valid_dependency_value();
}
#else
extern "C" __attribute__((visibility("default"))) int32_t
xllm_native_library_valid_dependency_value() {
  return 7;
}
#endif

#if defined(TEST_NATIVE_HOST_DEPENDENCY)
extern "C" int32_t xllm_native_library_fixture_host_value();
#endif

extern "C" __attribute__((visibility("default"))) int32_t
xllm_native_library_fixture_value() {
#if defined(TEST_NATIVE_HOST_DEPENDENCY)
  return xllm_native_library_fixture_host_value();
#else
  return 42;
#endif
}

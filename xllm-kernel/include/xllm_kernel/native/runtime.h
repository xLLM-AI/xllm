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

#include <cstdint>
#include <filesystem>

namespace xllm_kernel::native {
// The loaded kernel library owns resources, independent of the host binary.
std::filesystem::path resource_root();
// Call before ACL/device initialization; preserve other vendors after ours.
void configure_resources();
}  // namespace xllm_kernel::native

extern "C" int32_t xllm_kernel_native_abi_version();

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

#include <acl/acl.h>
#include <torch/types.h>

#include <cstdint>

namespace xllm_kernel::native {
// Borrow the caller's current stream. The runtime retains the device/context.
aclrtStream current_stream();
// The caller retains this tensor through enqueue. Torch's NPU allocator owns
// stream ordering and graph-pool lifetime; no global workspace is retained.
torch::Tensor allocate_workspace(uint64_t bytes);
}  // namespace xllm_kernel::native

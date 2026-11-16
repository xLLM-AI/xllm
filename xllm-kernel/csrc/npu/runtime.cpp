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

#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "xllm_kernel/aclnn/launcher.h"
#include "xllm_kernel/native/npu.h"

namespace xllm_kernel::native {
aclrtStream current_stream() {
  return c10_npu::getCurrentNPUStream().stream(false);
}

torch::Tensor allocate_workspace(uint64_t bytes) {
  if (bytes == 0) {
    return {};
  }
  return torch::empty(
      {static_cast<int64_t>(bytes)},
      torch::TensorOptions().device(torch::kPrivateUse1).dtype(torch::kUInt8));
}
}  // namespace xllm_kernel::native

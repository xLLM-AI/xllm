/* Copyright 2025-2026 The xLLM Authors.

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

#include <torch_npu/csrc/aten/CustomFunctions.h>

#include "npu_ops_api.h"
#include "ops_npu/npu_ops.h"

namespace xllm::kernel::npu {

torch::Tensor apply_npu_moe_token_unpermute(
    const torch::Tensor& permuted_tokens,
    const torch::Tensor& sorted_indices,
    const std::optional<torch::Tensor>& probes,
    bool padded_mode,
    c10::OptionalIntArrayRef restore_shape) {
  if (!padded_mode) {
    return at_npu::native::custom_ops::npu_moe_token_unpermute(
        permuted_tokens, sorted_indices, probes.value());
  } else {
    return at_npu::native::custom_ops::npu_moe_token_unpermute(
        permuted_tokens,
        sorted_indices,
        probes.value(),
        padded_mode,
        restore_shape.value());
  }
}

}  // namespace xllm::kernel::npu

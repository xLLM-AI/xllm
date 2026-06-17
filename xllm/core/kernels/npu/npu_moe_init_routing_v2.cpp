/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <torch_npu/csrc/aten/CustomFunctions.h>

#include "npu_ops_api.h"
#include "ops_npu/npu_ops.h"

namespace xllm::kernel::npu {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
apply_npu_moe_init_routing_v2(const torch::Tensor& x,
                              const torch::Tensor& expert_idx,
                              const std::optional<torch::Tensor>& scale,
                              const std::optional<torch::Tensor>& offset,
                              int active_num,
                              int expert_capacity,
                              int expert_num,
                              int drop_pad_mode,
                              int expert_tokens_num_type,
                              bool expert_tokens_num_flag,
                              int quant_mode,
                              torch::IntArrayRef active_expert_range,
                              int row_idx_type) {
  return at_npu::native::custom_ops::npu_moe_init_routing_v2(
      x,
      expert_idx,
      c10::nullopt,
      c10::nullopt,
      active_num,
      expert_capacity,
      expert_num,
      0,
      expert_tokens_num_type,
      expert_tokens_num_flag,
      quant_mode,
      active_expert_range,
      row_idx_type);
}

}  // namespace xllm::kernel::npu

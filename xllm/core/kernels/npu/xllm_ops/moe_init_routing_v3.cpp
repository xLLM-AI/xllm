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

#include <torch/torch.h>

#include <algorithm>
#include <tuple>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
moe_init_routing_v3(const torch::Tensor& x,
                    const torch::Tensor& expert_idx,
                    int64_t active_num,
                    int64_t expert_num,
                    torch::IntArrayRef active_expert_range,
                    int64_t quant_mode) {
  CHECK(x.dim() == 2) << "x must be [tokens, hidden]";
  CHECK(expert_idx.dim() == 2) << "expert_idx must be [tokens, topk]";
  CHECK(active_expert_range.size() == 2)
      << "active_expert_range must contain [start, end]";
  CHECK_GE(active_expert_range[0], 0);
  CHECK_LE(active_expert_range[0], active_expert_range[1]);
  CHECK_LE(active_expert_range[1], expert_num);
  CHECK(quant_mode == -1 || quant_mode == 1) << "quant_mode must be -1 or 1";

  const int64_t token_count = x.size(0) * expert_idx.size(1);
  const int64_t output_tokens =
      active_num > 0 ? std::min(token_count, active_num) : token_count;
  const torch::ScalarType output_dtype =
      quant_mode == -1 ? x.scalar_type() : torch::kInt8;
  torch::Tensor expanded_x =
      torch::empty({output_tokens, x.size(1)}, x.options().dtype(output_dtype));
  torch::Tensor expanded_row_idx =
      torch::empty({token_count}, expert_idx.options().dtype(torch::kInt32));
  torch::Tensor group_list =
      torch::empty({expert_num, 2}, x.options().dtype(torch::kInt64));
  torch::Tensor expanded_scale =
      torch::empty({output_tokens}, x.options().dtype(torch::kFloat32));

  const c10::optional<torch::Tensor> scale = c10::nullopt;
  const c10::optional<torch::Tensor> offset = c10::nullopt;
  constexpr int64_t kDropPadMode = 0;
  constexpr int64_t kExpertCapacity = -1;
  constexpr int64_t kExpertTokensNumTypeKeyValue = 2;
  constexpr bool kExpertTokensNumFlag = true;
  constexpr int64_t kRowIndexTypeGather = 0;
  EXEC_NPU_CMD(aclnnMoeInitRoutingV3,
               x,
               expert_idx,
               scale,
               offset,
               output_tokens,
               kExpertCapacity,
               expert_num,
               kDropPadMode,
               kExpertTokensNumTypeKeyValue,
               kExpertTokensNumFlag,
               quant_mode,
               active_expert_range,
               kRowIndexTypeGather,
               expanded_x,
               expanded_row_idx,
               group_list,
               expanded_scale);
  return std::make_tuple(
      expanded_x, expanded_row_idx, group_list, expanded_scale);
}

}  // namespace xllm::kernel::npu

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

#include <tuple>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {

std::tuple<torch::Tensor, torch::Tensor> moe_grouped_matmul_swiglu_quant(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale,
    const torch::Tensor& x_scale,
    const torch::Tensor& group_list) {
  CHECK(x.dim() == 2) << "x must be [M, K]";
  CHECK(weight.dim() == 3) << "weight must be [E, K, 2N]";
  CHECK(weight_scale.dim() == 2) << "weight_scale must be [E, 2N]";
  CHECK(x_scale.dim() == 1) << "x_scale must be [M]";
  CHECK(group_list.dim() == 2 && group_list.size(1) == 2)
      << "group_list must be [E, 2]";
  CHECK(weight.size(2) % 2 == 0) << "the fused projection width must be even";

  const bool exact_moe_api =
      aclnn::detail::get_op_api_func_addr(
          "aclnnMoeGroupedMatmulSwigluQuantGetWorkspaceSize") != nullptr &&
      aclnn::detail::get_op_api_func_addr("aclnnMoeGroupedMatmulSwigluQuant") !=
          nullptr;
  const int64_t output_width = weight.size(2) / 2;
  torch::Tensor output =
      torch::empty({x.size(0), output_width}, x.options().dtype(torch::kInt8));
  torch::Tensor output_scale =
      torch::empty({x.size(0)}, x.options().dtype(torch::kFloat32));
  const c10::optional<torch::Tensor> bias = c10::nullopt;
  const c10::optional<torch::Tensor> offset = c10::nullopt;
  if (exact_moe_api) {
    const int64_t expert_count = weight.size(0);
    const int64_t input_width = weight.size(1);
    const int64_t fused_width = weight.size(2);
    CHECK(input_width % 16 == 0 && fused_width % 32 == 0)
        << "private NZ weight dimensions must be aligned to 16 and 32";
    const std::vector<int64_t> physical_shape = {
        expert_count, fused_width / 32, input_width / 16, 16, 32};
    const std::vector<int64_t> physical_stride = {
        physical_shape[1] * physical_shape[2] * physical_shape[3] *
            physical_shape[4],
        physical_shape[2] * physical_shape[3] * physical_shape[4],
        physical_shape[3] * physical_shape[4],
        physical_shape[4],
        1};
    torch::Tensor physical_weight = weight.as_strided(
        physical_shape, physical_stride, weight.storage_offset());
    EXEC_NPU_CMD(aclnnMoeGroupedMatmulSwigluQuant,
                 x,
                 physical_weight,
                 weight_scale,
                 x_scale,
                 group_list,
                 output,
                 output_scale);
    return std::make_tuple(output, output_scale);
  }

  torch::Tensor output_offset =
      torch::empty({}, x.options().dtype(torch::kFloat32));
  torch::Tensor cumulative_group_list =
      torch::cumsum(group_list.select(1, 1).to(torch::kInt64), 0);
  EXEC_NPU_CMD(aclnnGroupedMatmulSwigluQuantWeightNZ,
               x,
               weight,
               bias,
               offset,
               weight_scale,
               x_scale,
               cumulative_group_list,
               output,
               output_scale,
               output_offset);
  return std::make_tuple(output, output_scale);
}

}  // namespace xllm::kernel::npu

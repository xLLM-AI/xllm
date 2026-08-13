/* Copyright 2025-2026 The xLLM Authors.

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

#include "npu_ops_api.h"

namespace xllm::kernel::npu {

torch::Tensor group_gemm(const torch::Tensor& x,
                         const torch::Tensor& weight,
                         const std::optional<torch::Tensor>& scale,
                         const std::optional<torch::Tensor>& per_token_scale,
                         const torch::Tensor& group_list,
                         int64_t split_item,
                         int64_t group_type,
                         int64_t group_list_type,
                         std::optional<at::ScalarType> output_dtype) {
  std::vector<torch::Tensor> x_list = {x};
  std::vector<torch::Tensor> weight_list = {weight};
  std::vector<torch::Tensor> scale_storage;
  std::vector<torch::Tensor> per_token_scale_storage;
  std::optional<torch::TensorList> scale_list = std::nullopt;
  if (scale.has_value()) {
    scale_storage.push_back(scale.value());
    scale_list = torch::TensorList(scale_storage);
  }
  std::optional<torch::TensorList> per_token_scale_list = std::nullopt;
  if (per_token_scale.has_value()) {
    per_token_scale_storage.push_back(per_token_scale.value());
    per_token_scale_list = torch::TensorList(per_token_scale_storage);
  }
  auto outputs =
      apply_npu_grouped_matmul(torch::TensorList(x_list),
                               torch::TensorList(weight_list),
                               /*bias=*/std::nullopt,
                               scale_list,
                               /*offset=*/std::nullopt,
                               /*antiquant_scale=*/std::nullopt,
                               /*antiquant_offset=*/std::nullopt,
                               per_token_scale_list,
                               group_list,
                               /*activation_input=*/std::nullopt,
                               /*activation_quant_scale=*/std::nullopt,
                               /*activation_quant_offset=*/std::nullopt,
                               split_item,
                               group_type,
                               group_list_type,
                               /*act_type=*/std::nullopt,
                               /*tuning_config=*/c10::nullopt,
                               output_dtype);
  return outputs.back();
}

}  // namespace xllm::kernel::npu

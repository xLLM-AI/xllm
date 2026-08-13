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

// Python-facing single-tensor wrapper around apply_npu_grouped_matmul.
//
// The native C++ W8A8 FusedMoE path (fused_moe.cpp:1028-1073) builds single-
// element x_list/weight_list and either omits scale_list/per_token_scale_list
// (GEMM1, int8 x int8 -> int32) or supplies single-element lists (GEMM2, int8
// x int8 -> hidden dtype). apply_npu_grouped_matmul converts an omitted list
// into a genuinely empty TensorList (zero tensors) via value_or_empty_tensor_list,
// which is the contract the op-plugin aclnnGroupedMatmulV5
// PerTokenScaleOptional check expects. Calling torch.ops.npu.npu_grouped_matmul
// directly from Python forwards an empty list that the binding coerces into a
// 2-dim placeholder tensor, tripping error 161002 ("PerTokenScaleOptional dim
// num must be 1 ... now is 2"). Routing through this wrapper reproduces the
// C++ argument contract exactly. This is functionally identical to
// xllm::kernel::group_gemm (ops_api.cpp), which itself only builds the same
// TensorLists and forwards to apply_npu_grouped_matmul; it lives in npu_kernels
// (alongside apply_npu_grouped_matmul) to avoid the kernels->xllm_ops->kernels
// dependency cycle that a higher-layer placement would create.
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

  std::optional<torch::TensorList> scale_list = std::nullopt;
  if (scale.has_value()) {
    scale_list = torch::TensorList({scale.value()});
  }
  std::optional<torch::TensorList> per_token_scale_list = std::nullopt;
  if (per_token_scale.has_value()) {
    per_token_scale_list = torch::TensorList({per_token_scale.value()});
  }

  auto outputs = apply_npu_grouped_matmul(
      torch::TensorList(x_list),
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

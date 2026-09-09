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

#include "mlu_ops_api.h"

namespace xllm::kernel::mlu {

torch::Tensor group_gemm(const torch::Tensor& a,
                         const torch::Tensor& b,
                         const torch::Tensor& token_count,
                         torch::Tensor& output,
                         const std::optional<torch::Tensor>& a_scale,
                         const std::optional<torch::Tensor>& b_scale,
                         const std::optional<torch::List<int64_t>>& quant_flag,
                         const int64_t max_dim,
                         const bool trans_a,
                         const bool trans_b,
                         const int64_t a_quant_bit) {
  tmo::torch_api::group_gemm(a,
                             b,
                             token_count,
                             output,
                             /*gather_idx=*/std::nullopt,
                             /*c=*/std::nullopt,
                             /*alpha=*/std::nullopt,
                             /*beta=*/std::nullopt,
                             a_scale,
                             b_scale,
                             /*bias=*/std::nullopt,
                             /*a_calibration=*/std::nullopt,
                             /*b_calibration=*/std::nullopt,
                             quant_flag,
                             /*b_offset=*/std::nullopt,
                             /*tile_config=*/std::nullopt,
                             max_dim,
                             trans_a,
                             trans_b,
                             a_quant_bit);
  return output;
}

}  // namespace xllm::kernel::mlu

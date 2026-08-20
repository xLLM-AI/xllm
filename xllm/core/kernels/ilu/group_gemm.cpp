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

#include "ilu_ops_api.h"

namespace xllm::kernel::ilu {

torch::Tensor group_gemm(torch::Tensor& input,
                         torch::Tensor& weight,
                         torch::Tensor& tokens_per_experts,
                         const std::optional<torch::Tensor>& dst_to_src,
                         torch::Tensor& output) {
  infer::moe_w16a16_group_gemm(
      output,
      input,
      weight,
      tokens_per_experts,
      dst_to_src,
      /*bias=*/std::nullopt,
      /*format=*/"TN",
      /*persistent=*/0,
      /*output_n=*/tokens_per_experts.sum().item<int64_t>());

  return output;
}

}  // namespace xllm::kernel::ilu

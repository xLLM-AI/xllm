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

#include <glog/logging.h>

#include <vector>

#include "dcu_ops_api.h"
#include "grouped_gemm_ck.h"

namespace xllm::kernel::dcu {

torch::Tensor group_gemm(const torch::Tensor& input,
                         const torch::Tensor& weight,
                         const torch::Tensor& token_count,
                         std::optional<torch::Tensor> output_opt) {
  CHECK(input.is_contiguous()) << "input must be contiguous";
  CHECK(weight.is_contiguous()) << "weight must be contiguous";
  CHECK(token_count.is_contiguous()) << "token_count must be contiguous";
  const int64_t N = weight.size(1);  // output feature dim
  const int64_t K = weight.size(2);  // input feature dim
  const int num_experts = static_cast<int>(token_count.size(0));

  // Allocate output if not provided.
  torch::Tensor output;
  if (output_opt.has_value()) {
    output = output_opt.value();
  } else {
    output = torch::empty({input.size(0), N}, input.options());
  }

  // Pull token_count to host.
  auto token_count_cpu = token_count.to(torch::kInt32).cpu();
  const auto* token_count_ptr = token_count_cpu.data_ptr<int32_t>();

  std::vector<torch::Tensor> a_tensors;
  std::vector<torch::Tensor> b_tensors;
  std::vector<torch::Tensor> c_tensors;
  a_tensors.reserve(num_experts);
  b_tensors.reserve(num_experts);
  c_tensors.reserve(num_experts);

  int64_t token_offset = 0;
  for (int32_t e = 0; e < num_experts; ++e) {
    int32_t M_e = token_count_ptr[e];
    if (M_e == 0) continue;

    a_tensors.emplace_back(input.slice(0, token_offset, token_offset + M_e));
    b_tensors.emplace_back(weight[e]);
    c_tensors.emplace_back(output.slice(0, token_offset, token_offset + M_e));

    token_offset += M_e;
  }

  if (a_tensors.empty()) {
    return output;
  }

  aiter::native::ck_grouped_gemm_out(a_tensors, b_tensors, c_tensors);

  return output;
}

}  // namespace xllm::kernel::dcu

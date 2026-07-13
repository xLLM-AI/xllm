/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "kernels/npu/npu_ops_api.h"
#include "kernels/ops_api.h"

namespace xllm {
namespace kernel {
namespace npu {
namespace {

torch::Tensor swiglu_oai_reference(const torch::Tensor& input) {
  const int64_t input_dim = input.dim();
  const int64_t last_dim = input.size(input_dim - 1);
  const int64_t split_size = last_dim / 2;
  torch::Tensor gate =
      input.narrow(input_dim - 1, 0, split_size).clamp_max(7.0);
  torch::Tensor up =
      input.narrow(input_dim - 1, split_size, split_size).clamp(-7.0, 7.0);
  return (up + 1.0) * gate * torch::sigmoid(gate * 1.702);
}

}  // namespace

TEST(NpuActivationTest, SwigluOaiMatchesReferenceFormula) {
  const torch::Tensor input = torch::tensor(
      {{-8.0f, -1.0f, 0.5f, 8.0f}, {2.0f, 9.0f, -9.0f, 3.0f}}, torch::kFloat32);

  const torch::Tensor output = active(input, xllm::kernel::kActModeSwigluOai);
  const torch::Tensor expected = swiglu_oai_reference(input);

  EXPECT_TRUE(torch::allclose(output, expected, /*rtol=*/1e-6, /*atol=*/1e-6));
}

}  // namespace npu
}  // namespace kernel
}  // namespace xllm

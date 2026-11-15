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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "core/kernels/ops_api.h"
#include "core/util/tensor_helper.h"

namespace xllm {
namespace {

class MoeExpandInputMluTest
    : public ::testing::TestWithParam<torch::ScalarType> {};

TEST_P(MoeExpandInputMluTest, ReusesByteBufferWithoutOverwritingPadding) {
  torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);
  constexpr int64_t kHiddenSize = 128;
  torch::Tensor input_cpu = torch::arange(5 * kHiddenSize)
                                .remainder(17)
                                .reshape({5, kHiddenSize})
                                .to(GetParam());
  torch::Tensor indices_cpu = torch::tensor({4, 0, 4, 1, 3, 1}, torch::kInt32);
  torch::Tensor expected =
      input_cpu.index_select(/*dim=*/0, indices_cpu.to(torch::kInt64));

  kernel::MoeExpandInputParams params;
  params.input = input_cpu.to(device);
  params.gather_index = indices_cpu.to(device);
  params.start_expert_id = 0;
  params.expert_size = 2;
  torch::Tensor allocated = kernel::moe_expand_input(params);
  EXPECT_TRUE(torch::equal(allocated.cpu(), expected));

  int64_t row_bytes = kHiddenSize * params.input.element_size();
  int64_t token_count = indices_cpu.numel();
  torch::Tensor storage =
      torch::full({(token_count + 2) * row_bytes},
                  -1,
                  params.input.options().dtype(torch::kChar));
  torch::Tensor output_bytes =
      storage.slice(/*dim=*/0, row_bytes, (token_count + 1) * row_bytes);
  params.output =
      view_as_dtype(output_bytes, GetParam()).view({token_count, kHiddenSize});

  torch::Tensor output = kernel::moe_expand_input(params);
  EXPECT_EQ(output.data_ptr(), output_bytes.data_ptr());
  EXPECT_TRUE(torch::equal(output.cpu(), expected));

  params.input.add_(1);
  torch::Tensor reused = kernel::moe_expand_input(params);
  EXPECT_EQ(reused.data_ptr(), output_bytes.data_ptr());
  EXPECT_TRUE(torch::equal(reused.cpu(), expected + 1));
  EXPECT_TRUE(torch::equal(allocated.cpu(), expected));

  torch::Tensor storage_cpu = storage.cpu();
  EXPECT_TRUE(
      storage_cpu.slice(/*dim=*/0, 0, row_bytes).eq(-1).all().item<bool>());
  EXPECT_TRUE(storage_cpu.slice(/*dim=*/0, (token_count + 1) * row_bytes)
                  .eq(-1)
                  .all()
                  .item<bool>());
}

INSTANTIATE_TEST_SUITE_P(FloatingPoint,
                         MoeExpandInputMluTest,
                         ::testing::Values(torch::kFloat32,
                                           torch::kFloat16,
                                           torch::kBFloat16));

}  // namespace
}  // namespace xllm

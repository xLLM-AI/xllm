/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include <vector>

#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

namespace xllm::kernel::npu::tilelang {
namespace {

class TileLangSpecVerifyTokenUpdateTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { torch_npu::init_npu("npu:0"); }
  static void TearDownTestSuite() { torch_npu::finalize_npu(); }
};

TEST_F(TileLangSpecVerifyTokenUpdateTest, ReportsCompiledWidths) {
  EXPECT_TRUE(has_spec_verify_token_update_specialization(4));
  EXPECT_TRUE(has_spec_verify_token_update_specialization(5));
  EXPECT_TRUE(has_spec_verify_token_update_specialization(6));
  EXPECT_FALSE(has_spec_verify_token_update_specialization(3));
  EXPECT_FALSE(has_spec_verify_token_update_specialization(8));
}

TEST_F(TileLangSpecVerifyTokenUpdateTest, PacksBaseAndDraftTokens) {
  const torch::Device device("npu:0");
  const torch::TensorOptions i32 =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  const torch::TensorOptions i64 =
      torch::TensorOptions().dtype(torch::kInt64).device(device);
  const torch::Tensor base_token = torch::tensor({42}, i32);
  std::vector<torch::Tensor> draft_tokens;
  draft_tokens.reserve(5);
  for (int64_t token = 1; token <= 5; ++token) {
    draft_tokens.emplace_back(torch::tensor({token}, i64));
  }
  torch::Tensor persistent_tokens = torch::full({8}, -1, i32);

  spec_verify_token_update(base_token, draft_tokens, persistent_tokens);

  EXPECT_TRUE(
      torch::equal(persistent_tokens.cpu(),
                   torch::tensor({42, 1, 2, 3, 4, 5, 0, 0}, torch::kInt32)));
}

TEST_F(TileLangSpecVerifyTokenUpdateTest, PacksMtp3AndZerosTail) {
  const torch::Device device("npu:0");
  const auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
  const auto i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
  const auto base_token = torch::tensor({42}, i32);
  std::vector<torch::Tensor> draft_tokens = {
      torch::tensor({1}, i64),
      torch::tensor({2}, i64),
      torch::tensor({3}, i64),
  };
  auto persistent_tokens = torch::full({8}, -1, i32);

  spec_verify_token_update(base_token, draft_tokens, persistent_tokens);

  EXPECT_TRUE(
      torch::equal(persistent_tokens.cpu(),
                   torch::tensor({42, 1, 2, 3, 0, 0, 0, 0}, torch::kInt32)));
}

TEST_F(TileLangSpecVerifyTokenUpdateTest, PacksMtp4AndZerosTail) {
  const torch::Device device("npu:0");
  const auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
  const auto i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
  const auto base_token = torch::tensor({42}, i32);
  std::vector<torch::Tensor> draft_tokens = {
      torch::tensor({1}, i64),
      torch::tensor({2}, i64),
      torch::tensor({3}, i64),
      torch::tensor({4}, i64),
  };
  auto persistent_tokens = torch::full({8}, -1, i32);

  spec_verify_token_update(base_token, draft_tokens, persistent_tokens);

  EXPECT_TRUE(
      torch::equal(persistent_tokens.cpu(),
                   torch::tensor({42, 1, 2, 3, 4, 0, 0, 0}, torch::kInt32)));
}

}  // namespace
}  // namespace xllm::kernel::npu::tilelang

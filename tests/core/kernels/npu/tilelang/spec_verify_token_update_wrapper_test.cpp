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

  const torch::Device device_{"npu:0"};
  const torch::TensorOptions i32_ =
      torch::TensorOptions().dtype(torch::kInt32).device(device_);
  const torch::TensorOptions i64_ =
      torch::TensorOptions().dtype(torch::kInt64).device(device_);
};

TEST_F(TileLangSpecVerifyTokenUpdateTest, ReportsCompiledWidths) {
  EXPECT_TRUE(has_spec_verify_token_update_specialization(4));
  EXPECT_TRUE(has_spec_verify_token_update_specialization(5));
  EXPECT_TRUE(has_spec_verify_token_update_specialization(6));
  EXPECT_FALSE(has_spec_verify_token_update_specialization(3));
  EXPECT_FALSE(has_spec_verify_token_update_specialization(8));
}

TEST_F(TileLangSpecVerifyTokenUpdateTest, PacksSupportedWidthsAndZerosTail) {
  struct TokenPackingCase {
    int64_t spec_width;
    std::vector<int32_t> base_tokens;
    std::vector<int64_t> draft_tokens;
    std::vector<int32_t> expected;
  };
  const std::vector<TokenPackingCase> test_cases = {
      {/*spec_width=*/4,
       /*base_tokens=*/{42, -1, -1, -1},
       /*draft_tokens=*/{1, 2, 3},
       /*expected=*/{42, 1, 2, 3, 0, 0, 0, 0}},
      {/*spec_width=*/5,
       /*base_tokens=*/{42, -1, -1, -1, -1},
       /*draft_tokens=*/{1, 2, 3, 4},
       /*expected=*/{42, 1, 2, 3, 4, 0, 0, 0}},
      {/*spec_width=*/6,
       /*base_tokens=*/{42, -1, -1, -1, -1, -1},
       /*draft_tokens=*/{1, 2, 3, 4, 5},
       /*expected=*/{42, 1, 2, 3, 4, 5, 0, 0}},
  };

  for (const TokenPackingCase& test_case : test_cases) {
    SCOPED_TRACE(::testing::Message() << "spec_width=" << test_case.spec_width);
    std::vector<torch::Tensor> draft_tokens;
    draft_tokens.reserve(test_case.draft_tokens.size());
    for (const int64_t token : test_case.draft_tokens) {
      draft_tokens.emplace_back(torch::tensor({token}, i64_));
    }
    torch::Tensor persistent_tokens = torch::full({8}, -1, i32_);

    spec_verify_token_update(torch::tensor(test_case.base_tokens, i32_),
                             draft_tokens,
                             persistent_tokens,
                             test_case.spec_width);

    EXPECT_TRUE(torch::equal(
        persistent_tokens.cpu(),
        torch::tensor(test_case.expected, torch::dtype(torch::kInt32))));
  }
}

TEST_F(TileLangSpecVerifyTokenUpdateTest, PacksMultipleSequencesRowMajor) {
  const torch::Tensor base_tokens =
      torch::tensor({42, -1, -2, -3, 84, -1, -2, -3}, i32_);
  std::vector<torch::Tensor> draft_tokens = {
      torch::tensor({1, 11}, i64_),
      torch::tensor({2, 12}, i64_),
      torch::tensor({3, 13}, i64_),
  };
  torch::Tensor persistent_tokens = torch::full({8}, -1, i32_);

  spec_verify_token_update(
      base_tokens, draft_tokens, persistent_tokens, /*spec_width=*/4);

  EXPECT_TRUE(torch::equal(
      persistent_tokens.cpu(),
      torch::tensor({42, 1, 2, 3, 84, 11, 12, 13}, torch::kInt32)));
}

}  // namespace
}  // namespace xllm::kernel::npu::tilelang

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

#include "layers/common/dcp_attention_merge.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>
#include <limits>

namespace xllm::layer {
namespace {

TEST(DcpAttentionMergeTest, WeightsPartialOutputsByNaturalLogLse) {
  torch::Tensor partial_outputs =
      torch::tensor({2.0f, 4.0f, 6.0f, 8.0f}).reshape({2, 1, 1, 1, 2});
  torch::Tensor partial_lse =
      torch::tensor({std::log(2.0f), std::log(6.0f)}).reshape({2, 1, 1, 1});

  const DcpAttentionResult result =
      merge_dcp_attention_shards(partial_outputs, partial_lse);

  EXPECT_TRUE(torch::allclose(
      result.output, torch::tensor({5.0f, 7.0f}).reshape({1, 1, 1, 2})));
  EXPECT_TRUE(torch::allclose(
      result.lse, torch::tensor({std::log(8.0f)}).reshape({1, 1, 1})));
}

TEST(DcpAttentionMergeTest, IgnoresEmptyShardAndPreservesFiniteShard) {
  const float negative_infinity = -std::numeric_limits<float>::infinity();
  torch::Tensor partial_outputs =
      torch::tensor({0.0f, 0.0f, 3.0f, 9.0f}).reshape({2, 1, 1, 1, 2});
  torch::Tensor partial_lse =
      torch::tensor({negative_infinity, std::log(4.0f)}).reshape({2, 1, 1, 1});

  const DcpAttentionResult result =
      merge_dcp_attention_shards(partial_outputs, partial_lse);

  EXPECT_TRUE(torch::equal(result.output,
                           torch::tensor({3.0f, 9.0f}).reshape({1, 1, 1, 2})));
  EXPECT_TRUE(torch::allclose(
      result.lse, torch::tensor({std::log(4.0f)}).reshape({1, 1, 1})));
}

TEST(DcpAttentionMergeTest, ReturnsZeroAndNegativeInfinityWhenAllShardsEmpty) {
  const float negative_infinity = -std::numeric_limits<float>::infinity();
  torch::Tensor partial_outputs = torch::ones({2, 1, 1, 2, 3});
  torch::Tensor partial_lse = torch::full({2, 1, 2, 1}, negative_infinity);

  const DcpAttentionResult result =
      merge_dcp_attention_shards(partial_outputs, partial_lse);

  EXPECT_TRUE(torch::equal(result.output, torch::zeros({1, 1, 2, 3})));
  EXPECT_TRUE(torch::isneginf(result.lse).all().item<bool>());
}

}  // namespace
}  // namespace xllm::layer

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

#include "core/framework/sampling/logits_utils.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>

namespace xllm {
namespace {

TEST(LogitsUtilsTest, MinPUsesRelativeProbabilityAndPreservesDisabledRows) {
  torch::Tensor logits =
      torch::tensor({{2.0F, 1.0F, 0.0F}, {2.0F, 1.0F, 0.0F}});
  apply_min_p(logits, torch::tensor({0.5F, 0.0F}));
  EXPECT_EQ(logits[0][0].item<float>(), 2.0F);
  EXPECT_TRUE(std::isinf(logits[0][1].item<float>()));
  EXPECT_TRUE(std::isinf(logits[0][2].item<float>()));
  EXPECT_TRUE(torch::equal(logits[1], torch::tensor({2.0F, 1.0F, 0.0F})));
}

TEST(LogitsUtilsTest, TopKPreservesTiesAndDisabledRows) {
  torch::Tensor logits =
      torch::tensor({{3.0F, 3.0F, 1.0F}, {3.0F, 2.0F, 1.0F}});
  apply_top_k_top_p(logits, {}, torch::tensor({1, 0}, torch::kLong), {});
  EXPECT_EQ(logits[0][0].item<float>(), 3.0F);
  EXPECT_EQ(logits[0][1].item<float>(), 3.0F);
  EXPECT_TRUE(std::isinf(logits[0][2].item<float>()));
  EXPECT_TRUE(torch::equal(logits[1], torch::tensor({3.0F, 2.0F, 1.0F})));
}

TEST(LogitsUtilsTest, CombinedTopKTopPFiltersCpuLogits) {
  torch::Tensor logits = torch::tensor({{3.0F, 2.0F, 1.0F, 0.0F}});
  apply_top_k_top_p(
      logits, {}, torch::tensor({3}, torch::kLong), torch::tensor({0.8F}));
  EXPECT_TRUE(std::isfinite(logits[0][0].item<float>()));
  EXPECT_TRUE(std::isfinite(logits[0][1].item<float>()));
  EXPECT_TRUE(std::isinf(logits[0][2].item<float>()));
  EXPECT_TRUE(std::isinf(logits[0][3].item<float>()));
}

TEST(LogitsUtilsTest, TopPRemovesProbabilityAtTheCutoff) {
  torch::Tensor logits = torch::zeros({1, 4});
  apply_top_k_top_p(logits, {}, {}, torch::tensor({0.5F}));
  EXPECT_EQ(logits.isfinite().sum().item<int64_t>(), 2);
}

TEST(LogitsUtilsTest, PaddingDoesNotOverwriteFrequencyOrPresencePenalty) {
  torch::Tensor logits = torch::tensor({{5.0F, 3.0F, 1.0F}});
  apply_frequency_presence_penalties(logits,
                                     torch::tensor({{0, 1, 0}}, torch::kLong),
                                     torch::tensor({{2, 1, 0}}, torch::kInt),
                                     torch::tensor({0.5F}),
                                     torch::tensor({1.0F}));
  EXPECT_TRUE(torch::allclose(logits, torch::tensor({{3.0F, 1.5F, 1.0F}})));
}

TEST(LogitsUtilsTest, RepetitionPenaltyIgnoresPaddedTokenZero) {
  torch::Tensor logits = torch::tensor({{4.0F, -4.0F, 2.0F}});
  apply_repetition_penalties(logits,
                             torch::tensor({{1, 0}}, torch::kLong),
                             torch::tensor({2.0F}),
                             torch::tensor({1}, torch::kInt));
  EXPECT_TRUE(torch::equal(logits, torch::tensor({{4.0F, -8.0F, 2.0F}})));
}

TEST(LogitsUtilsTest, RepetitionPenaltyPreservesBlockedTokens) {
  torch::Tensor logits = torch::tensor({{4.0F, -INFINITY, -2.0F}});
  apply_repetition_penalties(logits,
                             torch::tensor({{1, 2}}, torch::kLong),
                             torch::tensor({2.0F}),
                             torch::tensor({2}, torch::kInt));
  EXPECT_FALSE(logits.isnan().any().item<bool>());
  EXPECT_EQ(logits[0][1].item<float>(), -INFINITY);
  EXPECT_EQ(logits[0][2].item<float>(), -4.0F);
}

}  // namespace
}  // namespace xllm

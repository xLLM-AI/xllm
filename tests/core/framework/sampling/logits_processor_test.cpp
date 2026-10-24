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

#include "core/framework/sampling/logits_processor.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>

#include "tests/core/framework/sampling/sampling_test_utils.h"

namespace xllm {
namespace {

TEST(LogitsProcessorTest, MinPUsesRelativeProbabilityAndPreservesDisabledRows) {
  torch::Tensor logits =
      torch::tensor({{2.0F, 1.0F, 0.0F}, {2.0F, 1.0F, 0.0F}});
  SamplingParameters params =
      testutil::make_sampling_params(2, SamplingMode::RANDOM);
  params.min_p = torch::tensor({0.5F, 0.0F});
  process_logits(logits, params);
  EXPECT_EQ(logits[0][0].item<float>(), 2.0F);
  EXPECT_TRUE(std::isinf(logits[0][1].item<float>()));
  EXPECT_TRUE(std::isinf(logits[0][2].item<float>()));
  EXPECT_TRUE(torch::equal(logits[1], torch::tensor({2.0F, 1.0F, 0.0F})));
}

TEST(LogitsProcessorTest, TopKPreservesTiesAndDisabledRows) {
  torch::Tensor logits =
      torch::tensor({{3.0F, 3.0F, 1.0F}, {3.0F, 2.0F, 1.0F}});
  SamplingParameters params =
      testutil::make_sampling_params(2, SamplingMode::RANDOM);
  params.top_k = torch::tensor({1, 0}, torch::kLong);
  process_logits(logits, params);
  EXPECT_EQ(logits[0][0].item<float>(), 3.0F);
  EXPECT_EQ(logits[0][1].item<float>(), 3.0F);
  EXPECT_TRUE(std::isinf(logits[0][2].item<float>()));
  EXPECT_TRUE(torch::equal(logits[1], torch::tensor({3.0F, 2.0F, 1.0F})));
}

TEST(LogitsProcessorTest, CombinedTopKTopPFiltersCpuLogits) {
  torch::Tensor logits = torch::tensor({{3.0F, 2.0F, 1.0F, 0.0F}});
  SamplingParameters params =
      testutil::make_sampling_params(1, SamplingMode::RANDOM);
  params.top_k = torch::tensor({3}, torch::kLong);
  params.top_p = torch::tensor({0.8F});
  process_logits(logits, params);
  EXPECT_TRUE(std::isfinite(logits[0][0].item<float>()));
  EXPECT_TRUE(std::isfinite(logits[0][1].item<float>()));
  EXPECT_TRUE(std::isinf(logits[0][2].item<float>()));
  EXPECT_TRUE(std::isinf(logits[0][3].item<float>()));
}

TEST(LogitsProcessorTest, TopPRemovesProbabilityAtTheCutoff) {
  torch::Tensor logits = torch::zeros({1, 4});
  SamplingParameters params =
      testutil::make_sampling_params(1, SamplingMode::RANDOM);
  params.top_p = torch::tensor({0.5F});
  process_logits(logits, params);
  EXPECT_EQ(logits.isfinite().sum().item<int64_t>(), 2);
}

TEST(LogitsProcessorTest, PaddingDoesNotOverwriteFrequencyOrPresencePenalty) {
  torch::Tensor logits = torch::tensor({{5.0F, 3.0F, 1.0F}});
  SamplingParameters params =
      testutil::make_sampling_params(1, SamplingMode::RANDOM);
  params.unique_token_ids = torch::tensor({{0, 1, 0}}, torch::kLong);
  params.unique_token_counts = torch::tensor({{2, 1, 0}}, torch::kInt);
  params.frequency_penalties = torch::tensor({0.5F});
  params.presence_penalties = torch::tensor({1.0F});
  apply_sampling_penalties(logits, params);
  EXPECT_TRUE(torch::allclose(logits, torch::tensor({{3.0F, 1.5F, 1.0F}})));
}

TEST(LogitsProcessorTest, RepetitionPenaltyIgnoresPaddedTokenZero) {
  torch::Tensor logits = torch::tensor({{4.0F, -4.0F, 2.0F}});
  SamplingParameters params =
      testutil::make_sampling_params(1, SamplingMode::RANDOM);
  params.unique_token_ids = torch::tensor({{1, 0}}, torch::kLong);
  params.repetition_penalties = torch::tensor({2.0F});
  params.unique_token_ids_lens = torch::tensor({1}, torch::kInt);
  apply_sampling_penalties(logits, params);
  EXPECT_TRUE(torch::equal(logits, torch::tensor({{4.0F, -8.0F, 2.0F}})));
}

TEST(LogitsProcessorTest, RepetitionPenaltyPreservesBlockedTokens) {
  torch::Tensor logits = torch::tensor({{4.0F, -INFINITY, -2.0F}});
  SamplingParameters params =
      testutil::make_sampling_params(1, SamplingMode::RANDOM);
  params.unique_token_ids = torch::tensor({{1, 2}}, torch::kLong);
  params.repetition_penalties = torch::tensor({2.0F});
  params.unique_token_ids_lens = torch::tensor({2}, torch::kInt);
  apply_sampling_penalties(logits, params);
  EXPECT_FALSE(logits.isnan().any().item<bool>());
  EXPECT_EQ(logits[0][1].item<float>(), -INFINITY);
  EXPECT_EQ(logits[0][2].item<float>(), -4.0F);
}

}  // namespace
}  // namespace xllm

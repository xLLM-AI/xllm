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

#include "core/framework/sampling/logits_processor.h"
#include "core/framework/sampling/sampler.h"
#include "core/framework/speculative/rejection_sampler.h"
#include "tests/core/framework/sampling/sampling_test_utils.h"

namespace xllm {
namespace {

SamplingParameters make_sampling_params(int64_t rows, bool random) {
  return testutil::make_sampling_params(
      rows, random ? SamplingMode::RANDOM : SamplingMode::GREEDY);
}

TEST(SamplerControlsTest, GreedyHonorsBiasAndAllowedTokenMask) {
  auto params = make_sampling_params(1, false);
  params.logits_bias() = torch::tensor({{-INFINITY, 10.0F, 0.0F}});
  torch::Tensor logits = torch::tensor({{100.0F, 1.0F, 2.0F}});
  EXPECT_EQ(sample(logits, params).next_tokens[0].item<int64_t>(), 1);
}

TEST(SamplerControlsTest, ReportsRawLogprobsBeforeConstraints) {
  auto params = make_sampling_params(1, false);
  params.logprobs() = true;
  params.max_top_logprobs() = 2;
  params.logits_bias() = torch::tensor({{-INFINITY, 0.0F, 10.0F}});
  torch::Tensor logits = torch::tensor({{4.0F, 2.0F, 1.0F}});
  const torch::Tensor expected = torch::log_softmax(logits, -1);
  const auto result = sample(logits, params);
  EXPECT_EQ(result.next_tokens[0].item<int64_t>(), 2);
  EXPECT_NEAR(
      result.logprobs[0].item<float>(), expected[0][2].item<float>(), 1e-6);
  EXPECT_EQ(result.top_tokens[0][0].item<int64_t>(), 0);
  EXPECT_NEAR(result.top_logprobs[0][0].item<float>(),
              expected[0][0].item<float>(),
              1e-6);
}

TEST(SamplerControlsTest, AppliesRepetitionBeforePresenceAndFrequency) {
  auto params = make_sampling_params(1, false);
  params.unique_token_ids() = torch::tensor({{0}}, torch::kLong);
  params.unique_token_counts() = torch::tensor({{1}}, torch::kInt);
  params.unique_token_ids_lens() = torch::tensor({1}, torch::kInt);
  params.repetition_penalties() = torch::tensor({2.0F});
  params.frequency_penalties() = torch::tensor({1.0F});
  params.presence_penalties() = torch::tensor({1.0F});
  torch::Tensor logits = torch::tensor({{6.0F, 1.5F}});
  EXPECT_EQ(sample(logits, params).next_tokens[0].item<int64_t>(), 1);
  EXPECT_NEAR(logits[0][0].item<float>(), 1.0F, 1e-6);
}

TEST(SamplerControlsTest, SeededSamplingIsIndependentOfBatchPosition) {
  auto one = make_sampling_params(1, true);
  one.seeds() = torch::tensor({12345}, torch::kLong);
  one.seed_offsets() = torch::tensor({7}, torch::kLong);
  auto batch = make_sampling_params(2, true);
  batch.seeds() = torch::tensor({6789, 12345}, torch::kLong);
  batch.seed_offsets() = torch::tensor({2, 7}, torch::kLong);
  torch::Tensor logits = torch::arange(32, torch::kFloat).unsqueeze(0) / 20;
  torch::Tensor batch_logits =
      torch::cat({torch::zeros_like(logits), logits}, 0);
  const auto alone = sample(logits, one);
  const auto together = sample(batch_logits, batch);
  EXPECT_EQ(alone.next_tokens[0].item<int64_t>(),
            together.next_tokens[1].item<int64_t>());
}

TEST(SamplerControlsTest, SeededSamplingDoesNotChangeGlobalGenerator) {
  auto params = make_sampling_params(1, true);
  params.seeds() = torch::tensor({42}, torch::kLong);
  params.seed_offsets() = torch::tensor({0}, torch::kLong);
  torch::Tensor logits = torch::zeros({1, 64});
  const auto& generator =
      torch::globalContext().defaultGenerator(torch::Device(torch::kCPU));
  const torch::Tensor before = generator.get_state().clone();
  const auto first = sample(logits, params);
  const auto second = sample(logits, params);
  EXPECT_TRUE(torch::equal(first.next_tokens, second.next_tokens));
  EXPECT_TRUE(torch::equal(before, generator.get_state()));
}

TEST(SamplerControlsTest, OverlapAdvancesBadWordSuffixAndSeedByAcceptedTokens) {
  auto params = make_sampling_params(2, false);
  params.bad_words() = torch::tensor({{{1, 2, 3}}, {{1, 2, 3}}}, torch::kLong);
  params.bad_word_suffix() = torch::tensor({{-1, 1}, {-1, -1}}, torch::kLong);
  params.seeds() = torch::tensor({42, -1}, torch::kLong);
  params.seed_offsets() = torch::tensor({7, 10}, torch::kLong);
  advance_sampling_history(
      params, torch::tensor({{1, 2, -1}, {2, -1, -1}}, torch::kLong), {1, 0});
  torch::Tensor logits =
      torch::tensor({{0.F, 0.F, 0.F, 100.F}, {0.F, 0.F, 0.F, 100.F}});
  const auto result = sample(logits, params);
  EXPECT_NE(result.next_tokens[0].item<int64_t>(), 3);
  EXPECT_NE(result.next_tokens[1].item<int64_t>(), 3);
  EXPECT_EQ(params.seed_offsets()[0].item<int64_t>(), 8);
  EXPECT_EQ(params.seed_offsets()[1].item<int64_t>(), 12);
}

TEST(SamplerControlsTest, OverlapLeavesNewRequestsUnchanged) {
  auto params = make_sampling_params(2, false);
  params.bad_words() = torch::tensor({{{2, 3}}, {{2, 3}}}, torch::kLong);
  params.bad_word_suffix() = torch::full({2, 1}, -1, torch::kLong);
  advance_sampling_history(params, torch::tensor({2}, torch::kLong), {-1, 0});
  torch::Tensor logits =
      torch::tensor({{0.F, 0.F, 0.F, 100.F}, {0.F, 0.F, 0.F, 100.F}});
  const auto result = sample(logits, params);
  EXPECT_EQ(result.next_tokens[0].item<int64_t>(), 3);
  EXPECT_NE(result.next_tokens[1].item<int64_t>(), 3);
}

TEST(RejectionSamplerControlsTest, BadWordsAdvanceAcrossDraftAndBonusTokens) {
  auto params = make_sampling_params(1, false);
  params.bad_words() = torch::tensor({{{1, 2, 3}, {-1, -1, 4}}}, torch::kLong);
  params.bad_word_suffix() = torch::tensor({{-1, 1}}, torch::kLong);
  torch::Tensor logits = torch::tensor({{0.F, 0.F, 10.F, 0.F, 100.F},
                                        {0.F, 0.F, 0.F, 10.F, 100.F},
                                        {0.F, 0.F, 0.F, 10.F, 100.F}});
  auto result = rejection_sample(
      params, DraftProposal(torch::tensor({{2, 3}}, torch::kLong)), logits);
  EXPECT_EQ(result.next_tokens[0][0].item<int64_t>(), 2);
  EXPECT_NE(result.next_tokens[0][1].item<int64_t>(), 3);
  EXPECT_EQ(result.next_tokens[0][2].item<int64_t>(), -1);
  params.bad_words() = torch::tensor({{{1, 2, 3}}}, torch::kLong);
  params.bad_word_suffix() = torch::full({1, 2}, -1, torch::kLong);
  logits = torch::tensor(
      {{0.F, 10.F, 0.F, 0.F}, {0.F, 0.F, 10.F, 0.F}, {0.F, 0.F, 0.F, 10.F}});
  result = rejection_sample(
      params, DraftProposal(torch::tensor({{1, 2}}, torch::kLong)), logits);
  EXPECT_EQ(result.next_tokens[0][0].item<int64_t>(), 1);
  EXPECT_EQ(result.next_tokens[0][1].item<int64_t>(), 2);
  EXPECT_NE(result.next_tokens[0][2].item<int64_t>(), 3);
}

TEST(RejectionSamplerControlsTest, SeededVerifyMatchesOrdinaryTokenOffsets) {
  for (const int64_t seed : {-2, 0, 42}) {
    auto params = make_sampling_params(1, true);
    params.seeds() = torch::tensor({seed}, torch::kLong);
    params.seed_offsets() = torch::tensor({7}, torch::kLong);
    torch::Tensor logits = torch::zeros({3, 16});
    std::vector<int64_t> expected;
    expected.reserve(3);
    for (int64_t column = 0; column < 3; ++column) {
      auto ordinary = params;
      ordinary.seed_offsets() = params.seed_offsets() + column;
      torch::Tensor row = logits[column].unsqueeze(0).clone();
      expected.emplace_back(
          sample(row, ordinary).next_tokens[0].item<int64_t>());
    }
    const auto proposal =
        torch::tensor({{expected[0], expected[1]}}, torch::kLong);
    for (const int64_t num_draft_tokens : {0, 2}) {
      const auto result = rejection_sample(
          params,
          DraftProposal(proposal.narrow(1, 0, num_draft_tokens)),
          logits.narrow(0, 0, num_draft_tokens + 1));
      EXPECT_TRUE(torch::equal(result.next_tokens,
                               torch::tensor(expected, torch::kLong)
                                   .unsqueeze(0)
                                   .narrow(1, 0, num_draft_tokens + 1)));
      EXPECT_EQ(result.num_accepted_tokens.scalar_type(), torch::kInt32);
      EXPECT_TRUE(
          torch::equal(result.num_accepted_tokens,
                       torch::full({1, 1}, num_draft_tokens, torch::kInt32)));
    }
  }
}

TEST(RejectionSamplerControlsTest, CandidatePenaltiesCountDraftPrefix) {
  auto params = make_sampling_params(1, false);
  params.unique_token_ids() = torch::tensor({{0}}, torch::kLong);
  params.unique_token_counts() = torch::tensor({{0}}, torch::kInt);
  params.unique_token_ids_lens() = torch::tensor({1}, torch::kInt);
  params.frequency_penalties() = torch::tensor({2.F});
  params.presence_penalties() = torch::tensor({1.F});
  params.repetition_penalties() = torch::tensor({2.F});
  torch::Tensor logits =
      torch::tensor({{0.F, 6.F, 2.F}, {0.F, 6.F, 2.F}, {0.F, 6.F, 2.F}});
  const auto result = rejection_sample(
      params, DraftProposal(torch::tensor({{1, 1}}, torch::kLong)), logits);
  EXPECT_EQ(result.next_tokens[0][0].item<int64_t>(), 1);
  EXPECT_EQ(result.next_tokens[0][1].item<int64_t>(), 2);
  EXPECT_EQ(result.next_tokens[0][2].item<int64_t>(), -1);
}

TEST(RejectionSamplerControlsTest, ReportsRawLogprobsBeforeTokenConstraints) {
  auto params = make_sampling_params(1, false);
  params.logprobs() = true;
  params.max_top_logprobs() = 2;
  params.logits_bias() = torch::tensor({{-INFINITY, 0.F, 100.F}});
  torch::Tensor logits = torch::tensor({{4.F, 2.F, 1.F}, {4.F, 2.F, 1.F}});
  const torch::Tensor expected = torch::log_softmax(logits[0], -1);
  const auto result = rejection_sample(
      params, DraftProposal(torch::tensor({{2}}, torch::kLong)), logits);
  EXPECT_NEAR(
      result.logprobs[0][0].item<float>(), expected[2].item<float>(), 1e-6);
  EXPECT_EQ(result.top_tokens[0][0][0].item<int64_t>(), 0);
}

TEST(SamplerControlsTest, MinPRemovesLowProbabilityCandidates) {
  auto params = make_sampling_params(1, true);
  params.seeds() = torch::tensor({42}, torch::kLong);
  params.seed_offsets() = torch::tensor({0}, torch::kLong);
  params.min_p() = torch::tensor({0.5F});
  params.return_logits() = true;
  torch::Tensor logits = torch::tensor({{2.0F, 1.0F, 0.0F}});
  const auto result = sample(logits, params);
  EXPECT_EQ(result.next_tokens[0].item<int64_t>(), 0);
  const auto probs = torch::softmax(result.draft_logits, -1);
  EXPECT_EQ(probs[0][1].item<float>(), 0);
  EXPECT_EQ(probs[0][2].item<float>(), 0);
}

TEST(RejectionSamplerControlsTest, MinimumLengthIncludesBonusRow) {
  const auto expected = torch::tensor({{0, 0, 0, 0},
                                       {1, 0, 0, 0},
                                       {1, 1, 0, 0},
                                       {1, 1, 1, 2},
                                       {1, 1, 1, 1},
                                       {1, 1, 1, 1},
                                       {0, 0, 0, 0}},
                                      torch::kLong);
  const auto remaining = torch::tensor({0, 1, 2, 3, 4, 5, 5}, torch::kLong);
  const auto stop_ids = torch::tensor(
      {{0, -1}, {0, -1}, {0, -1}, {0, 2}, {0, 2}, {0, 2}, {-1, -1}},
      torch::kLong);
  for (const auto mode :
       {SamplingMode::GREEDY, SamplingMode::RANDOM, SamplingMode::MIXED}) {
    auto params = testutil::make_sampling_params(7, mode);
    params.min_token_positions() = torch::full({7}, 100, torch::kLong);
    params.min_tokens_remaining() = remaining.clone();
    params.min_token_ids() = stop_ids.clone();
    if (mode != SamplingMode::GREEDY) {
      params.top_k() = torch::ones({7}, torch::kLong);
    }
    if (mode == SamplingMode::MIXED) {
      params.do_sample() = torch::tensor(
          {false, true, false, true, false, true, false}, torch::kBool);
    }
    auto logits = torch::tensor({{10.0F, 5.0F, 0.0F}}).repeat({28, 1});
    logits.view({7, 4, 3}).narrow(0, 3, 3).select(2, 2).fill_(20);
    const auto fixed = rejection_sample(
        params, DraftProposal(expected.narrow(1, 0, 3).clone()), logits);
    EXPECT_TRUE(torch::equal(fixed.next_tokens, expected));
    EXPECT_TRUE(torch::equal(params.min_tokens_remaining(), remaining));
    EXPECT_TRUE(torch::equal(params.min_token_ids(), stop_ids));
  }
}

TEST(RejectionSamplerControlsTest, MinimumLengthRejectsEarlyDraftEos) {
  for (const auto mode :
       {SamplingMode::GREEDY, SamplingMode::RANDOM, SamplingMode::MIXED}) {
    auto params = testutil::make_sampling_params(2, mode);
    params.min_tokens_remaining() = torch::tensor({1, 0}, torch::kLong);
    params.min_token_ids() = torch::tensor({{0, -1}, {0, -1}}, torch::kLong);
    if (mode != SamplingMode::GREEDY) {
      params.top_k() = torch::ones({2}, torch::kLong);
    }
    if (mode == SamplingMode::MIXED) {
      params.do_sample() = torch::tensor({false, true}, torch::kBool);
    }
    auto logits = torch::tensor({{10.0F, 5.0F, 0.0F}}).repeat({6, 1});
    const auto result = rejection_sample(
        params, DraftProposal(torch::zeros({2, 2}, torch::kLong)), logits);
    EXPECT_TRUE(
        torch::equal(result.next_tokens,
                     torch::tensor({{1, -1, -1}, {0, 0, 0}}, torch::kLong)));
  }
}

TEST(RejectionSamplerControlsTest, MinimumLengthWithoutDraftsUsesBonusRow) {
  for (const auto mode :
       {SamplingMode::GREEDY, SamplingMode::RANDOM, SamplingMode::MIXED}) {
    auto params = testutil::make_sampling_params(2, mode);
    params.min_tokens_remaining() = torch::tensor({1, 0}, torch::kLong);
    params.min_token_ids() = torch::tensor({{0, -1}, {0, -1}}, torch::kLong);
    if (mode != SamplingMode::GREEDY) {
      params.top_k() = torch::ones({2}, torch::kLong);
    }
    if (mode == SamplingMode::MIXED) {
      params.do_sample() = torch::tensor({false, true}, torch::kBool);
    }
    auto logits = torch::tensor({{10.0F, 5.0F, 0.0F}}).repeat({2, 1});
    const auto result = rejection_sample(
        params, DraftProposal(torch::empty({2, 0}, torch::kLong)), logits);
    EXPECT_TRUE(torch::equal(result.next_tokens,
                             torch::tensor({{1}, {0}}, torch::kLong)));
  }
}

}  // namespace
}  // namespace xllm

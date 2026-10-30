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
#include "core/framework/speculative/spec_verify.h"
#include "tests/core/framework/sampling/sampling_test_utils.h"

namespace xllm {
namespace {

SamplingParameters make_sampling_params(int64_t rows, bool random) {
  return testutil::make_sampling_params(
      rows, random ? SamplingMode::RANDOM : SamplingMode::GREEDY);
}

TEST(SamplerControlsTest, GreedyHonorsBiasAndAllowedTokenMask) {
  auto params = make_sampling_params(1, false);
  params.logits_bias = torch::tensor({{-INFINITY, 10.0F, 0.0F}});
  torch::Tensor logits = torch::tensor({{100.0F, 1.0F, 2.0F}});
  EXPECT_EQ(sample(logits, params).next_tokens[0].item<int64_t>(), 1);
}

TEST(SamplerControlsTest, ReportsRawLogprobsBeforeConstraints) {
  auto params = make_sampling_params(1, false);
  params.logprobs = true;
  params.max_top_logprobs = 2;
  params.logits_bias = torch::tensor({{-INFINITY, 0.0F, 10.0F}});
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
  params.unique_token_ids = torch::tensor({{0}}, torch::kLong);
  params.unique_token_counts = torch::tensor({{1}}, torch::kInt);
  params.unique_token_ids_lens = torch::tensor({1}, torch::kInt);
  params.repetition_penalties = torch::tensor({2.0F});
  params.frequency_penalties = torch::tensor({1.0F});
  params.presence_penalties = torch::tensor({1.0F});
  torch::Tensor logits = torch::tensor({{6.0F, 1.5F}});
  EXPECT_EQ(sample(logits, params).next_tokens[0].item<int64_t>(), 1);
  EXPECT_NEAR(logits[0][0].item<float>(), 1.0F, 1e-6);
}

TEST(SamplerControlsTest, SeededSamplingIsIndependentOfBatchPosition) {
  auto one = make_sampling_params(1, true);
  one.seeds = torch::tensor({12345}, torch::kLong);
  one.seed_offsets = torch::tensor({7}, torch::kLong);
  auto batch = make_sampling_params(2, true);
  batch.seeds = torch::tensor({6789, 12345}, torch::kLong);
  batch.seed_offsets = torch::tensor({2, 7}, torch::kLong);
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
  params.seeds = torch::tensor({42}, torch::kLong);
  params.seed_offsets = torch::tensor({0}, torch::kLong);
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
  params.bad_words = torch::tensor({{{1, 2, 3}}, {{1, 2, 3}}}, torch::kLong);
  params.bad_word_suffix = torch::tensor({{-1, 1}, {-1, -1}}, torch::kLong);
  params.seeds = torch::tensor({42, -1}, torch::kLong);
  params.seed_offsets = torch::tensor({7, 10}, torch::kLong);
  advance_sampling_history(
      params, torch::tensor({{1, 2, -1}, {2, -1, -1}}, torch::kLong), {1, 0});
  torch::Tensor logits =
      torch::tensor({{0.F, 0.F, 0.F, 100.F}, {0.F, 0.F, 0.F, 100.F}});
  const auto result = sample(logits, params);
  EXPECT_NE(result.next_tokens[0].item<int64_t>(), 3);
  EXPECT_NE(result.next_tokens[1].item<int64_t>(), 3);
  EXPECT_EQ(params.seed_offsets[0].item<int64_t>(), 8);
  EXPECT_EQ(params.seed_offsets[1].item<int64_t>(), 12);
}

TEST(SamplerControlsTest, OverlapLeavesNewRequestsUnchanged) {
  auto params = make_sampling_params(2, false);
  params.bad_words = torch::tensor({{{2, 3}}, {{2, 3}}}, torch::kLong);
  params.bad_word_suffix = torch::full({2, 1}, -1, torch::kLong);
  advance_sampling_history(params, torch::tensor({2}, torch::kLong), {-1, 0});
  torch::Tensor logits =
      torch::tensor({{0.F, 0.F, 0.F, 100.F}, {0.F, 0.F, 0.F, 100.F}});
  const auto result = sample(logits, params);
  EXPECT_EQ(result.next_tokens[0].item<int64_t>(), 3);
  EXPECT_NE(result.next_tokens[1].item<int64_t>(), 3);
}

TEST(RejectionSamplerControlsTest, BadWordsAdvanceAcrossDraftAndBonusTokens) {
  auto params = make_sampling_params(1, false);
  params.bad_words = torch::tensor({{{1, 2, 3}, {-1, -1, 4}}}, torch::kLong);
  params.bad_word_suffix = torch::tensor({{-1, 1}}, torch::kLong);
  torch::Tensor logits = torch::tensor({{0.F, 0.F, 10.F, 0.F, 100.F},
                                        {0.F, 0.F, 0.F, 10.F, 100.F},
                                        {0.F, 0.F, 0.F, 10.F, 100.F}});
  auto result = rejection_sample(
      params, DraftProposal(torch::tensor({{2, 3}}, torch::kLong)), logits);
  EXPECT_EQ(result.next_tokens[0][0].item<int64_t>(), 2);
  EXPECT_NE(result.next_tokens[0][1].item<int64_t>(), 3);
  EXPECT_EQ(result.next_tokens[0][2].item<int64_t>(), -1);
  params.bad_words = torch::tensor({{{1, 2, 3}}}, torch::kLong);
  params.bad_word_suffix = torch::full({1, 2}, -1, torch::kLong);
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
    params.seeds = torch::tensor({seed}, torch::kLong);
    params.seed_offsets = torch::tensor({7}, torch::kLong);
    torch::Tensor logits = torch::zeros({3, 16});
    std::vector<int64_t> expected;
    expected.reserve(3);
    for (int64_t column = 0; column < 3; ++column) {
      auto ordinary = params;
      ordinary.seed_offsets = params.seed_offsets + column;
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
  params.unique_token_ids = torch::tensor({{0}}, torch::kLong);
  params.unique_token_counts = torch::tensor({{0}}, torch::kInt);
  params.unique_token_ids_lens = torch::tensor({1}, torch::kInt);
  params.frequency_penalties = torch::tensor({2.F});
  params.presence_penalties = torch::tensor({1.F});
  params.repetition_penalties = torch::tensor({2.F});
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
  params.logprobs = true;
  params.max_top_logprobs = 2;
  params.logits_bias = torch::tensor({{-INFINITY, 0.F, 100.F}});
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
  params.seeds = torch::tensor({42}, torch::kLong);
  params.seed_offsets = torch::tensor({0}, torch::kLong);
  params.min_p = torch::tensor({0.5F});
  params.return_logits = true;
  torch::Tensor logits = torch::tensor({{2.0F, 1.0F, 0.0F}});
  const auto result = sample(logits, params);
  EXPECT_EQ(result.next_tokens[0].item<int64_t>(), 0);
  const auto probs = torch::softmax(result.draft_logits, -1);
  EXPECT_EQ(probs[0][1].item<float>(), 0);
  EXPECT_EQ(probs[0][2].item<float>(), 0);
}

TEST(RaggedRejectionSamplerTest, MapsMixedRequestParametersOnPackedRows) {
  auto requests = make_sampling_params(3, false);
  requests.sampling_mode = SamplingMode::MIXED;
  requests.do_sample = torch::tensor({false, true, true}, torch::kBool);
  requests.temperatures = torch::tensor({0.0F, 0.5F, 2.0F});
  const torch::Tensor owners = torch::tensor({2, 2, 0, 1, 1, -1}, torch::kInt);
  torch::Tensor logits = torch::tensor({{0.0F, 1.0F, 2.0F}}).repeat({6, 1});
  torch::Tensor reference = logits.clone();
  reference[5].zero_();
  auto expanded = make_sampling_params(6, false);
  expanded.sampling_mode = SamplingMode::MIXED;
  expanded.do_sample =
      torch::tensor({true, true, false, true, true, false}, torch::kBool);
  expanded.temperatures = torch::tensor({2.0F, 2.0F, 0.0F, 0.5F, 0.5F, 0.0F});
  torch::manual_seed(681);
  const auto expected = sample(reference, expanded).next_tokens;
  torch::manual_seed(681);
  const auto actual =
      spec_verify::rejection_sample_ragged(logits, requests, owners);
  EXPECT_TRUE(torch::equal(actual, expected));
  EXPECT_TRUE(torch::equal(logits, reference));
  EXPECT_EQ(actual[2].item<int64_t>(), 2);
  EXPECT_EQ(actual[5].item<int64_t>(), 0);
  EXPECT_TRUE(
      torch::equal(requests.temperatures, torch::tensor({0.0F, 0.5F, 2.0F})));
}

TEST(RaggedRejectionSamplerTest, ReusesNormalTopKTopPAndMinPTransforms) {
  auto requests = make_sampling_params(2, true);
  requests.temperatures = torch::tensor({0.5F, 1.5F});
  requests.top_k = torch::tensor({1, 3}, torch::kLong);
  requests.top_p = torch::tensor({1.0F, 0.7F});
  requests.min_p = torch::tensor({0.0F, 0.2F});
  const torch::Tensor owners = torch::tensor({1, 0, 1, 0}, torch::kInt);
  torch::Tensor logits = torch::tensor({{2.0F, 1.0F, 0.0F}}).repeat({4, 1});
  torch::Tensor reference = logits.clone();
  auto expanded = make_sampling_params(4, true);
  expanded.temperatures = torch::tensor({1.5F, 0.5F, 1.5F, 0.5F});
  expanded.top_k = torch::tensor({3, 1, 3, 1}, torch::kLong);
  expanded.top_p = torch::tensor({0.7F, 1.0F, 0.7F, 1.0F});
  expanded.min_p = torch::tensor({0.2F, 0.0F, 0.2F, 0.0F});
  torch::manual_seed(91);
  const auto expected = sample(reference, expanded).next_tokens;
  torch::manual_seed(91);
  const auto actual =
      spec_verify::rejection_sample_ragged(logits, requests, owners);
  EXPECT_TRUE(torch::equal(actual, expected));
  EXPECT_TRUE(torch::equal(logits, reference));
}

TEST(RaggedRejectionSamplerTest, PreservesTargetAndResidualDistributions) {
  torch::manual_seed(403);
  constexpr int64_t kTrials = 20000;
  auto requests = make_sampling_params(1, true);
  requests.temperatures = torch::tensor({0.5F});
  // At temperature .5 these logits yield p=(.2,.3,.5), not raw softmax.
  const torch::Tensor probabilities = torch::tensor({0.2F, 0.3F, 0.5F});
  torch::Tensor logits = (probabilities.log() * 0.5F).repeat({kTrials, 1});
  const auto sampled = spec_verify::rejection_sample_ragged(
      logits, requests, torch::zeros({kTrials}, torch::kInt));
  for (int64_t token = 0; token < 3; ++token) {
    EXPECT_NEAR(sampled.eq(token).to(torch::kFloat32).mean().item<float>(),
                probabilities[token].item<float>(),
                0.015);
  }
  // For deterministic proposal d=1, matching accepts with probability p(d).
  // A mismatch supplies the residual p(x)/(1-p(d)) without a second draw.
  const auto rejected = sampled.masked_select(sampled.ne(1));
  EXPECT_NEAR(
      rejected.eq(0).to(torch::kFloat32).mean().item<float>(), 0.2 / 0.7, 0.02);
  EXPECT_NEAR(
      rejected.eq(2).to(torch::kFloat32).mean().item<float>(), 0.5 / 0.7, 0.02);
}

TEST(RaggedRejectionSamplerTest, DrawsFreshRandomValuesOnEveryInvocation) {
  auto requests = make_sampling_params(1, true);
  torch::Tensor first_logits = torch::zeros({256, 8});
  torch::Tensor second_logits = first_logits.clone();
  const auto owners = torch::zeros({256}, torch::kInt);
  torch::manual_seed(593);
  const auto first =
      spec_verify::rejection_sample_ragged(first_logits, requests, owners);
  const auto second =
      spec_verify::rejection_sample_ragged(second_logits, requests, owners);
  EXPECT_FALSE(torch::equal(first, second));
}

TEST(RaggedRejectionSamplerTest, IgnoresNonfiniteUncommittedPaddingLogits) {
  auto requests = make_sampling_params(1, true);
  requests.temperatures = torch::tensor({0.5F});
  torch::Tensor logits =
      torch::tensor({{0.0F, 20.0F, 0.0F}, {NAN, INFINITY, -INFINITY}});
  const auto sampled = spec_verify::rejection_sample_ragged(
      logits, requests, torch::tensor({0, -1}, torch::kInt));
  EXPECT_EQ(sampled[0].item<int64_t>(), 1);
  EXPECT_TRUE(torch::isfinite(logits[1]).all().item<bool>());
  EXPECT_TRUE(logits[1].eq(0).all().item<bool>());
}

TEST(RaggedRejectionSamplerTest, MinimumLengthUsesOwnersAndCandidateOffsets) {
  const auto owners = torch::tensor({1, 0, 1, 0, -1}, torch::kInt);
  const auto columns = torch::tensor({0, 0, 2, 1, -1}, torch::kInt);
  for (const auto mode :
       {SamplingMode::GREEDY, SamplingMode::RANDOM, SamplingMode::MIXED}) {
    auto requests = testutil::make_sampling_params(2, mode);
    requests.min_token_positions = torch::tensor({8, 9}, torch::kLong);
    requests.min_tokens_remaining = torch::tensor({1, 2}, torch::kLong);
    requests.min_token_ids = torch::tensor({{0, -1}, {0, -1}}, torch::kLong);
    if (mode != SamplingMode::GREEDY) {
      requests.top_k = torch::ones({2}, torch::kLong);
    }
    if (mode == SamplingMode::MIXED) {
      requests.do_sample = torch::tensor({false, true}, torch::kBool);
    }
    auto logits = torch::tensor({{10.0F, 5.0F, 0.0F}}).repeat({5, 1});
    const auto sampled =
        spec_verify::rejection_sample_ragged(logits, requests, owners, columns);
    EXPECT_TRUE(torch::equal(sampled.narrow(0, 0, 4),
                             torch::tensor({1, 1, 0, 0}, torch::kLong)));
    EXPECT_TRUE(logits[4].eq(0).all().item<bool>());
    EXPECT_TRUE(torch::equal(requests.min_tokens_remaining,
                             torch::tensor({1, 2}, torch::kLong)));
  }
}

TEST(RejectionSamplerControlsTest,
     MinimumLengthMatchesRaggedOffsetsIncludingBonus) {
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
  const auto owners = torch::arange(7, torch::kInt).repeat_interleave(4);
  const auto columns = torch::arange(4, torch::kInt).repeat({7});
  for (const auto mode :
       {SamplingMode::GREEDY, SamplingMode::RANDOM, SamplingMode::MIXED}) {
    auto params = testutil::make_sampling_params(7, mode);
    params.min_token_positions = torch::full({7}, 100, torch::kLong);
    params.min_tokens_remaining = remaining.clone();
    params.min_token_ids = stop_ids.clone();
    if (mode != SamplingMode::GREEDY) {
      params.top_k = torch::ones({7}, torch::kLong);
    }
    if (mode == SamplingMode::MIXED) {
      params.do_sample = torch::tensor(
          {false, true, false, true, false, true, false}, torch::kBool);
    }
    auto logits = torch::tensor({{10.0F, 5.0F, 0.0F}}).repeat({28, 1});
    logits.view({7, 4, 3}).narrow(0, 3, 3).select(2, 2).fill_(20);
    auto ragged_logits = logits.clone();
    const auto fixed = rejection_sample(
        params, DraftProposal(expected.narrow(1, 0, 3).clone()), logits);
    const auto ragged = spec_verify::rejection_sample_ragged(
        ragged_logits, params, owners, columns);
    EXPECT_TRUE(torch::equal(fixed.next_tokens, expected));
    EXPECT_TRUE(torch::equal(ragged.view({7, 4}), expected));
    EXPECT_TRUE(torch::equal(params.min_tokens_remaining, remaining));
    EXPECT_TRUE(torch::equal(params.min_token_ids, stop_ids));
  }
}

TEST(RejectionSamplerControlsTest, MinimumLengthRejectsEarlyDraftEos) {
  for (const auto mode :
       {SamplingMode::GREEDY, SamplingMode::RANDOM, SamplingMode::MIXED}) {
    auto params = testutil::make_sampling_params(2, mode);
    params.min_tokens_remaining = torch::tensor({1, 0}, torch::kLong);
    params.min_token_ids = torch::tensor({{0, -1}, {0, -1}}, torch::kLong);
    if (mode != SamplingMode::GREEDY) {
      params.top_k = torch::ones({2}, torch::kLong);
    }
    if (mode == SamplingMode::MIXED) {
      params.do_sample = torch::tensor({false, true}, torch::kBool);
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
    params.min_tokens_remaining = torch::tensor({1, 0}, torch::kLong);
    params.min_token_ids = torch::tensor({{0, -1}, {0, -1}}, torch::kLong);
    if (mode != SamplingMode::GREEDY) {
      params.top_k = torch::ones({2}, torch::kLong);
    }
    if (mode == SamplingMode::MIXED) {
      params.do_sample = torch::tensor({false, true}, torch::kBool);
    }
    auto logits = torch::tensor({{10.0F, 5.0F, 0.0F}}).repeat({2, 1});
    const auto result = rejection_sample(
        params, DraftProposal(torch::empty({2, 0}, torch::kLong)), logits);
    EXPECT_TRUE(torch::equal(result.next_tokens,
                             torch::tensor({{1}, {0}}, torch::kLong)));
  }
}

TEST(RaggedRejectionSamplerTest, ClippingPreservesTwoTokenJointDistribution) {
  torch::manual_seed(1059);
  constexpr int64_t kTrials = 6000;
  const torch::Tensor first_probs = torch::tensor({0.2F, 0.3F, 0.5F});
  const torch::Tensor transitions = torch::tensor(
      {{0.6F, 0.3F, 0.1F}, {0.1F, 0.2F, 0.7F}, {0.3F, 0.5F, 0.2F}});
  const torch::Tensor expected = first_probs.unsqueeze(1) * transitions;
  for (const int64_t width : {5, 7}) {
    for (const int64_t budget : {int64_t{0}, int64_t{2}, width}) {
      SCOPED_TRACE(width);
      SCOPED_TRACE(budget);
      auto request = make_sampling_params(1, true);
      request.temperatures = torch::tensor({0.5F});
      // The deterministic proposal is token 1 at every position. Each row
      // after the anchor therefore conditions on a speculative token 1.
      torch::Tensor logits =
          transitions[1].log().repeat({kTrials, budget + 1, 1}) * 0.5F;
      logits.select(1, 0).copy_(first_probs.log() * 0.5F);
      logits = logits.view({-1, 3});
      const auto sampled =
          spec_verify::rejection_sample_ragged(
              logits,
              request,
              torch::zeros({kTrials * (budget + 1)}, torch::kInt))
              .view({kTrials, budget + 1});
      const auto first = sampled.select(1, 0);
      // After a rejection (or zero-budget bonus), ordinary next-step target
      // sampling must condition on the actually committed first token.
      torch::Tensor continuation =
          transitions.index_select(0, first).log() * 0.5F;
      auto continuation_params = make_sampling_params(kTrials, true);
      continuation_params.temperatures = torch::full({kTrials}, 0.5F);
      torch::Tensor second =
          sample(continuation, continuation_params).next_tokens;
      if (budget > 0) {
        second = torch::where(first.eq(1), sampled.select(1, 1), second);
      }
      const torch::Tensor joint_ids = first * 3 + second;
      for (int64_t token = 0; token < 9; ++token) {
        EXPECT_NEAR(
            joint_ids.eq(token).to(torch::kFloat32).mean().item<float>(),
            expected.flatten()[token].item<float>(),
            0.02);
      }
    }
  }
}

}  // namespace
}  // namespace xllm

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

#include "core/framework/sampling/rejection_sampler.h"
#include "core/framework/sampling/sampler.h"

namespace xllm {
namespace {

SamplingParameters make_sampling_params(int64_t rows, bool random) {
  SamplingParameters params;
  params.selected_token_idxes = torch::arange(rows, torch::kInt);
  params.sample_idxes = torch::arange(rows, torch::kInt);
  params.do_sample = torch::full({rows}, random, torch::kBool);
  params.all_greedy_sample = !random;
  params.all_random_sample = random;
  return params;
}

TEST(SamplerControlsTest, GreedyHonorsBiasAndAllowedTokenMask) {
  auto params = make_sampling_params(1, false);
  params.logits_bias = torch::tensor({{-INFINITY, 10.0F, 0.0F}});
  torch::Tensor logits = torch::tensor({{100.0F, 1.0F, 2.0F}});
  Sampler sampler;
  EXPECT_EQ(sampler.forward(logits, params).next_tokens[0].item<int64_t>(), 1);
}

TEST(SamplerControlsTest, ReportsRawLogprobsBeforeConstraints) {
  auto params = make_sampling_params(1, false);
  params.logprobs = true;
  params.max_top_logprobs = 2;
  params.logits_bias = torch::tensor({{-INFINITY, 0.0F, 10.0F}});
  torch::Tensor logits = torch::tensor({{4.0F, 2.0F, 1.0F}});
  const torch::Tensor expected = torch::log_softmax(logits, -1);
  Sampler sampler;
  const auto result = sampler.forward(logits, params);
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
  Sampler sampler;
  EXPECT_EQ(sampler.forward(logits, params).next_tokens[0].item<int64_t>(), 1);
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
  Sampler sampler;
  const auto alone = sampler.forward(logits, one);
  const auto together = sampler.forward(batch_logits, batch);
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
  Sampler sampler;
  const auto first = sampler.forward(logits, params);
  const auto second = sampler.forward(logits, params);
  EXPECT_TRUE(torch::equal(first.next_tokens, second.next_tokens));
  EXPECT_TRUE(torch::equal(before, generator.get_state()));
}

TEST(SamplerControlsTest, MinPRemovesLowProbabilityCandidates) {
  auto params = make_sampling_params(1, true);
  params.seeds = torch::tensor({42}, torch::kLong);
  params.seed_offsets = torch::tensor({0}, torch::kLong);
  params.min_p = torch::tensor({0.5F});
  torch::Tensor logits = torch::tensor({{2.0F, 1.0F, 0.0F}});
  Sampler sampler;
  const auto result = sampler.forward(logits, params);
  EXPECT_EQ(result.next_tokens[0].item<int64_t>(), 0);
  EXPECT_EQ(result.probs[0][1].item<float>(), 0);
  EXPECT_EQ(result.probs[0][2].item<float>(), 0);
}

TEST(SamplerControlsTest, MtpMinimumEndsInsideVerifyWindow) {
  auto params = make_sampling_params(4, false);
  params.sampling_positions = torch::tensor({9, 9, 9, 9}, torch::kLong);
  params.min_tokens_remaining = torch::tensor({2, 2, 2, 2}, torch::kLong);
  params.min_token_mask = torch::tensor({{false, true, false}}).repeat({4, 1});
  params.logits_bias = torch::tensor({{0.0F, 0.0F, -INFINITY}}).repeat({4, 1});
  auto actual = sampling_params_for_positions(
      params, torch::tensor({9, 10, 11, 12}, torch::kLong));
  torch::Tensor logits = torch::tensor({{1.0F, 5.0F, 100.0F}}).repeat({4, 1});
  Sampler sampler;
  EXPECT_TRUE(torch::equal(sampler.forward(logits, actual).next_tokens,
                           torch::tensor({0, 0, 1, 1}, torch::kLong)));
  EXPECT_EQ(params.min_tokens_remaining[3].item<int64_t>(), 2);
}

TEST(SamplerControlsTest, MtpOverlapRepairCorrectsMinimumAndSeedOffset) {
  auto params = make_sampling_params(2, true);
  params.sampling_positions = torch::tensor({20, 20}, torch::kLong);
  params.min_tokens_remaining = torch::tensor({-1, -1}, torch::kLong);
  params.seed_offsets = torch::tensor(
      {(int64_t{3} << 32) + 8, (int64_t{3} << 32) + 8}, torch::kLong);
  params.seeds = torch::tensor({42, 42}, torch::kLong);
  params.selected_token_idxes = torch::tensor({1, 3}, torch::kInt);
  const auto actual = sampling_params_for_positions(
      params, torch::tensor({0, 17, 0, 18}, torch::kLong));
  EXPECT_TRUE(torch::equal(actual.min_tokens_remaining,
                           torch::tensor({2, 1}, torch::kLong)));
  EXPECT_EQ(actual.seed_offsets[0].item<int64_t>(), (int64_t{3} << 32) + 5);
  EXPECT_EQ(actual.seed_offsets[1].item<int64_t>(), (int64_t{3} << 32) + 6);
  EXPECT_TRUE(torch::equal(params.sampling_positions,
                           torch::tensor({20, 20}, torch::kLong)));
}

TEST(SamplerControlsTest, MtpSeededWindowMatchesIndividualTargetPositions) {
  auto params = make_sampling_params(4, true);
  params.seeds = torch::tensor({42, 42, 42, 42}, torch::kLong);
  params.seed_offsets = torch::tensor({8, 8, 8, 8}, torch::kLong);
  params.sampling_positions = torch::tensor({20, 20, 20, 20}, torch::kLong);
  auto actual = sampling_params_for_positions(
      params, torch::tensor({17, 18, 19, 20}, torch::kLong));
  torch::Tensor logits = torch::zeros({4, 64});
  Sampler sampler;
  const auto window = sampler.forward(logits, actual);
  for (int64_t row = 0; row < 4; ++row) {
    auto single = make_sampling_params(1, true);
    single.seeds = torch::tensor({42}, torch::kLong);
    single.seed_offsets = torch::tensor({5 + row}, torch::kLong);
    torch::Tensor single_logits = logits[row].unsqueeze(0).clone();
    EXPECT_EQ(
        window.next_tokens[row].item<int64_t>(),
        sampler.forward(single_logits, single).next_tokens[0].item<int64_t>());
  }
  EXPECT_TRUE(torch::equal(window.seed_offsets,
                           torch::tensor({5, 6, 7, 8}, torch::kLong)));
}

TEST(SamplerControlsTest, SeededMtpRejectionIsBatchIndependentAndPrivate) {
  const auto& global =
      torch::globalContext().defaultGenerator(torch::Device(torch::kCPU));
  const torch::Tensor before = global.get_state().clone();
  const torch::Tensor logits = torch::zeros({2, 5, 16});
  const torch::Tensor draft_ids = torch::zeros({2, 4}, torch::kLong);
  const torch::Tensor bonus = torch::full({2, 1}, 15, torch::kLong);
  const torch::Tensor seeds = torch::tensor({7, 42}, torch::kLong);
  const torch::Tensor offsets = torch::tensor({3, 9}, torch::kLong);
  const RejectionSampler batch(
      torch::tensor({true, true}), true, false, false, 0);
  const RejectionSampler one(torch::tensor({true}), true, false, false, 0);
  for (bool dense : {false, true}) {
    std::optional<torch::Tensor> proposal_probs;
    if (dense) {
      proposal_probs = torch::full({2, 4, 16}, 1.0F / 16);
    }
    const DraftProposal proposal(draft_ids, proposal_probs);
    const auto together =
        batch.forward(proposal, logits, bonus, true, seeds, offsets);
    const auto repeat =
        batch.forward(proposal, logits, bonus, true, seeds, offsets);
    const DraftProposal single(
        draft_ids.narrow(0, 1, 1),
        dense ? std::optional<torch::Tensor>(proposal_probs->narrow(0, 1, 1))
              : std::nullopt);
    const auto alone = one.forward(single,
                                   logits.narrow(0, 1, 1),
                                   bonus.narrow(0, 1, 1),
                                   true,
                                   seeds.narrow(0, 1, 1),
                                   offsets.narrow(0, 1, 1));
    EXPECT_TRUE(torch::equal(together.next_tokens, repeat.next_tokens));
    EXPECT_TRUE(torch::equal(together.next_tokens[1], alone.next_tokens[0]));
    EXPECT_GE(alone.next_tokens[0][0].item<int64_t>(), 0);
    EXPECT_TRUE(torch::equal(before, global.get_state()));
  }
}

TEST(SamplerControlsTest, SeededMtpRecoveryExcludesImpossibleTokens) {
  const RejectionSampler sampler(torch::tensor({true}), true, false, false, 0);
  const DraftProposal proposal(torch::tensor({{0, 0}}, torch::kLong),
                               std::nullopt);
  const torch::Tensor logits = torch::tensor({{{-INFINITY, 0.0F, -INFINITY},
                                               {-INFINITY, 0.0F, -INFINITY},
                                               {-INFINITY, 0.0F, -INFINITY}}});
  const auto result = sampler.forward(proposal,
                                      logits,
                                      torch::tensor({{1}}, torch::kLong),
                                      true,
                                      torch::tensor({42}, torch::kLong),
                                      torch::tensor({0}, torch::kLong));
  EXPECT_TRUE(torch::equal(result.next_tokens,
                           torch::tensor({{1, -1, -1}}, torch::kLong)));
}

}  // namespace
}  // namespace xllm

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

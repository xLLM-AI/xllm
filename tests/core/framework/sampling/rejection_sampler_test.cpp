/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "core/framework/speculative/rejection_sampler.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch/types.h>

#if defined(USE_NPU)
#include <torch_npu/csrc/aten/NPUGeneratorImpl.h>
#endif

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <tuple>

#include "core/framework/sampling/npu_sampling_policy.h"
#include "core/framework/sampling/sampler.h"
#include "tests/core/framework/sampling/sampling_test_utils.h"

namespace xllm {

using testutil::flatten_rows;
using testutil::get_test_device;
using testutil::get_test_options;
using testutil::make_rejection_params;
using testutil::make_sampling_params;
using testutil::random_sample_from_probs;

TEST(RejectionSamplerTest, Basic) {
  const auto options = get_test_options(torch::kFloat32);

  // Each position drafts a one-hot q token; the target agrees at positions
  // 0-1 (p/q = 1, always accepted) and gives zero mass at position 2
  // (always rejected, recovery deterministic from the one-hot residual).
  const auto draft_token_ids =
      torch::tensor({{1, 2, 3}}, options.dtype(torch::kInt64));
  // shape: [1, 3, 5]
  auto draft_probs = torch::tensor({{0.0f, 1.0f, 0.0f, 0.0f, 0.0f},
                                    {0.0f, 0.0f, 1.0f, 0.0f, 0.0f},
                                    {0.0f, 0.0f, 0.0f, 1.0f, 0.0f}},
                                   options)
                         .reshape({1, 3, 5});
  // shape: [1, 3, 5]; the last row doubles as the bonus distribution.
  auto target_probs = torch::tensor({{0.0f, 1.0f, 0.0f, 0.0f, 0.0f},
                                     {0.0f, 0.0f, 1.0f, 0.0f, 0.0f},
                                     {1.0f, 0.0f, 0.0f, 0.0f, 0.0f}},
                                    options)
                          .reshape({1, 3, 5});

  DraftProposal draft_proposal =
      DraftProposal(draft_token_ids, draft_probs.log());
  auto masked_output = random_sample_from_probs(
      draft_proposal, target_probs, /*uniform_seed=*/1);
  auto desired_masked_output =
      torch::tensor({{1, 2, 0, -1}}, options.dtype(torch::kInt64));
  EXPECT_TRUE(torch::allclose(masked_output, desired_masked_output));
}

TEST(RejectionSamplerTest, GreedyDraftUsesDeltaProposalDistribution) {
  const auto options = get_test_options(torch::kFloat32);

  // Greedy draft q is a delta at the drafted token. The target agrees at
  // position 0 (p/q = 1, always accepted) and disagrees at position 1
  // (p/q = 0, always rejected; recovery deterministic from the one-hot
  // residual). The last target row doubles as the bonus distribution.
  const auto draft_token_ids =
      torch::tensor({{0, 1, 2, 1}}, options.dtype(torch::kInt64));

  const auto target_probs = torch::tensor({{{1.0f, 0.0f, 0.0f},
                                            {0.0f, 0.0f, 1.0f},
                                            {1.0f, 0.0f, 0.0f},
                                            {0.0f, 1.0f, 0.0f}}},
                                          options);
  DraftProposal draft_proposal = DraftProposal(draft_token_ids);

  auto masked_output = random_sample_from_probs(
      draft_proposal, target_probs, /*uniform_seed=*/1);
  auto desired_masked_output =
      torch::tensor({{0, 2, -1, -1, -1}}, options.dtype(torch::kInt64));
  EXPECT_TRUE(torch::allclose(masked_output, desired_masked_output));
}

TEST(RejectionSamplerTest, Greedy) {
  const auto options = get_test_options(torch::kFloat32);

  int64_t batch_size = 2;
  int64_t n_speculative_tokens = 3;
  int64_t vocab_size = 4;
  int64_t n_bonus_tokens = 1;

  torch::Tensor full_logits = torch::randn(
      {batch_size, n_speculative_tokens + n_bonus_tokens, vocab_size}, options);
  const auto draft_token_ids =
      full_logits.argmax(/*dim=*/-1)
          .slice(/*dim=*/-1, /*start=*/0, /*end=*/n_speculative_tokens);

  SamplingParameters rejection_params = make_rejection_params();
  SampleOutput output = rejection_sample(rejection_params,
                                         DraftProposal(draft_token_ids),
                                         flatten_rows(full_logits));

  EXPECT_TRUE(
      torch::allclose(output.next_tokens, full_logits.argmax(/*dim=*/-1)));
  // Full-width acceptance, not the clamped gather index of width - 1.
  EXPECT_TRUE(
      torch::equal(output.num_accepted_tokens,
                   torch::full({batch_size, 1},
                               static_cast<int64_t>(n_speculative_tokens),
                               options.dtype(torch::kInt32))));
  const auto no_draft =
      rejection_sample(rejection_params,
                       DraftProposal(draft_token_ids.narrow(1, 0, 0)),
                       flatten_rows(full_logits.narrow(/*dim=*/1,
                                                       /*start=*/0,
                                                       /*length=*/1)));
  EXPECT_TRUE(torch::equal(
      no_draft.num_accepted_tokens,
      torch::zeros({batch_size, 1}, options.dtype(torch::kInt32))));
}

TEST(RejectionSamplerTest, GreedyFromTokenIds) {
  const auto device = get_test_device();
  const auto options = torch::dtype(torch::kFloat32).device(device);
  const auto draft_token_ids = torch::tensor({{1, 2, 3}, {3, 1, 0}}, device);
  const auto target_token_ids = torch::tensor({{1, 4, 3}, {3, 2, 0}}, device);
  const auto bonus_token_ids = torch::tensor({{5}, {6}}, device);

  // One-hot [B, n + 1, V] logits whose argmax reproduces target + bonus.
  const int64_t vocab_size = 7;
  torch::Tensor target_logits = torch::zeros({2, 4, vocab_size}, options);
  for (int64_t row = 0; row < 2; ++row) {
    for (int64_t col = 0; col < 3; ++col) {
      target_logits.index_put_({row, col, target_token_ids.index({row, col})},
                               10.0f);
    }
    target_logits.index_put_({row, 3, bonus_token_ids.index({row, 0})}, 10.0f);
  }
  SamplingParameters rejection_params = make_rejection_params();
  SampleOutput output = rejection_sample(rejection_params,
                                         DraftProposal(draft_token_ids),
                                         flatten_rows(target_logits));

  EXPECT_TRUE(
      torch::equal(output.next_tokens,
                   torch::tensor({{1, 4, -1, -1}, {3, 2, -1, -1}}, device)));
  EXPECT_TRUE(torch::equal(
      output.num_accepted_tokens,
      torch::tensor({{1}, {1}}, torch::dtype(torch::kInt32).device(device))));
}

TEST(RejectionSamplerTest, GreedyForwardAllowsUndefinedDraftProbs) {
  const auto options = get_test_options(torch::kFloat32);
  const auto device = get_test_device();
  SamplingParameters rejection_params = make_rejection_params();
  const auto draft_token_ids = torch::tensor({{1, 2}, {3, 4}}, device);
  const auto target_logits = torch::tensor({{{0.0f, 4.0f, 1.0f, 2.0f, 3.0f},
                                             {0.0f, 1.0f, 2.0f, 4.0f, 3.0f},
                                             {4.0f, 0.0f, 1.0f, 2.0f, 3.0f}},
                                            {{0.0f, 1.0f, 2.0f, 4.0f, 3.0f},
                                             {0.0f, 1.0f, 2.0f, 3.0f, 4.0f},
                                             {0.0f, 4.0f, 1.0f, 2.0f, 3.0f}}},
                                           options);
  DraftProposal draft_proposal = DraftProposal(draft_token_ids);
  SampleOutput output = rejection_sample(
      rejection_params, draft_proposal, flatten_rows(target_logits));

  EXPECT_TRUE(torch::equal(output.next_tokens,
                           torch::tensor({{1, 3, -1}, {3, 4, 1}}, device)));
}

TEST(RejectionSamplerTest, GreedyRejectsGrammarMaskedDraftToken) {
  const auto options = get_test_options(torch::kFloat32);

  // The JSON grammar mask removes draft token 0 from the target distribution.
  const auto draft_token_ids =
      torch::tensor({{0}}, options.dtype(torch::kInt64));

  const torch::Tensor target_logits =
      torch::tensor({{{0.0F, 1.0F, 0.0F}, {0.0F, 0.0F, 2.0F}}}, options);
  SamplingParameters rejection_params = make_rejection_params();
  SampleOutput output = rejection_sample(rejection_params,
                                         DraftProposal(draft_token_ids),
                                         flatten_rows(target_logits));

  const auto expected_masked_output =
      torch::tensor({{1, -1}}, options.dtype(torch::kInt64));
  EXPECT_TRUE(torch::equal(output.next_tokens, expected_masked_output));
}

TEST(RejectionSamplerTest, RandomRejectsGrammarMaskedDraftToken) {
  const auto options = get_test_options(torch::kFloat32);

  // The grammar mask removes draft token 0 from the target distribution:
  // p(0)/q(0) = 0 rejects the draft regardless of the draw, and the
  // one-hot residual recovers token 1 deterministically.
  const auto draft_token_ids =
      torch::tensor({{0}}, options.dtype(torch::kInt64));
  const auto draft_probs = torch::tensor({{{1.0F, 0.0F, 0.0F}}}, options);
  const auto target_probs = torch::tensor({{{0.0F, 1.0F, 0.0F}}}, options);

  DraftProposal draft_proposal =
      DraftProposal(draft_token_ids, draft_probs.log());
  auto masked_output = random_sample_from_probs(
      draft_proposal, target_probs, /*uniform_seed=*/1);

  const auto expected_masked_output =
      torch::tensor({{1, -1}}, options.dtype(torch::kInt64));
  EXPECT_TRUE(torch::equal(masked_output, expected_masked_output));
}

TEST(RejectionSamplerTest, RandomForwardUsesGreedyDraftWithoutDraftProbs) {
  const auto options = get_test_options(torch::kFloat32);
  const auto device = get_test_device();
  SamplingParameters rejection_params =
      make_rejection_params(torch::tensor({true}, device));

  DraftProposal draft_proposal = DraftProposal(torch::tensor({{1, 2}}, device));
  const auto target_logits = torch::tensor({{{-1000.0f, 1000.0f, -1000.0f},
                                             {-1000.0f, -1000.0f, 1000.0f},
                                             {1000.0f, -1000.0f, -1000.0f}}},
                                           options);
  auto output = rejection_sample(
      rejection_params, draft_proposal, flatten_rows(target_logits));

  EXPECT_TRUE(
      torch::equal(output.next_tokens, torch::tensor({{1, 2, 0}}, device)));
}

TEST(RejectionSamplerTest, RandomFallsBackToUniformOnDegenerateResidual) {
  const auto options = get_test_options(torch::kFloat32);
  const auto device = get_test_device();

  // The target gives the drafted token zero mass (always rejected), and the
  // residual keeps only the non-drafted tokens, so the recovered token is a
  // free draw over the remaining vocab.
  const int64_t vocab_size = 3;
  const auto draft_token_ids = torch::tensor({{0}}, device);
  const auto draft_probs = torch::tensor({{{1.0f, 0.0f, 0.0f}}}, options);
  const auto target_probs = torch::tensor({{{0.0f, 0.5f, 0.5f}}}, options);
  DraftProposal draft_proposal =
      DraftProposal(draft_token_ids, draft_probs.log());
  auto masked_output = random_sample_from_probs(
      draft_proposal, target_probs, /*uniform_seed=*/4);

  const auto resampled = masked_output.index({0, 0}).item<int64_t>();
  EXPECT_GE(resampled, 0);
  EXPECT_LT(resampled, vocab_size);
  EXPECT_EQ(masked_output.index({0, 1}).item<int64_t>(), -1);
}

TEST(RejectionSamplerTest, LogProbs) {
  const auto options = get_test_options(torch::kFloat32);
  const auto device = get_test_device();
  const auto do_sample = torch::tensor({false, true, false, true}, device);
  const int64_t max_top_logprobs = 2;
  SamplingParameters rejection_params =
      make_rejection_params(do_sample, /*logprobs=*/true, max_top_logprobs);

  int64_t batch_size = 4;
  int64_t n_speculative_tokens = 4;
  int64_t vocab_size = 8;

  const auto draft_token_ids =
      torch::randint(0,
                     vocab_size,
                     {batch_size, n_speculative_tokens},
                     torch::dtype(torch::kInt64).device(device));
  auto draft_logits =
      torch::randn({batch_size, n_speculative_tokens, vocab_size}, options);

  auto target_logits =
      torch::randn({batch_size, n_speculative_tokens + 1, vocab_size}, options);

  DraftProposal draft_proposal = DraftProposal(draft_token_ids, draft_logits);
  auto output = rejection_sample(
      rejection_params, draft_proposal, flatten_rows(target_logits));

  const auto logprobs =
      torch::log_softmax(target_logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
  const auto selected_tokens = output.next_tokens;

  auto expected_logprobs =
      logprobs
          .gather(/*dim=*/-1,
                  selected_tokens.clamp_min(0).unsqueeze(/*dim=*/-1))
          .squeeze(/*dim=*/-1);
  expected_logprobs.masked_fill_(selected_tokens.lt(0), 0);
  EXPECT_TRUE(torch::equal(output.logprobs, expected_logprobs));

  auto [top_k_values, top_k_indices] = logprobs.topk(
      max_top_logprobs, /*dim=*/-1, /*largest=*/true, /*sorted=*/true);
  EXPECT_TRUE(torch::equal(output.top_logprobs, top_k_values));
  EXPECT_TRUE(torch::equal(output.top_tokens, top_k_indices));
}

TEST(RejectionSamplerTest, Random) {
  const auto options = get_test_options(torch::kFloat32);

  torch::manual_seed(100);

  int64_t vocab_size = 50;
  int64_t num_samples = 500000;

  auto target_prob = torch::randn({vocab_size}, options).softmax(/*dim=*/-1);
  auto target_probs =
      target_prob.reshape({1, 1, -1}).repeat({num_samples, 1, 1});

  auto draft_logits = torch::randn({num_samples, 1, vocab_size}, options);
  auto draft_token_ids = gumbel_sample(draft_logits);

  DraftProposal draft_proposal = DraftProposal(draft_token_ids, draft_logits);
  auto output = random_sample_from_probs(
      draft_proposal, target_probs, /*uniform_seed=*/100);

  auto accepted_ids = output.slice(/*dim=*/-1, /*start=*/0, /*end=*/-1);
  accepted_ids = accepted_ids.masked_select(accepted_ids.ge(0));

  // calculate the probability of each sampled token
  auto bincount = accepted_ids.bincount(/*weights=*/torch::nullopt,
                                        /*minlength=*/vocab_size);
  auto sample_prob = bincount.to(torch::kFloat) / num_samples;

  EXPECT_TRUE(torch::allclose(target_prob,
                              sample_prob,
                              /*rtol=*/1e-2,
                              /*atol=*/1e-2));
}

namespace {

class SparseDraftTest
    : public testing::TestWithParam<std::tuple<torch::Device, bool, bool>> {
#if defined(XLLM_NPU_SAMPLING_ABLATION)
 protected:
  void SetUp() override {
    const uint8_t normalization = std::get<1>(GetParam()) ? 8 : 0;
    const uint8_t verification = std::get<2>(GetParam()) ? 32 : 0;
    xllm_npu_sampling_ablation_mask = 23 | normalization | verification;
  }
  void TearDown() override { xllm_npu_sampling_ablation_mask = saved_mask_; }

 private:
  uint8_t saved_mask_ = xllm_npu_sampling_ablation_mask;
#endif
};

torch::Tensor sample_sparse_draft(const DraftProposal& proposal,
                                  const torch::Tensor& probabilities,
                                  int64_t seed) {
#if defined(USE_NPU)
  if (probabilities.device().type() == torch::kPrivateUse1) {
    auto generator =
        at_npu::detail::getDefaultNPUGenerator(probabilities.device().index());
    generator.set_current_seed(static_cast<uint64_t>(seed));
  }
#endif
  return random_sample_from_probs(proposal, probabilities, seed);
}

TEST_P(SparseDraftTest, PreservesRoundedTopLogprobTieIds) {
  const auto options = torch::TensorOptions()
                           .device(std::get<0>(GetParam()))
                           .dtype(torch::kBFloat16);
  torch::Tensor logits = torch::zeros({192, 65536}, options);
  logits.narrow(1, 0, 16).copy_(
      torch::arange(16, options.dtype(torch::kFloat32)).mul(1e-8));
  const torch::Tensor tokens =
      torch::full({64, 2}, 15, options.dtype(torch::kInt64));
  SamplingParameters params;
  params.sampling_mode() = SamplingMode::GREEDY;
  params.logprobs() = true;
  params.max_top_logprobs() = 5;
  const auto [expected_values, expected_tokens] =
      torch::log_softmax(
          logits.view({64, 3, 65536}), /*dim=*/-1, /*dtype=*/torch::kFloat32)
          .topk(/*k=*/5, /*dim=*/-1);
  const SampleOutput actual =
      rejection_sample(params, DraftProposal(tokens), logits);
  EXPECT_TRUE(torch::equal(actual.top_tokens, expected_tokens));
  EXPECT_TRUE(torch::equal(actual.top_logprobs, expected_values));
}

TEST_P(SparseDraftTest, GrammarRejectionDoesNotReplacePointMassProposal) {
  const auto options = torch::TensorOptions().device(std::get<0>(GetParam()));
  const auto ids_options = options.dtype(torch::kInt64);
  const torch::Tensor tokens = torch::tensor({{1, 2}, {1, 2}}, ids_options);
  const DraftProposal proposal(tokens);
  const torch::Tensor invalid = torch::tensor({{true, false}, {false, true}},
                                              options.dtype(torch::kBool));
  const torch::Tensor target =
      torch::full({6, 4}, -INFINITY, options)
          .scatter_(1,
                    torch::tensor({{1}, {2}, {3}, {1}, {2}, {3}}, ids_options),
                    0.0F);
  SamplingParameters params;
  params.sampling_mode() = SamplingMode::RANDOM;
  const SampleOutput masked =
      rejection_sample(params, proposal, target.clone(), invalid);
  EXPECT_TRUE(
      torch::equal(masked.next_tokens,
                   torch::tensor({{1, -1, -1}, {1, 2, -1}}, ids_options)));
  const SampleOutput unmasked =
      rejection_sample(params, proposal, target.clone());
  EXPECT_TRUE(torch::equal(unmasked.next_tokens,
                           torch::tensor({{1, 2, 3}, {1, 2, 3}}, ids_options)));
  EXPECT_TRUE(torch::equal(proposal.token_ids(), tokens));
}

TEST_P(SparseDraftTest, PointMassMatchesOneHotDenseVerification) {
  const auto options = torch::TensorOptions().device(std::get<0>(GetParam()));
  for (const int64_t rows : {1, 8}) {
    const torch::Tensor tokens =
        torch::tensor({{1, 2}}, options.dtype(torch::kInt64)).repeat({rows, 1});
    // Include a forced rejection alongside nontrivial acceptance probabilities.
    tokens.select(0, 0).select(0, 1).fill_(-1);
    const torch::Tensor dense_logits =
        torch::full({rows, 2, 4}, -INFINITY, options)
            .scatter_(-1, tokens.clamp_min(0).unsqueeze(-1), 0.0F);
    const torch::Tensor target =
        torch::tensor({{{0.1F, 0.2F, 0.3F, 0.4F}, {0.4F, 0.3F, 0.2F, 0.1F}}},
                      options)
            .repeat({rows, 1, 1});
    const DraftProposal point_mass(tokens);
    const DraftProposal dense(tokens, dense_logits);
    for (const int64_t seed : {1, 42, 100}) {
      EXPECT_TRUE(torch::equal(sample_sparse_draft(point_mass, target, seed),
                               sample_sparse_draft(dense, target, seed)));
    }
  }
}

TEST_P(SparseDraftTest, SelectedLogprobsPreserveLargeCommonOffsets) {
  const auto options = torch::TensorOptions()
                           .device(std::get<0>(GetParam()))
                           .dtype(torch::kBFloat16);
  const torch::Tensor tokens =
      torch::zeros({64, 2}, options.dtype(torch::kInt64));
  SamplingParameters params;
  params.sampling_mode() = SamplingMode::GREEDY;
  params.logprobs() = true;
  for (const float offset : {8192.0F, 1e8F}) {
    torch::Tensor logits = torch::full({192, 65536}, offset, options);
    const torch::Tensor expected =
        torch::log_softmax(logits.view({64, 3, 65536}), -1, torch::kFloat32)
            .select(-1, 0);
    const SampleOutput actual =
        rejection_sample(params, DraftProposal(tokens), logits);
    EXPECT_TRUE(torch::allclose(actual.logprobs, expected, 1e-5, 5e-6));
    EXPECT_TRUE(torch::equal(actual.next_tokens,
                             torch::zeros({64, 3}, tokens.options())));
  }
}

TEST_P(SparseDraftTest, MatchesDenseWideVocabularyVerification) {
  const auto options = torch::TensorOptions()
                           .device(std::get<0>(GetParam()))
                           .dtype(torch::kFloat32);
  const auto ids_options = options.dtype(torch::kInt64);
  constexpr int64_t kVocab = 248320;
  const torch::Tensor ids =
      torch::arange(16, ids_options).view({1, 1, 16}).expand({8, 7, 16});
  const torch::Tensor small_logits = torch::zeros({8, 7, 16}, options);
  const torch::Tensor tokens = torch::full({8, 7}, 7, ids_options);
  const torch::Tensor dense_logits =
      torch::full(
          {8, 7, kVocab}, -std::numeric_limits<float>::infinity(), options)
          .scatter_(2, ids, small_logits);
  torch::Tensor target = torch::full(
      {64, kVocab}, -std::numeric_limits<float>::infinity(), options);
  target.narrow(1, 0, 16).copy_(torch::arange(16, options).mul(0.125));
  target.select(1, 7).fill_(-std::numeric_limits<float>::infinity());
  SamplingParameters params;
  params.sampling_mode() = SamplingMode::RANDOM;
  params.logprobs() = true;
#if defined(USE_NPU)
  auto generator = at_npu::detail::getDefaultNPUGenerator(0);
  generator.set_current_seed(42);
#endif
  torch::manual_seed(42);
  torch::Tensor dense_target = target.clone();
  const SampleOutput expected = rejection_sample(
      params, DraftProposal(tokens, dense_logits), dense_target);
#if defined(USE_NPU)
  generator.set_current_seed(42);
#endif
  torch::manual_seed(42);
  const SampleOutput actual = rejection_sample(
      params,
      DraftProposal(
          tokens, std::make_unique<SparseDraftDistribution>(ids, small_logits)),
      target);
  EXPECT_TRUE(torch::equal(actual.next_tokens, expected.next_tokens));
  EXPECT_TRUE(torch::allclose(actual.logprobs, expected.logprobs, 1e-5, 5e-6));
}

TEST_P(SparseDraftTest, MatchesDenseSupportAndSentinelRecovery) {
  const auto options = torch::TensorOptions()
                           .device(std::get<0>(GetParam()))
                           .dtype(torch::kFloat32);
  const auto ids_options = options.dtype(torch::kInt64);
  const torch::Tensor tokens = torch::tensor({{1, 3}, {-1, 2}}, ids_options);
  const torch::Tensor candidates =
      torch::tensor({{{1, 2}, {3, 4}}, {{1, 2}, {2, 3}}}, ids_options);
  const torch::Tensor sparse_logits =
      torch::tensor(
          {{{1.0F, 0.0F}, {1.0F, 0.0F}}, {{0.25F, 0.75F}, {0.5F, 0.5F}}},
          options)
          .log();
  const torch::Tensor dense_logits =
      torch::full({2, 2, 5}, -std::numeric_limits<float>::infinity(), options)
          .scatter_(2, candidates, sparse_logits);
  // Accept the first draft, then recover outside q's support; the sentinel
  // row must recover directly from p, without subtracting its stored q.
  const torch::Tensor target_probs = torch::tensor(
      {{{0.0F, 1.0F, 0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F, 0.0F, 0.0F}},
       {{0.0F, 0.0F, 0.0F, 0.0F, 1.0F}, {0.0F, 0.0F, 1.0F, 0.0F, 0.0F}}},
      options);
  const DraftProposal sparse = DraftProposal(
      tokens,
      std::make_unique<SparseDraftDistribution>(candidates, sparse_logits));
  const torch::Tensor sparse_result =
      sample_sparse_draft(sparse, target_probs, 123);
  const torch::Tensor dense_result = sample_sparse_draft(
      DraftProposal(tokens, dense_logits), target_probs, 123);
  EXPECT_TRUE(torch::equal(sparse_result, dense_result));
  EXPECT_TRUE(torch::equal(
      sparse_result, torch::tensor({{1, 0, -1}, {4, -1, -1}}, ids_options)));
}

TEST_P(SparseDraftTest, MatchesDenseRandomRejection) {
  const auto options = torch::TensorOptions()
                           .device(std::get<0>(GetParam()))
                           .dtype(torch::kFloat32);
  constexpr int64_t kRows = 256;
  const torch::Tensor candidates =
      torch::tensor({{{0, 2}, {1, 3}}}, options.dtype(torch::kInt64))
          .repeat({kRows, 1, 1});
  const torch::Tensor sparse_logits =
      torch::tensor({{{0.25F, 0.75F}, {0.5F, 0.5F}}}, options)
          .log()
          .repeat({kRows, 1, 1});
  const torch::Tensor dense_logits =
      torch::full(
          {kRows, 2, 4}, -std::numeric_limits<float>::infinity(), options)
          .scatter_(2, candidates, sparse_logits);
  const torch::Tensor tokens =
      torch::tensor({{2, 1}}, options.dtype(torch::kInt64)).repeat({kRows, 1});
  const torch::Tensor target_probs =
      torch::tensor(
          {{{0.25F, 0.25F, 0.25F, 0.25F}, {0.125F, 0.125F, 0.25F, 0.5F}}},
          options)
          .repeat({kRows, 1, 1});
  for (int64_t seed = 0; seed < 4; ++seed) {
    const torch::Tensor dense_result = sample_sparse_draft(
        DraftProposal(tokens, dense_logits), target_probs, seed);
    const torch::Tensor sparse_result = sample_sparse_draft(
        DraftProposal(tokens,
                      std::make_unique<SparseDraftDistribution>(candidates,
                                                                sparse_logits)),
        target_probs,
        seed);
    EXPECT_TRUE(torch::equal(sparse_result, dense_result));
  }
}

TEST_P(SparseDraftTest, RecoveredDistributionMatchesTarget) {
  constexpr int64_t kRows = 2048;
  constexpr int64_t kRounds = 16;
  const auto options = torch::TensorOptions()
                           .device(std::get<0>(GetParam()))
                           .dtype(torch::kFloat32);
  const torch::Tensor candidates =
      torch::tensor({{{0, 2}}}, options.dtype(torch::kInt64))
          .repeat({kRows, 1, 1});
  const torch::Tensor logits =
      torch::tensor({{{0.25F, 0.75F}}}, options).log().repeat({kRows, 1, 1});
  const torch::Tensor target =
      torch::tensor({{{0.1F, 0.2F, 0.3F, 0.4F}}}, options)
          .repeat({kRows, 1, 1});
  torch::Tensor counts = torch::zeros({4}, torch::kInt64);
  for (int64_t round = 0; round < kRounds; ++round) {
    const torch::Tensor selected = gumbel_sample(logits);
    const torch::Tensor tokens =
        candidates.gather(-1, selected.unsqueeze(-1)).squeeze(-1);
    const torch::Tensor output = sample_sparse_draft(
        DraftProposal(
            tokens,
            std::make_unique<SparseDraftDistribution>(candidates, logits)),
        target,
        100 + round);
    counts.add_(output.select(1, 0).cpu().bincount({}, 4));
  }
  const torch::Tensor observed = counts.to(torch::kFloat32) / (kRows * kRounds);
  EXPECT_TRUE(torch::allclose(observed,
                              target.select(0, 0).select(0, 0).cpu(),
                              /*rtol=*/0.0,
                              /*atol=*/0.015));
}

#if defined(USE_NPU)
const std::array<torch::Device, 2> kSparseDevices = {
    torch::Device(torch::kCPU),
    torch::Device(torch::kPrivateUse1, 0)};
#else
const std::array<torch::Device, 1> kSparseDevices = {
    torch::Device(torch::kCPU)};
#endif

#if defined(XLLM_NPU_SAMPLING_ABLATION)
INSTANTIATE_TEST_SUITE_P(Devices,
                         SparseDraftTest,
                         testing::Combine(testing::ValuesIn(kSparseDevices),
                                          testing::Bool(),
                                          testing::Bool()));
#else
INSTANTIATE_TEST_SUITE_P(Devices,
                         SparseDraftTest,
                         testing::Combine(testing::ValuesIn(kSparseDevices),
                                          testing::Values(true),
                                          testing::Values(true)));
#endif

}  // namespace

}  // namespace xllm

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

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <cstdint>

#include "core/framework/sampling/sampler.h"
#include "tests/core/framework/sampling/sampling_test_utils.h"

namespace xllm {

using testutil::flatten_rows;
using testutil::get_test_device;
using testutil::get_test_options;
using testutil::make_rejection_params;
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

  auto masked_output = random_sample_from_probs(
      target_probs, /*uniform_seed=*/1, draft_token_ids, draft_probs.log());
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
  auto masked_output = random_sample_from_probs(
      target_probs, /*uniform_seed=*/1, draft_token_ids);
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
  SampleOutput output = rejection_sample(
      rejection_params, draft_token_ids, {}, flatten_rows(full_logits));

  EXPECT_TRUE(
      torch::allclose(output.next_tokens, full_logits.argmax(/*dim=*/-1)));
  // Full-width acceptance, not the clamped gather index of width - 1.
  EXPECT_TRUE(
      torch::equal(output.num_accepted_tokens,
                   torch::full({batch_size, 1},
                               static_cast<int64_t>(n_speculative_tokens),
                               options.dtype(torch::kInt32))));
  const auto no_draft = rejection_sample(
      rejection_params,
      draft_token_ids.narrow(/*dim=*/1, /*start=*/0, /*length=*/0),
      {},
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

  const int64_t vocab_size = 7;
  torch::Tensor target_logits = torch::zeros({2, 4, vocab_size}, options);
  for (int64_t row = 0; row < 2; ++row) {
    for (int64_t col = 0; col < 3; ++col) {
      target_logits.index_put_({row, col, target_token_ids.index({row, col})},
                               10.0f);
    }
    target_logits.index_put_({row, 3, bonus_token_ids.index({row, 0})}, 10.0f);
  }
  target_logits.index_put_({1, 0, 2}, /*value=*/5.0f);
  SamplingParameters rejection_params = make_rejection_params();
  rejection_params.filter_bitmask() = torch::tensor(
      {{0x7f}, {0x7f & ~(1 << 3)}}, torch::dtype(torch::kInt32).device(device));
  SampleOutput output = rejection_sample(
      rejection_params, draft_token_ids, {}, flatten_rows(target_logits));

  EXPECT_TRUE(
      torch::equal(output.next_tokens,
                   torch::tensor({{1, 4, -1, -1}, {2, -1, -1, -1}}, device)));
  EXPECT_TRUE(torch::equal(
      output.num_accepted_tokens,
      torch::tensor({{1}, {0}}, torch::dtype(torch::kInt32).device(device))));
}

TEST(RejectionSamplerTest, RandomForwardUsesGreedyDraftWithoutDraftProbs) {
  const auto options = get_test_options(torch::kFloat32);
  const auto device = get_test_device();
  SamplingParameters rejection_params =
      make_rejection_params(torch::tensor({true}, device));

  const auto draft_token_ids = torch::tensor({{1, 2}}, device);
  const auto target_logits = torch::tensor({{{-1000.0f, 1000.0f, -1000.0f},
                                             {-1000.0f, -1000.0f, 1000.0f},
                                             {1000.0f, -1000.0f, -1000.0f}}},
                                           options);
  auto output = rejection_sample(
      rejection_params, draft_token_ids, {}, flatten_rows(target_logits));

  EXPECT_TRUE(
      torch::equal(output.next_tokens, torch::tensor({{1, 2, 0}}, device)));
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

  auto output = rejection_sample(rejection_params,
                                 draft_token_ids,
                                 draft_logits,
                                 flatten_rows(target_logits));

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

  auto output = random_sample_from_probs(
      target_probs, /*uniform_seed=*/100, draft_token_ids, draft_logits);

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

}  // namespace xllm

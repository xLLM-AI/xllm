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

#include <cstdint>
#include <tuple>
#include <vector>

#include "core/framework/sampling/json_object_grammar.h"
#include "core/framework/sampling/rejection_sampler.h"
#include "core/framework/speculative/metrics.h"
#include "core/framework/speculative/verify_layout.h"

namespace xllm {

namespace {

SpecMetrics calculate_from_widths(
    const torch::Tensor& tokens,
    const std::vector<int32_t>& num_draft_tokens) {
  return make_spec_metrics(tokens,
                           make_verify_layouts(tokens, &num_draft_tokens, {}));
}

TEST(SpeculativeTokenStatsTest, CountsTokensPerSequence) {
  const torch::Tensor tokens = torch::tensor(
      {{10, 11, 12, -1, -1, -1}, {20, 21, 22, 23, 24, 25}}, torch::kInt32);
  // The 1-arg make_verify_layouts is the full-width default the static MTP,
  // suffix, unified, and pipeline paths use.
  const auto metrics = make_spec_metrics(tokens, make_verify_layouts(tokens));

  EXPECT_EQ(metrics.num_committed_tokens, 9);
  EXPECT_EQ(metrics.num_accepted_tokens_per_pos,
            (std::vector<int64_t>{2, 2, 1, 1, 1}));
  ASSERT_EQ(metrics.sequence_stats.size(), 2);
  EXPECT_EQ(metrics.sequence_stats[0].accepted_tokens, 2);
  EXPECT_EQ(metrics.sequence_stats[1].accepted_tokens, 5);
}

TEST(SpeculativeTokenStatsTest, CountsAdaptiveTokensPerSequence) {
  const torch::Tensor tokens = torch::tensor(
      {{10, 11, 12, -1, -1}, {20, 21, 22, -1, -1}, {30, 31, 32, 33, -1}},
      torch::kInt32);
  const std::vector<int32_t> num_draft_tokens = {4, 2, 0};
  const SpecMetrics aggregate = calculate_from_widths(tokens, num_draft_tokens);
  EXPECT_EQ(aggregate.num_accepted_tokens_per_pos,
            (std::vector<int64_t>{2, 2, 0, 0}));

  ASSERT_EQ(aggregate.sequence_stats.size(), 3);
  EXPECT_EQ(aggregate.sequence_stats[0].accepted_tokens, 2);
  EXPECT_EQ(aggregate.sequence_stats[0].proposed_tokens, 4);
  EXPECT_EQ(aggregate.sequence_stats[1].accepted_tokens, 2);
  EXPECT_EQ(aggregate.sequence_stats[1].proposed_tokens, 2);
  EXPECT_EQ(aggregate.sequence_stats[2].accepted_tokens, 0);
  EXPECT_EQ(aggregate.sequence_stats[2].proposed_tokens, 0);
}

TEST(SpeculativeTokenStatsTest, GreedyExcludesTargetTokens) {
  const torch::Tensor num_draft_tokens = torch::tensor(
      {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}, {1, 2, 3}}, torch::kInt64);
  // Reject at each possible position, then accept the entire final row.
  const torch::Tensor target_tokens = torch::tensor(
      {{9, 2, 3}, {1, 9, 3}, {1, 2, 9}, {1, 2, 3}}, torch::kInt64);
  const torch::Tensor bonus_tokens = torch::full({4, 1}, 8, torch::kInt64);
  const auto sampled = RejectionSampler::greedy_sample_from_token_ids(
      num_draft_tokens,
      target_tokens,
      bonus_tokens,
      /*mask_out_rejected_tokens=*/true);
  const torch::Tensor& masked_tokens = std::get<1>(sampled);
  ASSERT_TRUE(torch::equal(
      masked_tokens,
      torch::tensor(
          {{9, -1, -1, -1}, {1, 9, -1, -1}, {1, 2, 9, -1}, {1, 2, 3, 8}},
          torch::kInt64)));

  const auto output_stats = calculate_from_widths(masked_tokens, {3, 3, 3, 3});
  const auto& stats = output_stats.sequence_stats;
  ASSERT_EQ(stats.size(), 4);
  for (size_t row = 0; row < stats.size(); ++row) {
    EXPECT_EQ(stats[row].accepted_tokens, static_cast<int64_t>(row));
    EXPECT_EQ(stats[row].proposed_tokens, 3);
  }
  EXPECT_EQ(output_stats.num_committed_tokens, 10);
  EXPECT_EQ(output_stats.num_accepted_tokens_per_pos,
            (std::vector<int64_t>{3, 2, 1}));
}

TEST(SpeculativeTokenStatsTest, BlockRandomExcludesTargetTokens) {
  const torch::Tensor num_draft_tokens = torch::zeros({4, 3}, torch::kInt64);
  const torch::Tensor draft_probs =
      torch::tensor({0.9f, 0.1f}).view({1, 1, 2}).repeat({4, 3, 1});
  const torch::Tensor target_probs =
      torch::tensor({0.8f, 0.2f}).view({1, 1, 2}).repeat({4, 3, 1});
  // The fixed draws reject at positions 0, 1, 2, then accept all drafts.
  // Residual probability exists only for token 1, so recovery is
  // deterministic.
  const torch::Tensor uniform_rand = torch::tensor({{0.95f, 0.1f, 0.1f},
                                                    {0.1f, 0.95f, 0.1f},
                                                    {0.1f, 0.1f, 0.95f},
                                                    {0.1f, 0.1f, 0.1f}});
  const torch::Tensor bonus_tokens = torch::ones({4, 1}, torch::kInt64);
  const DraftProposal proposal(num_draft_tokens, draft_probs);
  const auto sampled =
      RejectionSampler::random_sample(proposal,
                                      target_probs,
                                      uniform_rand,
                                      bonus_tokens,
                                      /*mask_out_rejected_tokens=*/true);
  const torch::Tensor& masked_tokens = std::get<1>(sampled);
  ASSERT_TRUE(torch::equal(
      masked_tokens,
      torch::tensor(
          {{1, -1, -1, -1}, {0, 1, -1, -1}, {0, 0, 1, -1}, {0, 0, 0, 1}},
          torch::kInt64)));

  const auto stats =
      calculate_from_widths(masked_tokens, {3, 3, 3, 3}).sequence_stats;
  ASSERT_EQ(stats.size(), 4);
  for (size_t row = 0; row < stats.size(); ++row) {
    EXPECT_EQ(stats[row].accepted_tokens, static_cast<int64_t>(row));
    EXPECT_EQ(stats[row].proposed_tokens, 3);
  }
}

TEST(SpeculativeTokenStatsTest, BlockHandlesSingleAndZeroDrafts) {
  const torch::Tensor num_draft_tokens =
      torch::tensor({{1}, {1}}, torch::kInt64);
  const torch::Tensor target_tokens = torch::tensor({{9}, {1}}, torch::kInt64);
  const torch::Tensor bonus_tokens = torch::full({2, 1}, 8, torch::kInt64);
  const auto sampled = RejectionSampler::greedy_sample_from_token_ids(
      num_draft_tokens,
      target_tokens,
      bonus_tokens,
      /*mask_out_rejected_tokens=*/true);
  const torch::Tensor& single_tokens = std::get<1>(sampled);
  const auto stats =
      calculate_from_widths(single_tokens, {1, 1}).sequence_stats;
  ASSERT_EQ(stats.size(), 2);
  EXPECT_EQ(stats[0].accepted_tokens, 0);
  EXPECT_EQ(stats[0].proposed_tokens, 1);
  EXPECT_EQ(stats[1].accepted_tokens, 1);
  EXPECT_EQ(stats[1].proposed_tokens, 1);

  // A fully pruned row emits only the target bonus and proposes no drafts.
  const torch::Tensor empty_drafts = torch::empty({2, 0}, torch::kInt64);
  const auto bonus_only = RejectionSampler::greedy_sample_from_token_ids(
      empty_drafts,
      empty_drafts,
      bonus_tokens,
      /*mask_out_rejected_tokens=*/true);
  ASSERT_TRUE(torch::equal(std::get<1>(bonus_only), bonus_tokens));
  const torch::Tensor& bonus_only_tokens = std::get<1>(bonus_only);
  const auto zero_stats =
      calculate_from_widths(bonus_only_tokens, {0, 0}).sequence_stats;
  ASSERT_EQ(zero_stats.size(), 2);
  for (const SpeculativeTokenStats& row_stats : zero_stats) {
    EXPECT_EQ(row_stats.accepted_tokens, 0);
    EXPECT_EQ(row_stats.proposed_tokens, 0);
  }
}

}  // namespace
}  // namespace xllm

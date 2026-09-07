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

#include "core/distributed_runtime/worker_service.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "core/common/metrics.h"
#include "core/framework/sampling/rejection_sampler.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/runtime/options.h"
#include "core/runtime/speculative_worker_impl.h"

namespace xllm {

class WorkerServiceTestPeer final {
 public:
  static std::vector<SpeculativeTokenStats> record_speculative_metrics(
      WorkerService& service,
      const torch::Tensor& tokens,
      const std::vector<SpeculativeTokenStats>& output_stats) {
    return service.record_speculative_metrics_from_output(tokens, output_stats);
  }
};

namespace {

TEST(SpeculativeTokenStatsTest, CountsTokensPerSequence) {
  // Row 0 accepts two draft tokens; row 1 accepts all five.
  const torch::Tensor tokens = torch::tensor(
      {{10, 11, 12, -1, -1, -1}, {20, 21, 22, 23, 24, 25}}, torch::kInt32);
  const auto stats =
      calculate_speculative_output_stats(tokens, /*num_speculative_tokens=*/5);

  EXPECT_EQ(stats.committed_tokens, 9);
  EXPECT_EQ(stats.accepted_per_position, (std::vector<int64_t>{2, 2, 1, 1, 1}));
  ASSERT_EQ(stats.sequence_stats.size(), 2);
  EXPECT_EQ(stats.sequence_stats[0].accepted_tokens, 2);
  EXPECT_EQ(stats.sequence_stats[0].proposed_tokens, 5);
  EXPECT_EQ(stats.sequence_stats[1].accepted_tokens, 5);
  EXPECT_EQ(stats.sequence_stats[1].proposed_tokens, 5);
  EXPECT_EQ(stats.sequence_stats[0].accepted_tokens +
                stats.sequence_stats[1].accepted_tokens,
            7);
}

TEST(SpeculativeTokenStatsTest, CountsAdaptiveTokensPerSequence) {
  const torch::Tensor tokens = torch::tensor(
      {{10, 11, 12, -1, -1}, {20, 21, 22, -1, -1}, {30, 31, 32, 33, -1}},
      torch::kInt32);
  const std::vector<int32_t> proposed_tokens = {4, 2, 0};

  const std::vector<SpeculativeTokenStats> mtp_stats =
      calculate_mtp_speculative_token_stats(tokens, proposed_tokens);
  ASSERT_EQ(mtp_stats.size(), 3);
  EXPECT_EQ(mtp_stats[0].accepted_tokens, 2);
  EXPECT_EQ(mtp_stats[0].proposed_tokens, 4);
  EXPECT_EQ(mtp_stats[1].accepted_tokens, 2);
  EXPECT_EQ(mtp_stats[1].proposed_tokens, 2);
  EXPECT_EQ(mtp_stats[2].accepted_tokens, 0);
  EXPECT_EQ(mtp_stats[2].proposed_tokens, 0);

  const std::vector<SpeculativeTokenStats> block_stats =
      calculate_block_speculative_token_stats(tokens, proposed_tokens);
  ASSERT_EQ(block_stats.size(), 3);
  EXPECT_EQ(block_stats[0].accepted_tokens, 2);
  EXPECT_EQ(block_stats[0].proposed_tokens, 4);
  EXPECT_EQ(block_stats[1].accepted_tokens, 2);
  EXPECT_EQ(block_stats[1].proposed_tokens, 2);
  EXPECT_EQ(block_stats[2].accepted_tokens, 0);
  EXPECT_EQ(block_stats[2].proposed_tokens, 0);
}

TEST(SpeculativeTokenStatsTest, BlockGreedyExcludesTargetTokens) {
  const torch::Tensor draft_tokens = torch::tensor(
      {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}, {1, 2, 3}}, torch::kInt64);
  // Reject at each possible position, then accept the entire final row.
  const torch::Tensor target_tokens = torch::tensor(
      {{9, 2, 3}, {1, 9, 3}, {1, 2, 9}, {1, 2, 3}}, torch::kInt64);
  const torch::Tensor bonus_tokens = torch::full({4, 1}, 8, torch::kInt64);
  const auto sampled = RejectionSampler::greedy_sample_from_token_ids(
      draft_tokens,
      target_tokens,
      bonus_tokens,
      /*mask_out_rejected_tokens=*/true);
  const torch::Tensor& masked_tokens = std::get<1>(sampled);
  ASSERT_TRUE(torch::equal(
      masked_tokens,
      torch::tensor(
          {{9, -1, -1, -1}, {1, 9, -1, -1}, {1, 2, 9, -1}, {1, 2, 3, 8}},
          torch::kInt64)));

  const auto stats =
      calculate_block_speculative_token_stats(masked_tokens, {3, 3, 3, 3});
  ASSERT_EQ(stats.size(), 4);
  for (size_t row = 0; row < stats.size(); ++row) {
    EXPECT_EQ(stats[row].accepted_tokens, static_cast<int64_t>(row));
    EXPECT_EQ(stats[row].proposed_tokens, 3);
  }
}

TEST(SpeculativeTokenStatsTest, BlockRandomExcludesTargetTokens) {
  const torch::Tensor draft_tokens = torch::zeros({4, 3}, torch::kInt64);
  const torch::Tensor draft_probs =
      torch::tensor({0.9f, 0.1f}).view({1, 1, 2}).repeat({4, 3, 1});
  const torch::Tensor target_probs =
      torch::tensor({0.8f, 0.2f}).view({1, 1, 2}).repeat({4, 3, 1});
  // The fixed draws reject at positions 0, 1, 2, then accept all drafts.
  // Residual probability exists only for token 1, so recovery is deterministic.
  const torch::Tensor uniform_rand = torch::tensor({{0.95f, 0.1f, 0.1f},
                                                    {0.1f, 0.95f, 0.1f},
                                                    {0.1f, 0.1f, 0.95f},
                                                    {0.1f, 0.1f, 0.1f}});
  const torch::Tensor bonus_tokens = torch::ones({4, 1}, torch::kInt64);
  const DraftProposal proposal(draft_tokens, draft_probs);
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
      calculate_block_speculative_token_stats(masked_tokens, {3, 3, 3, 3});
  ASSERT_EQ(stats.size(), 4);
  for (size_t row = 0; row < stats.size(); ++row) {
    EXPECT_EQ(stats[row].accepted_tokens, static_cast<int64_t>(row));
    EXPECT_EQ(stats[row].proposed_tokens, 3);
  }
}

TEST(SpeculativeTokenStatsTest, BlockHandlesSingleAndZeroDrafts) {
  const torch::Tensor draft_tokens = torch::tensor({{1}, {1}}, torch::kInt64);
  const torch::Tensor target_tokens = torch::tensor({{9}, {1}}, torch::kInt64);
  const torch::Tensor bonus_tokens = torch::full({2, 1}, 8, torch::kInt64);
  const auto sampled = RejectionSampler::greedy_sample_from_token_ids(
      draft_tokens,
      target_tokens,
      bonus_tokens,
      /*mask_out_rejected_tokens=*/true);
  const auto stats =
      calculate_block_speculative_token_stats(std::get<1>(sampled), {1, 1});
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
  const auto zero_stats =
      calculate_block_speculative_token_stats(std::get<1>(bonus_only), {0, 0});
  ASSERT_EQ(zero_stats.size(), 2);
  for (const SpeculativeTokenStats& row_stats : zero_stats) {
    EXPECT_EQ(row_stats.accepted_tokens, 0);
    EXPECT_EQ(row_stats.proposed_tokens, 0);
  }
}

TEST(WorkerServiceMetricsTest, AdaptiveMtpUpdatesMetricsOnce) {
  runtime::Options options;
  options.enable_speculative_decode(true)
      .num_speculative_tokens(3)
      .speculative_algorithm("mtp");
  WorkerService service(options, torch::Device("npu:0"));
  // Two accepted drafts, one fully accepted pruned row, and a bonus-only row.
  const torch::Tensor tokens = torch::tensor(
      {{10, 11, 12, -1}, {20, 21, -1, -1}, {30, -1, -1, -1}}, torch::kInt32);
  const std::vector<SpeculativeTokenStats> output_stats = {
      {2, 3}, {1, 1}, {0, 0}};

  const double drafts_before = COUNTER_VALUE(speculative_num_drafts_total);
  const double committed_before =
      COUNTER_VALUE(speculative_num_committed_tokens_total);
  std::vector<double> positions_before;
  positions_before.reserve(3);
  for (int32_t position = 0; position < 3; ++position) {
    positions_before.emplace_back(
        MULTI_COUNTER_speculative_num_accepted_tokens_per_pos
            .get_stats({std::to_string(position)})
            ->get_value());
  }
  // The adaptive worker has already published these accurate totals.
  COUNTER_ADD(speculative_num_draft_tokens_total, 4);
  COUNTER_ADD(speculative_num_accepted_tokens_total, 3);
  const double proposed_before =
      COUNTER_VALUE(speculative_num_draft_tokens_total);
  const double accepted_before =
      COUNTER_VALUE(speculative_num_accepted_tokens_total);
  const auto result = WorkerServiceTestPeer::record_speculative_metrics(
      service, tokens, output_stats);

  ASSERT_EQ(result.size(), output_stats.size());
  for (size_t row = 0; row < result.size(); ++row) {
    EXPECT_EQ(result[row].accepted_tokens, output_stats[row].accepted_tokens);
    EXPECT_EQ(result[row].proposed_tokens, output_stats[row].proposed_tokens);
  }
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(speculative_num_drafts_total),
                   drafts_before + 3);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(speculative_num_committed_tokens_total),
                   committed_before + 6);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(speculative_num_draft_tokens_total),
                   proposed_before);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(speculative_num_accepted_tokens_total),
                   accepted_before);
  const std::vector<double> expected_positions = {2, 1, 0};
  for (int32_t position = 0; position < 3; ++position) {
    EXPECT_DOUBLE_EQ(MULTI_COUNTER_speculative_num_accepted_tokens_per_pos
                         .get_stats({std::to_string(position)})
                         ->get_value(),
                     positions_before[position] + expected_positions[position]);
  }
  const double mean_tokens =
      GAUGE_VALUE(speculative_mean_tokens_per_decode_step);
  EXPECT_DOUBLE_EQ(mean_tokens, (committed_before + 6) / (drafts_before + 3));
}

TEST(WorkerServiceMetricsTest, StaticMtpRecordsTokenTotals) {
  runtime::Options options;
  options.enable_speculative_decode(true)
      .num_speculative_tokens(3)
      .speculative_algorithm("mtp");
  WorkerService service(options, torch::Device("npu:0"));
  const torch::Tensor tokens = torch::tensor({{10, 11, -1, -1}}, torch::kInt32);
  const double drafts_before = COUNTER_VALUE(speculative_num_drafts_total);
  const double proposed_before =
      COUNTER_VALUE(speculative_num_draft_tokens_total);
  const double accepted_before =
      COUNTER_VALUE(speculative_num_accepted_tokens_total);
  const double committed_before =
      COUNTER_VALUE(speculative_num_committed_tokens_total);

  const auto result = WorkerServiceTestPeer::record_speculative_metrics(
      service, tokens, /*output_stats=*/{});

  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0].accepted_tokens, 1);
  EXPECT_EQ(result[0].proposed_tokens, 3);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(speculative_num_drafts_total),
                   drafts_before + 1);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(speculative_num_draft_tokens_total),
                   proposed_before + 3);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(speculative_num_accepted_tokens_total),
                   accepted_before + 1);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(speculative_num_committed_tokens_total),
                   committed_before + 2);
}

TEST(WorkerServiceMetricsTest, BlockDiffusionPreservesInlineMetrics) {
  const torch::Tensor tokens = torch::tensor({{10, 11, -1, -1}}, torch::kInt32);
  const std::vector<SpeculativeTokenStats> output_stats =
      calculate_block_speculative_token_stats(tokens, {3});
  const double drafts_before = COUNTER_VALUE(speculative_num_drafts_total);
  const double proposed_before =
      COUNTER_VALUE(speculative_num_draft_tokens_total);
  const double accepted_before =
      COUNTER_VALUE(speculative_num_accepted_tokens_total);
  const double committed_before =
      COUNTER_VALUE(speculative_num_committed_tokens_total);

  for (const std::string& algorithm :
       {std::string("dflash"), std::string("dspark")}) {
    runtime::Options options;
    options.enable_speculative_decode(true)
        .num_speculative_tokens(3)
        .speculative_algorithm(algorithm);
    WorkerService service(options, torch::Device("npu:0"));
    const auto result = WorkerServiceTestPeer::record_speculative_metrics(
        service, tokens, output_stats);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].accepted_tokens, 1);
    EXPECT_EQ(result[0].proposed_tokens, 3);
  }
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(speculative_num_drafts_total), drafts_before);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(speculative_num_draft_tokens_total),
                   proposed_before);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(speculative_num_accepted_tokens_total),
                   accepted_before);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(speculative_num_committed_tokens_total),
                   committed_before);
}

}  // namespace
}  // namespace xllm

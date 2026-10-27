/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "layers/mlu/dcp_indexer_candidate.h"

#include <framework/core/MLUStream.h>
#include <framework/core/device.h>
#include <framework/core/stream_guard.h>
#include <framework/graphs/MLUGraph.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "core/kv_cache/layout/kv_shard_layout.h"
#include "core/layers/mlu/dcp_batch_metadata.h"
#include "layers/common/attention_metadata.h"
#include "layers/common/kv_shard_batch_metadata.h"
#include "platform/platform.h"

namespace xllm::layer {
namespace {

TEST(DcpIndexerCandidateTest, PreservesInterleavedSlotsForFourRankTopology) {
  const KVShardLayout layout(
      /*physical_block_size=*/2, /*dcp_size=*/4, /*dcp_rank=*/2);
  const torch::Tensor global_slots =
      torch::tensor({0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15}, torch::kInt32);

  const torch::Tensor local_slots =
      localize_kv_shard_slots(global_slots, layout);

  EXPECT_TRUE(
      torch::equal(local_slots,
                   torch::tensor({-1, -1, -1, -1, 0, 1, -1, -1, 2, 3, -1, -1},
                                 torch::kInt32)));
}

TEST(DcpIndexerCandidateTest, CountsPartialPrefixesForEightRankTopology) {
  const KVShardLayout layout(
      /*physical_block_size=*/2, /*dcp_size=*/8, /*dcp_rank=*/6);
  const torch::Tensor global_context_lens =
      torch::tensor({0, 12, 13, 14, 15, 16, 30, 31, 32}, torch::kInt32);

  const torch::Tensor local_context_lens =
      localize_kv_shard_context_lens(global_context_lens, layout);

  EXPECT_TRUE(
      torch::equal(local_context_lens,
                   torch::tensor({0, 0, 1, 2, 2, 2, 4, 4, 4}, torch::kInt32)));
}

TEST(DcpIndexerCandidateTest,
     ExpandsPrefillQueriesIntoRankLocalCausalSelectorRows) {
  const KVShardLayout layout(
      /*physical_block_size=*/2, /*dcp_size=*/2, /*dcp_rank=*/1);
  AttentionMetadata metadata{};
  metadata.q_cu_seq_lens = torch::tensor({0, 3, 5}, torch::kInt32);
  metadata.kv_cu_seq_lens = torch::tensor({0, 3, 9}, torch::kInt32);
  metadata.block_table = torch::tensor({{5, 6}, {9, 10}}, torch::kInt32);
  metadata.slot_mapping = torch::arange(5, torch::kInt32);

  const KVShardCausalSelectorMetadata causal_metadata =
      build_kv_shard_causal_selector_metadata(metadata, layout);

  EXPECT_TRUE(
      torch::equal(causal_metadata.block_table,
                   torch::tensor({{5, 6}, {5, 6}, {5, 6}, {9, 10}, {9, 10}},
                                 torch::kInt32)));
  EXPECT_TRUE(torch::equal(causal_metadata.local_context_lens,
                           torch::tensor({0, 0, 1, 2, 2}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(causal_metadata.q_cu_seq_lens,
                           torch::tensor({0, 1, 2, 3, 4, 5}, torch::kInt32)));
}

TEST(DcpIndexerCandidateTest,
     DoesNotBuildPrefillOnlyCausalSelectorForDecodeMetadata) {
  const KVShardLayout layout(
      /*physical_block_size=*/2, /*dcp_size=*/2, /*dcp_rank=*/1);
  AttentionMetadata metadata{};
  metadata.q_cu_seq_lens = torch::tensor({0, 1, 2}, torch::kInt32);
  metadata.kv_cu_seq_lens = torch::tensor({0, 9, 18}, torch::kInt32);
  metadata.kv_seq_lens = torch::tensor({9, 9}, torch::kInt32);
  metadata.block_table = torch::tensor({{5, 6}, {9, 10}}, torch::kInt32);
  metadata.slot_mapping = torch::tensor({4, 5}, torch::kInt32);
  metadata.is_prefill = false;
  metadata.is_chunked_prefill = false;

  const KVShardCausalSelectorMetadata selector =
      build_kv_shard_causal_selector_metadata(metadata, layout);
  EXPECT_TRUE(selector.block_table.defined());
  EXPECT_TRUE(torch::equal(selector.q_cu_seq_lens,
                           torch::tensor({0, 1, 2}, torch::kInt32)));
}

TEST(DcpIndexerCandidateTest, BuildsRequestedPrefillCausalSelector) {
  const KVShardLayout layout(
      /*physical_block_size=*/2, /*dcp_size=*/2, /*dcp_rank=*/1);
  AttentionMetadata metadata{};
  metadata.q_cu_seq_lens = torch::tensor({0, 3}, torch::kInt32);
  metadata.kv_cu_seq_lens = torch::tensor({0, 3}, torch::kInt32);
  metadata.block_table = torch::tensor({{5}}, torch::kInt32);
  metadata.slot_mapping = torch::tensor({20, 21, 22}, torch::kInt32);
  metadata.is_prefill = true;

  const KVShardCausalSelectorMetadata selector =
      build_kv_shard_causal_selector_metadata(metadata, layout);

  EXPECT_TRUE(torch::equal(selector.block_table,
                           torch::tensor({{5}, {5}, {5}}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(selector.local_context_lens,
                           torch::tensor({0, 0, 1}, torch::kInt32)));
}

TEST(DcpIndexerCandidateTest,
     PreservesShardMetadataWithInt64CumulativeLengths) {
  const KVShardLayout layout(
      /*physical_block_size=*/2, /*dcp_size=*/2, /*dcp_rank=*/1);
  for (const bool is_prefill : {true, false}) {
    SCOPED_TRACE(is_prefill);
    AttentionMetadata metadata{};
    metadata.q_cu_seq_lens = torch::tensor({0, 3}, torch::kInt32);
    metadata.kv_cu_seq_lens = torch::tensor({0, 3}, torch::kInt64);
    metadata.kv_seq_lens = torch::tensor({3}, torch::kInt32);
    metadata.block_table = torch::tensor({{5}}, torch::kInt32);
    metadata.slot_mapping = torch::tensor({20, 21, 22}, torch::kInt32);
    metadata.is_prefill = is_prefill;
    metadata.is_chunked_prefill = !is_prefill;

    const auto shard_metadata = build_kv_shard_batch_metadata(metadata, layout);

    EXPECT_TRUE(torch::equal(shard_metadata->local_slot_mapping,
                             torch::tensor({-1, -1, 10}, torch::kInt32)));
    EXPECT_FALSE(shard_metadata->local_indexer_context_lens.defined());
    EXPECT_FALSE(shard_metadata->prefill_sorted_slots.defined());
    EXPECT_FALSE(shard_metadata->prefill_sorted_rows.defined());
    EXPECT_EQ(shard_metadata->kv_split_size, 2);
    EXPECT_EQ(shard_metadata->kv_split_rank, 1);
    EXPECT_FALSE(shard_metadata->causal_selector.block_table.defined());
    EXPECT_FALSE(shard_metadata->causal_selector.local_context_lens.defined());
    EXPECT_FALSE(shard_metadata->causal_selector.q_cu_seq_lens.defined());
    EXPECT_EQ(metadata.kv_cu_seq_lens.scalar_type(), torch::kInt64);
  }
}

TEST(DcpIndexerCandidateTest, SharedBuilderAcceptsRepeatedValidPrefillSlots) {
  AttentionMetadata metadata{};
  metadata.is_prefill = true;
  metadata.slot_mapping = torch::tensor({2, 2, -1, -1}, torch::kInt32);
  const KVShardLayout layout(
      /*physical_block_size=*/2, /*dcp_size=*/2, /*dcp_rank=*/1);
  const auto shared = build_kv_shard_batch_metadata(metadata, layout);
  EXPECT_TRUE(torch::equal(shared->local_slot_mapping,
                           torch::tensor({0, 0, -1, -1}, torch::kInt32)));
  EXPECT_FALSE(shared->prefill_sorted_slots.defined());
  EXPECT_ANY_THROW(build_prefill_slot_order(metadata.slot_mapping));
}

TEST(DcpIndexerCandidateTest, MluPrefillSortAllowsRepeatedPadding) {
  AttentionMetadata metadata{};
  metadata.is_prefill = true;
  metadata.slot_mapping = torch::tensor({3, -1, 2, -1}, torch::kInt32);
  auto [sorted_slots, sorted_rows] =
      build_prefill_slot_order(metadata.slot_mapping);
  sorted_slots = sorted_slots.to(torch::kInt64);
  EXPECT_TRUE(
      torch::equal(sorted_slots, torch::tensor({-1, -1, 2, 3}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(
      metadata.slot_mapping.index_select(0, sorted_rows).to(torch::kInt64),
      sorted_slots));
}

TEST(DcpIndexerCandidateTest, DecodeGraphRefreshesSlotsAndLengthsOnReplay) {
  const torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  AttentionMetadata metadata{};
  metadata.slot_mapping = torch::tensor({2, 3, -1, -1}, options);
  metadata.kv_seq_lens = torch::tensor({3, 4, 0, 0}, options);
  const KVShardLayout layout(/*physical_block_size=*/2,
                             /*dcp_size=*/2,
                             /*dcp_rank=*/1);
  build_kv_shard_batch_metadata(metadata, layout);
  localize_kv_shard_context_lens(metadata.kv_seq_lens, layout);
  torch_mlu::synchronize();
  torch_mlu::MLUGraph graph;
  std::shared_ptr<KVShardBatchMetadata> captured;
  {
    torch_mlu::mlu::MLUStreamGuard guard(
        torch_mlu::getStreamFromPool(/*isHighPriority=*/false, device.index()));
    graph.capture_begin();
    captured = build_kv_shard_batch_metadata(metadata, layout);
    captured->local_indexer_context_lens =
        localize_kv_shard_context_lens(metadata.kv_seq_lens, layout);
    graph.capture_end();
  }
  metadata.slot_mapping.copy_(torch::tensor({6, -1, -1, -1}, options));
  metadata.kv_seq_lens.copy_(torch::tensor({7, 0, 0, 0}, options));
  graph.replay();
  torch_mlu::synchronize();
  EXPECT_TRUE(torch::equal(captured->local_slot_mapping,
                           torch::tensor({2, -1, -1, -1}, options)));
  EXPECT_TRUE(torch::equal(captured->local_indexer_context_lens,
                           torch::tensor({3, 0, 0, 0}, options)));
  metadata.slot_mapping.copy_(torch::tensor({2, 3, 6, 7}, options));
  metadata.kv_seq_lens.copy_(torch::tensor({3, 4, 7, 8}, options));
  graph.replay();
  torch_mlu::synchronize();
  EXPECT_TRUE(torch::equal(captured->local_slot_mapping,
                           torch::tensor({0, 1, 2, 3}, options)));
  EXPECT_TRUE(torch::equal(captured->local_indexer_context_lens,
                           torch::tensor({1, 2, 3, 4}, options)));
  EXPECT_FALSE(captured->prefill_sorted_slots.defined());
  EXPECT_FALSE(captured->causal_selector.block_table.defined());
}

TEST(DcpIndexerCandidateTest, PreservesLocalCandidateWireFormatWithoutDcp) {
  const torch::Tensor local_scores =
      torch::tensor({{0.8f, 0.4f}, {0.7f, 0.1f}}, torch::kFloat32);
  const torch::Tensor local_global_slots =
      torch::tensor({{4, 12}, {5, 13}}, torch::kInt32);

  const DcpIndexerGatheredCandidates gathered =
      finish_dcp_indexer_candidate_gather(launch_dcp_indexer_candidate_gather(
          local_scores, local_global_slots, /*dcp_group=*/nullptr));

  EXPECT_TRUE(torch::equal(gathered.scores, local_scores.unsqueeze(0)));
  EXPECT_TRUE(
      torch::equal(gathered.global_slots, local_global_slots.unsqueeze(0)));
}

}  // namespace
}  // namespace xllm::layer

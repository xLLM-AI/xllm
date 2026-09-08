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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "framework/kv_cache/kv_shard_layout.h"
#include "layers/common/attention_metadata.h"
#include "layers/common/kv_shard_batch_metadata.h"

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
  AttentionMetadata metadata;
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

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

#include "layers/mlu/dcp_decode_context.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "framework/kv_cache/kv_shard_layout.h"
#include "layers/mlu/dsa_topk_state.h"

namespace xllm::layer {
namespace {

TEST(DcpDecodeContextTest, LocalizesOnlyOwnedCacheWriteSlots) {
  const DcpDecodeContext context(
      KVShardLayout(/*physical_block_size=*/4, /*dcp_size=*/2, /*dcp_rank=*/1),
      /*dcp_group=*/nullptr);
  torch::Tensor global_slots = torch::tensor({-1, 0, 3, 4, 7, 8, 12});

  torch::Tensor local_slots = context.localize_slots(global_slots);

  EXPECT_TRUE(
      torch::equal(local_slots, torch::tensor({-1, -1, -1, 0, 3, -1, 4})));
}

TEST(DcpDecodeContextTest, PacksOwnedTopkAndUpdatesEachContextLength) {
  const DcpDecodeContext context(
      KVShardLayout(/*physical_block_size=*/4, /*dcp_size=*/2, /*dcp_rank=*/0),
      /*dcp_group=*/nullptr);
  DsaTopkState global_state(
      torch::tensor({{4, 0, 8, 1, 12}, {5, 6, 7, 2, 3}}, torch::kInt32),
      torch::tensor({4, 3}, torch::kInt32));

  DsaTopkState local_state = context.localize_topk(global_state);

  EXPECT_TRUE(torch::equal(
      local_state.block_tables(),
      torch::tensor({{0, 4, 1, 0, 0}, {0, 0, 0, 0, 0}}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(local_state.context_lens(),
                           torch::tensor({3, 0}, torch::kInt32)));
}

TEST(DcpDecodeContextTest, ExpandsLogicalBlocksForReplicatedIndexerCache) {
  const DcpDecodeContext context(
      KVShardLayout(/*physical_block_size=*/4, /*dcp_size=*/2, /*dcp_rank=*/0),
      /*dcp_group=*/nullptr);
  torch::Tensor logical_blocks =
      torch::tensor({{3, 7, -1}, {0, 2, 4}}, torch::kInt32);

  torch::Tensor indexer_blocks =
      context.expand_indexer_block_table(logical_blocks);

  EXPECT_TRUE(
      torch::equal(indexer_blocks,
                   torch::tensor({{6, 7, 14, 15, -1, -1}, {0, 1, 4, 5, 8, 9}},
                                 torch::kInt32)));
}

TEST(DcpDecodeContextTest, DcpOnePreservesValidTopkEntries) {
  const DcpDecodeContext context(
      KVShardLayout(/*physical_block_size=*/4, /*dcp_size=*/1, /*dcp_rank=*/0),
      /*dcp_group=*/nullptr);
  DsaTopkState global_state(torch::tensor({{7, 3, 9, 11}}, torch::kInt32),
                            torch::tensor({2}, torch::kInt32));

  DsaTopkState local_state = context.localize_topk(global_state);

  EXPECT_TRUE(torch::equal(local_state.block_tables(),
                           torch::tensor({{7, 3, 0, 0}}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(local_state.context_lens(),
                           torch::tensor({2}, torch::kInt32)));
}

}  // namespace
}  // namespace xllm::layer

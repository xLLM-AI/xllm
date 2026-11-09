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

#include "core/disaggregation/pd_kv_response.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

#include "core/framework/request/sequence_kv_state.h"
#include "core/kv_cache/block/block.h"

namespace xllm {
namespace {

std::vector<uint64_t> group_ids(const proto::KVTransferGroup& group) {
  return {group.ids().begin(), group.ids().end()};
}

TEST(PDKVResponseTest, FlatKVExportsOnlyUnsharedSuffix) {
  KVCacheState state;
  auto* blocks = state.mutable_blocks(BlockType::KV);
  blocks->emplace_back(/*id=*/17, /*allocator=*/nullptr);
  blocks->emplace_back(/*id=*/0, /*allocator=*/nullptr);
  blocks->emplace_back(/*id=*/29, /*allocator=*/nullptr);
  state.set_shared_blocks_num(BlockType::KV, /*num=*/1);
  proto::DisaggResponse response;

  append_pd_kv_transfer_groups(
      state, /*include_linear_state=*/false, &response);

  ASSERT_EQ(response.groups_size(), 1);
  const auto& group = response.groups(0);
  EXPECT_EQ(group.group_id(), cache_group_id(BlockType::KV));
  EXPECT_EQ(group.remote_shared_num(), 1u);
  EXPECT_EQ(group_ids(group), (std::vector<uint64_t>{0, 29}));
  EXPECT_EQ((*blocks)[0].ref_count(), 1u);
}

TEST(PDKVResponseTest, CollectsOnlyUnsharedFlatBlockIdsForVirtualMemory) {
  KVCacheState state;
  auto* blocks = state.mutable_blocks(BlockType::KV);
  blocks->emplace_back(/*id=*/17, /*allocator=*/nullptr);
  blocks->emplace_back(/*id=*/0, /*allocator=*/nullptr);
  blocks->emplace_back(/*id=*/29, /*allocator=*/nullptr);
  state.set_shared_blocks_num(BlockType::KV, /*num=*/1);
  proto::DisaggResponse response;
  std::vector<int32_t> block_ids{91};
  state.mutable_blocks(BlockType::LINEAR)
      ->emplace_back(/*id=*/42, /*allocator=*/nullptr);

  append_pd_kv_transfer_groups(state,
                               /*include_linear_state=*/true,
                               &response,
                               &block_ids);

  ASSERT_EQ(response.groups_size(), 2);
  EXPECT_EQ(block_ids, (std::vector<int32_t>{91, 0, 29}));
  EXPECT_EQ(group_ids(response.groups(0)), (std::vector<uint64_t>{0, 29}));
  EXPECT_EQ(group_ids(response.groups(1)), (std::vector<uint64_t>{42}));
}

TEST(PDKVResponseTest, FlatKVRetainsEmptyGroupForFullySharedPrompt) {
  KVCacheState state;
  state.mutable_blocks(BlockType::KV)
      ->emplace_back(/*id=*/7,
                     /*allocator=*/nullptr);
  state.set_shared_blocks_num(BlockType::KV, /*num=*/1);
  proto::DisaggResponse response;
  std::vector<int32_t> block_ids;

  append_pd_kv_transfer_groups(
      state, /*include_linear_state=*/false, &response, &block_ids);

  EXPECT_TRUE(block_ids.empty());
  ASSERT_EQ(response.groups_size(), 1);
  EXPECT_EQ(response.groups(0).remote_shared_num(), 1u);
  EXPECT_TRUE(response.groups(0).ids().empty());
}

TEST(PDKVResponseTest, GroupedCachesPreserveOrderPrefixesAndSWAGaps) {
  KVCacheState state;
  state.mutable_blocks(BlockType::C128)
      ->emplace_back(/*id=*/128,
                     /*allocator=*/nullptr);
  state.mutable_blocks(BlockType::C4)
      ->emplace_back(/*id=*/4,
                     /*allocator=*/nullptr);
  auto* swa_blocks = state.mutable_blocks(BlockType::SWA);
  swa_blocks->emplace_back();
  swa_blocks->emplace_back(/*id=*/8, /*allocator=*/nullptr);
  state.set_shared_blocks_num(BlockType::C4, /*num=*/1);
  state.mutable_blocks(BlockType::EMBEDDING)
      ->emplace_back(
          /*id=*/99, /*allocator=*/nullptr);
  state.mutable_blocks(BlockType::LINEAR)
      ->emplace_back(/*id=*/42,
                     /*allocator=*/nullptr);
  proto::DisaggResponse response;

  std::vector<int32_t> block_ids;
  append_pd_kv_transfer_groups(
      state, /*include_linear_state=*/true, &response, &block_ids);

  EXPECT_TRUE(block_ids.empty());
  ASSERT_EQ(response.groups_size(), 4);
  EXPECT_EQ(response.groups(0).group_id(), cache_group_id(BlockType::SWA));
  EXPECT_EQ(group_ids(response.groups(0)),
            (std::vector<uint64_t>{std::numeric_limits<uint64_t>::max(), 8}));
  EXPECT_EQ(response.groups(1).group_id(), cache_group_id(BlockType::C4));
  EXPECT_EQ(response.groups(1).remote_shared_num(), 1u);
  EXPECT_EQ(group_ids(response.groups(1)), (std::vector<uint64_t>{4}));
  EXPECT_EQ(response.groups(2).group_id(), cache_group_id(BlockType::C128));
  EXPECT_EQ(group_ids(response.groups(2)), (std::vector<uint64_t>{128}));
  EXPECT_EQ(response.groups(3).group_id(), cache_group_id(BlockType::LINEAR));
  EXPECT_EQ(group_ids(response.groups(3)), (std::vector<uint64_t>{42}));
}

TEST(PDKVResponseTest, IgnoresEmptyGroupedLeavesAndUnrequestedLinearSlot) {
  KVCacheState state;
  state.mutable_blocks(BlockType::SWA);
  state.mutable_blocks(BlockType::LINEAR)
      ->emplace_back(/*id=*/42,
                     /*allocator=*/nullptr);
  proto::DisaggResponse response;

  append_pd_kv_transfer_groups(
      state, /*include_linear_state=*/false, &response);

  ASSERT_EQ(response.groups_size(), 1);
  EXPECT_EQ(response.groups(0).group_id(), cache_group_id(BlockType::KV));
  EXPECT_TRUE(response.groups(0).ids().empty());
}

TEST(PDKVResponseDeathTest, RejectsInvalidNonSWABlocks) {
  KVCacheState state;
  state.mutable_blocks(BlockType::C4)->emplace_back();
  proto::DisaggResponse response;

  EXPECT_DEATH(append_pd_kv_transfer_groups(
                   state, /*include_linear_state=*/false, &response),
               "non-SWA leaf");
}

TEST(PDKVResponseDeathTest, RequiresRequestedLinearStateSlot) {
  KVCacheState state;
  proto::DisaggResponse response;

  EXPECT_DEATH(append_pd_kv_transfer_groups(
                   state, /*include_linear_state=*/true, &response),
               "linear-state slot");
}

}  // namespace
}  // namespace xllm

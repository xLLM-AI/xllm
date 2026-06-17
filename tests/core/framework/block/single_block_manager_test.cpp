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

#include "single_block_manager.h"

#include <gtest/gtest.h>

namespace xllm {

TEST(SingleBlockManagerTest, AllocateAndFreeRoundTrip) {
  // id 0 is reserved for padding (matching BlockManagerImpl), so a pool sized
  // for 4 physical slots exposes 3 usable blocks.
  SingleBlockManager manager(4, "single");

  EXPECT_EQ(manager.num_total_blocks(), 3);
  EXPECT_EQ(manager.num_blocks_in_prefix_cache(), 0);
  EXPECT_EQ(manager.num_free_blocks(), 3);
  EXPECT_EQ(manager.num_used_blocks(), 0);
  EXPECT_DOUBLE_EQ(manager.kv_cache_utilization(), 0.0);

  auto blocks = manager.allocate(2);
  ASSERT_EQ(blocks.size(), 2);
  for (const auto& block : blocks) {
    EXPECT_GE(block.id(), 1) << "padding slot 0 must never be allocated";
  }
  EXPECT_EQ(manager.num_free_blocks(), 1);
  EXPECT_EQ(manager.num_used_blocks(), 2);
  EXPECT_DOUBLE_EQ(manager.kv_cache_utilization(), 2.0 / 3.0);

  manager.deallocate(Slice<Block>(blocks));
  EXPECT_EQ(manager.num_free_blocks(), 1);
  EXPECT_EQ(manager.num_used_blocks(), 0);

  blocks.clear();
  EXPECT_EQ(manager.num_free_blocks(), 3);
  EXPECT_EQ(manager.num_used_blocks(), 0);
  EXPECT_DOUBLE_EQ(manager.kv_cache_utilization(), 0.0);
}

TEST(SingleBlockManagerTest, AllocateReturnsEmptyWhenExhausted) {
  // id 0 is reserved internally, so 3 physical slots expose 2 usable blocks.
  SingleBlockManager manager(3, "single");
  auto blocks = manager.allocate(2);
  ASSERT_EQ(blocks.size(), 2);
  EXPECT_TRUE(manager.allocate(1).empty());
}

TEST(SingleBlockManagerTest, AllocateSingleDiesWhenExhausted) {
  // id 0 is reserved internally, so 2 physical slots expose a single usable
  // block, whose id is 1.
  SingleBlockManager manager(2, "single");
  Block block = manager.allocate();
  EXPECT_EQ(block.id(), 1);
  EXPECT_DEATH(manager.allocate(), "No more single blocks available");
}

TEST(SingleBlockManagerTest, PrefixCacheStyleApisAreSafeNoopsWhenDisabled) {
  SingleBlockManager manager(2, "single");

  const int32_t token_ids_arr[] = {1, 2, 3};
  const Slice<int32_t> token_ids(token_ids_arr, 3);

  // allocate_shared should be safely substitutable with BlockManager behavior
  // when prefix cache is disabled.
  const auto shared = manager.allocate_shared(token_ids);
  EXPECT_TRUE(shared.empty());

  std::vector<Block> blocks = manager.allocate(1);
  ASSERT_EQ(blocks.size(), 1u);

  // cache() overloads should be safe no-ops.
  EXPECT_NO_FATAL_FAILURE(manager.cache(token_ids, blocks));
  EXPECT_NO_FATAL_FAILURE(manager.cache(blocks));

  // get_merged_kvcache_event should be a safe no-op.
  KvCacheEvent event;
  // Avoid inserting a default-constructed XXH3Key, whose bytes would be
  // uninitialized and could trigger undefined behavior in hashing.
  uint8_t key_bytes[XXH3_128BITS_HASH_VALUE_LEN] = {};
  event.stored_cache.emplace(key_bytes);
  EXPECT_NO_FATAL_FAILURE(manager.get_merged_kvcache_event(&event));
  EXPECT_EQ(event.stored_cache.size(), 1u);
}

TEST(SingleBlockManagerTest, UsedBlocksAccountingDoesNotLeakWithAliases) {
  // id 0 is reserved internally, so 2 physical slots expose a single usable
  // block.
  SingleBlockManager manager(2, "single");
  EXPECT_EQ(manager.num_used_blocks(), 0u);
  EXPECT_EQ(manager.num_free_blocks(), 1u);

  {
    Block block = manager.allocate();
    EXPECT_EQ(manager.num_used_blocks(), 1u);
    EXPECT_EQ(manager.num_free_blocks(), 0u);

    // Create an alias to simulate an external holder (e.g. prefix-cache style
    // reference) that outlives the "sequence-owned" reference.
    Block alias = block;
    EXPECT_EQ(block.ref_count(), 2u);

    // Deallocate the sequence-owned reference while an alias still exists.
    manager.deallocate(Slice<Block>(&block, 1));
    EXPECT_EQ(manager.num_used_blocks(), 1u);

    // Drop the sequence-owned reference without calling deallocate again.
    block = Block();
    EXPECT_EQ(alias.ref_count(), 1u);
    EXPECT_EQ(manager.num_used_blocks(), 1u);
    EXPECT_EQ(manager.num_free_blocks(), 0u);
  }

  // When the last alias is released, used block accounting should converge.
  EXPECT_EQ(manager.num_used_blocks(), 0u);
  EXPECT_EQ(manager.num_free_blocks(), 1u);
}

TEST(SingleBlockManagerTest, ReservesPaddingSlotZero) {
  // Draining the whole pool must never yield the reserved padding id 0, which
  // padded batch rows use as kPaddingLinearStateId. 4 physical slots expose 3
  // usable blocks.
  SingleBlockManager manager(4, "single");
  EXPECT_EQ(manager.num_total_blocks(), 3);

  std::vector<Block> blocks = manager.allocate(3);
  ASSERT_EQ(blocks.size(), 3u);
  for (const auto& block : blocks) {
    EXPECT_NE(block.id(), 0) << "padding slot 0 must stay reserved";
  }
  EXPECT_TRUE(manager.allocate(1).empty());

  // Releasing every usable block must not free the reserved id either.
  blocks.clear();
  EXPECT_EQ(manager.num_free_blocks(), 3u);
}

TEST(SingleBlockManagerTest, ConstructorRejectsZeroBlocks) {
  EXPECT_DEATH({ SingleBlockManager manager(0, "single"); }, "No blocks");
}

}  // namespace xllm

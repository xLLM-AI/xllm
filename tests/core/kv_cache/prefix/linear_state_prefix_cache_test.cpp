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

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/kv_cache/block/linear_state_block_manager.h"
#include "framework/block/block_manager_pool_test_peer.h"
#include "util/hash_util.h"

namespace xllm {
namespace {

// Build a distinct prefix hash keyed by a single byte so tests can reference
// checkpoints by an easy-to-read tag.
XXH3Key make_hash(uint8_t tag) {
  std::array<uint8_t, XXH3_128BITS_HASH_VALUE_LEN> bytes{};
  bytes.fill(tag);
  return XXH3Key(bytes.data());
}

}  // namespace

// Padding slot 0 is never handed out, and a cache of N physical slots exposes
// N - 1 live slots before it runs dry.
TEST(LinearStatePrefixCacheTest, PaddingSlotReservedAndLiveSlotsExhaust) {
  LinearStateBlockManager cache({.num_blocks_ = 3, .block_size_ = 16});

  const auto blocks = cache.allocate(/*num_blocks=*/2);
  ASSERT_EQ(blocks.size(), 2u);
  EXPECT_NE(blocks[0].id(), 0);
  EXPECT_NE(blocks[1].id(), 0);
  EXPECT_NE(blocks[0].id(), blocks[1].id());
  EXPECT_TRUE(cache.allocate(/*num_blocks=*/1).empty());
}

// A live slot returns to the cache when its handle is dropped.
TEST(LinearStatePrefixCacheTest, DroppingLiveHandleReleasesSlot) {
  LinearStateBlockManager cache({.num_blocks_ = 2, .block_size_ = 16});
  int32_t reused_id = -1;
  {
    const auto live = cache.allocate(/*num_blocks=*/1);
    ASSERT_EQ(live.size(), 1u);
    reused_id = live.front().id();
  }
  const auto live_again = cache.allocate(/*num_blocks=*/1);
  ASSERT_EQ(live_again.size(), 1u);
  EXPECT_EQ(live_again.front().id(), reused_id);
}

TEST(LinearStatePrefixCacheTest, ReleasedLiveSlotCanBeCached) {
  LinearStateBlockManager cache({.num_blocks_ = 2, .block_size_ = 16});
  const XXH3Key hash = make_hash(2);
  int32_t released_slot = -1;

  {
    const auto live = cache.allocate(/*num_blocks=*/1);
    ASSERT_EQ(live.size(), 1u);
    released_slot = live.front().id();
    EXPECT_EQ(BlockManagerPoolTestPeer::insert_checkpoint(&cache, hash), -1);
    EXPECT_FALSE(BlockManagerPoolTestPeer::contains(&cache, hash));
  }

  EXPECT_EQ(BlockManagerPoolTestPeer::insert_checkpoint(&cache, hash),
            released_slot);
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, hash));
}

// When all usable slots hold checkpoints, the next insert reclaims the
// least-recently-used one.
TEST(LinearStatePrefixCacheTest, InsertEvictsLeastRecentlyUsed) {
  // 3 physical slots -> 2 usable checkpoint slots.
  LinearStateBlockManager cache({.num_blocks_ = 3, .block_size_ = 16});
  const XXH3Key h1 = make_hash(1);
  const XXH3Key h2 = make_hash(2);
  const XXH3Key h3 = make_hash(3);

  BlockManagerPoolTestPeer::insert_checkpoint(&cache, h1);
  BlockManagerPoolTestPeer::insert_checkpoint(&cache, h2);
  ASSERT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h1));
  ASSERT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h2));

  // Cache full: a third checkpoint evicts the oldest (h1).
  BlockManagerPoolTestPeer::insert_checkpoint(&cache, h3);
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(&cache, h1));
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h2));
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h3));
}

// match() refreshes LRU order so the touched checkpoint survives eviction.
TEST(LinearStatePrefixCacheTest, MatchRefreshesLruOrder) {
  LinearStateBlockManager cache({.num_blocks_ = 3, .block_size_ = 16});
  const XXH3Key h1 = make_hash(1);
  const XXH3Key h2 = make_hash(2);
  const XXH3Key h3 = make_hash(3);

  const int32_t first_slot =
      BlockManagerPoolTestPeer::insert_checkpoint(&cache, h1);
  BlockManagerPoolTestPeer::insert_checkpoint(&cache, h2);
  auto matched = BlockManagerPoolTestPeer::match(&cache, h1);
  ASSERT_EQ(matched.size(), 1u);
  EXPECT_EQ(matched.front().id(), first_slot);
  Block::release(std::move(matched));

  BlockManagerPoolTestPeer::insert_checkpoint(&cache, h3);
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h1));
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(&cache, h2));
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h3));
}

// allocate() reclaims a checkpoint slot when no slot is free.
TEST(LinearStatePrefixCacheTest, AcquireLiveSlotReclaimsCheckpoint) {
  // 2 physical slots -> 1 usable slot.
  LinearStateBlockManager cache({.num_blocks_ = 2, .block_size_ = 16});
  const XXH3Key h1 = make_hash(1);

  const int32_t checkpoint_slot =
      BlockManagerPoolTestPeer::insert_checkpoint(&cache, h1);
  ASSERT_GE(checkpoint_slot, 1);
  ASSERT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h1));

  // The only usable slot holds a checkpoint; acquiring a live slot must reclaim
  // it.
  const auto live = cache.allocate(/*num_blocks=*/1);
  ASSERT_EQ(live.size(), 1u);
  EXPECT_EQ(live.front().id(), checkpoint_slot);
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(&cache, h1));
}

TEST(LinearStatePrefixCacheTest, AcquireLiveSlotSkipsPinnedCheckpoint) {
  LinearStateBlockManager cache({.num_blocks_ = 3, .block_size_ = 16});
  const XXH3Key h1 = make_hash(1);
  const XXH3Key h2 = make_hash(2);

  const int32_t pinned_slot =
      BlockManagerPoolTestPeer::insert_checkpoint(&cache, h1);
  ASSERT_GE(pinned_slot, 1);
  auto restore_pin = BlockManagerPoolTestPeer::match(&cache, h1);
  ASSERT_FALSE(restore_pin.empty());
  EXPECT_EQ(restore_pin.front().id(), pinned_slot);

  const int32_t reclaimable_slot =
      BlockManagerPoolTestPeer::insert_checkpoint(&cache, h2);
  ASSERT_GE(reclaimable_slot, 1);

  const auto live = cache.allocate(/*num_blocks=*/1);
  ASSERT_EQ(live.size(), 1u);
  EXPECT_EQ(live.front().id(), reclaimable_slot);
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h1));
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(&cache, h2));
  Block::release(std::move(restore_pin));
}

TEST(LinearStatePrefixCacheTest, InsertFailsWhenAllCheckpointsPinned) {
  LinearStateBlockManager cache({.num_blocks_ = 3, .block_size_ = 16});
  const XXH3Key h1 = make_hash(1);
  const XXH3Key h2 = make_hash(2);
  const XXH3Key h3 = make_hash(3);

  BlockManagerPoolTestPeer::insert_checkpoint(&cache, h1);
  BlockManagerPoolTestPeer::insert_checkpoint(&cache, h2);

  auto restore_pin1 = BlockManagerPoolTestPeer::match(&cache, h1);
  auto restore_pin2 = BlockManagerPoolTestPeer::match(&cache, h2);
  ASSERT_FALSE(restore_pin1.empty());
  ASSERT_FALSE(restore_pin2.empty());

  EXPECT_EQ(BlockManagerPoolTestPeer::insert_checkpoint(&cache, h3), -1);
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h1));
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h2));
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(&cache, h3));
  Block::release(std::move(restore_pin1));
  Block::release(std::move(restore_pin2));
}

TEST(LinearStatePrefixCacheTest, ReleasedPinAllowsCheckpointReclaim) {
  LinearStateBlockManager cache({.num_blocks_ = 2, .block_size_ = 16});
  const XXH3Key h1 = make_hash(1);
  const XXH3Key h2 = make_hash(2);

  const int32_t slot = BlockManagerPoolTestPeer::insert_checkpoint(&cache, h1);
  ASSERT_GE(slot, 1);
  auto restore_pin = BlockManagerPoolTestPeer::match(&cache, h1);
  ASSERT_FALSE(restore_pin.empty());
  EXPECT_EQ(BlockManagerPoolTestPeer::insert_checkpoint(&cache, h2), -1);
  Block::release(std::move(restore_pin));

  EXPECT_EQ(BlockManagerPoolTestPeer::insert_checkpoint(&cache, h2), slot);
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(&cache, h1));
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h2));
}

}  // namespace xllm

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

  Block first = cache.allocate();
  Block second = cache.allocate();
  ASSERT_TRUE(first.is_valid());
  ASSERT_TRUE(second.is_valid());
  EXPECT_GE(first.id(), 1);
  EXPECT_GE(second.id(), 1);
  EXPECT_NE(first.id(), second.id());

  // No free slot and no checkpoint to reclaim -> invalid handle.
  Block third = cache.allocate();
  EXPECT_FALSE(third.is_valid());
}

// A live slot returns to the cache when its handle is dropped.
TEST(LinearStatePrefixCacheTest, DroppingLiveHandleReleasesSlot) {
  LinearStateBlockManager cache({.num_blocks_ = 2, .block_size_ = 16});
  int32_t reused_id = -1;
  {
    Block live = cache.allocate();
    ASSERT_TRUE(live.is_valid());
    reused_id = live.id();
    // Only one usable slot; it is now live.
    EXPECT_FALSE(cache.allocate().is_valid());
  }
  // Handle dropped: the slot is free again.
  Block live_again = cache.allocate();
  ASSERT_TRUE(live_again.is_valid());
  EXPECT_EQ(live_again.id(), reused_id);
}

TEST(LinearStatePrefixCacheTest, MatchesCachedCheckpoint) {
  LinearStateBlockManager cache({.num_blocks_ = 4, .block_size_ = 16});
  const XXH3Key h1 = make_hash(1);
  const int32_t slot = BlockManagerPoolTestPeer::insert_checkpoint(&cache, h1);
  auto pinned = BlockManagerPoolTestPeer::match(&cache, h1);
  ASSERT_EQ(pinned.size(), 1u);
  EXPECT_EQ(pinned.front().id(), slot);
  EXPECT_TRUE(pinned.front().is_shared());
  Block::release(std::move(pinned));

  EXPECT_TRUE(
      BlockManagerPoolTestPeer::match(&cache, make_hash(/*tag=*/2)).empty());
}

TEST(LinearStatePrefixCacheTest, ReleasedLiveSlotCanBeCached) {
  LinearStateBlockManager cache({.num_blocks_ = 2, .block_size_ = 16});
  const XXH3Key hash = make_hash(2);
  int32_t released_slot = -1;

  {
    Block slot_block = cache.allocate();
    ASSERT_TRUE(slot_block.is_valid());
    released_slot = slot_block.id();
    ASSERT_GE(released_slot, 1);
    EXPECT_EQ(BlockManagerPoolTestPeer::insert_checkpoint(&cache, hash), -1);
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

  BlockManagerPoolTestPeer::insert_checkpoint(&cache, h1);
  BlockManagerPoolTestPeer::insert_checkpoint(&cache, h2);
  // Touch h1 so h2 becomes the least-recently-used entry.
  auto matched = BlockManagerPoolTestPeer::match(&cache, h1);
  ASSERT_FALSE(matched.empty());
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
  Block live = cache.allocate();
  ASSERT_TRUE(live.is_valid());
  EXPECT_EQ(live.id(), checkpoint_slot);
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

  Block live = cache.allocate();
  ASSERT_TRUE(live.is_valid());
  EXPECT_EQ(live.id(), reclaimable_slot);
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h1));
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(&cache, h2));
  Block::release(std::move(restore_pin));
}

// Live slots are never reclaimed for checkpoints.
TEST(LinearStatePrefixCacheTest, InsertCannotReclaimLiveSlots) {
  LinearStateBlockManager cache({.num_blocks_ = 2, .block_size_ = 16});
  Block live = cache.allocate();
  ASSERT_TRUE(live.is_valid());

  // The only usable slot is live, so no checkpoint slot can be reserved.
  EXPECT_EQ(
      BlockManagerPoolTestPeer::insert_checkpoint(&cache, make_hash(/*tag=*/1)),
      -1);
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(&cache, make_hash(1)));
}

TEST(LinearStatePrefixCacheTest, InsertSkipsPinnedCheckpointSlots) {
  LinearStateBlockManager cache({.num_blocks_ = 3, .block_size_ = 16});
  const XXH3Key h1 = make_hash(1);
  const XXH3Key h2 = make_hash(2);
  const XXH3Key h3 = make_hash(3);

  const int32_t pinned_slot =
      BlockManagerPoolTestPeer::insert_checkpoint(&cache, h1);
  ASSERT_GE(pinned_slot, 1);
  ASSERT_GE(BlockManagerPoolTestPeer::insert_checkpoint(&cache, h2), 1);

  auto restore_pin = BlockManagerPoolTestPeer::match(&cache, h1);
  ASSERT_FALSE(restore_pin.empty());
  EXPECT_EQ(restore_pin.front().id(), pinned_slot);

  EXPECT_GE(BlockManagerPoolTestPeer::insert_checkpoint(&cache, h3), 1);
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h1));
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(&cache, h2));
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h3));
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

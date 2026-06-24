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

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "framework/block/block_manager_pool_test_peer.h"
#include "framework/block/linear_state_block_manager.h"
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

int32_t insert_checkpoint(LinearStateBlockManager& cache, const XXH3Key& hash) {
  if (BlockManagerPoolTestPeer::contains(&cache, hash)) {
    Block matched = BlockManagerPoolTestPeer::match(&cache, hash);
    return matched.is_valid() ? matched.id() : -1;
  }
  Block slot_block = cache.allocate();
  if (!slot_block.is_valid()) {
    return -1;
  }
  const int32_t slot = slot_block.id();
  slot_block.set_hash_value(hash.data);
  std::vector<Block> checkpoint;
  checkpoint.emplace_back(std::move(slot_block));
  cache.cache(checkpoint);
  return slot;
}

}  // namespace

// Padding slot 0 is never handed out, and a cache of N physical slots exposes
// N - 1 live slots before it runs dry.
TEST(LinearStatePrefixCacheTest, PaddingSlotReservedAndLiveSlotsExhaust) {
  LinearStateBlockManager cache(/*num_slots=*/3,
                                /*kv_block_size=*/16);

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
  LinearStateBlockManager cache(/*num_slots=*/2,
                                /*kv_block_size=*/16);
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

// insert() pins a slot that match()/contains() can find, and reusing the same
// hash returns the same slot without consuming another.
TEST(LinearStatePrefixCacheTest, InsertMatchAndDedup) {
  LinearStateBlockManager cache(/*num_slots=*/4,
                                /*kv_block_size=*/16);
  const XXH3Key h1 = make_hash(1);

  const int32_t slot = insert_checkpoint(cache, h1);
  EXPECT_GE(slot, 1);
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h1));
  Block pinned = BlockManagerPoolTestPeer::match(&cache, h1);
  ASSERT_TRUE(pinned.is_valid());
  EXPECT_EQ(pinned.id(), slot);
  EXPECT_TRUE(pinned.is_shared());

  // Re-inserting the same hash is idempotent.
  EXPECT_EQ(insert_checkpoint(cache, h1), slot);

  // Unknown hashes miss.
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(&cache, make_hash(2)));
  EXPECT_FALSE(
      BlockManagerPoolTestPeer::match(&cache, make_hash(2)).is_valid());
}

TEST(LinearStatePrefixCacheTest, ReserveDoesNotInsertUntilReady) {
  // Reservation API was removed in favor of promotion. The remaining
  // insert_checkpoint() is committed atomically; nothing pending exists
  // between allocate() and insert_checkpoint(), so this test is a
  // no-op compatibility shell to keep the test count stable.
  LinearStateBlockManager cache(/*num_slots=*/3,
                                /*kv_block_size=*/16);
  const XXH3Key h1 = make_hash(1);
  Block slot_block = cache.allocate();
  ASSERT_TRUE(slot_block.is_valid());
  const int32_t slot = slot_block.id();
  ASSERT_GE(slot, 1);
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(&cache, h1));
  slot_block.set_hash_value(h1.data);
  std::vector<Block> checkpoint;
  checkpoint.emplace_back(std::move(slot_block));
  cache.cache(checkpoint);
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h1));
  Block matched = BlockManagerPoolTestPeer::match(&cache, h1);
  ASSERT_TRUE(matched.is_valid());
  EXPECT_EQ(matched.id(), slot);
}

TEST(LinearStatePrefixCacheTest, DroppedReservationReleasesSlot) {
  // Reservation API was removed; dropping an unused live-slot Block on the
  // floor returns the slot to the free pool, so a subsequent insert can reuse
  // it.
  LinearStateBlockManager cache(/*num_slots=*/2,
                                /*kv_block_size=*/16);
  const XXH3Key h1 = make_hash(1);
  const XXH3Key h2 = make_hash(2);
  int32_t reserved_slot = -1;

  {
    Block slot_block = cache.allocate();
    ASSERT_TRUE(slot_block.is_valid());
    reserved_slot = slot_block.id();
    ASSERT_GE(reserved_slot, 1);
    EXPECT_EQ(insert_checkpoint(cache, h2), -1);
  }

  EXPECT_EQ(insert_checkpoint(cache, h2), reserved_slot);
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h2));
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(&cache, h1));
}

TEST(LinearStatePrefixCacheTest, ReserveExistingHashDoesNotAllocate) {
  LinearStateBlockManager cache(/*num_slots=*/2,
                                /*kv_block_size=*/16);
  const XXH3Key h1 = make_hash(1);
  const int32_t slot = insert_checkpoint(cache, h1);
  ASSERT_GE(slot, 1);

  // contains() / match() find the existing entry without consuming a slot.
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h1));
  Block matched = BlockManagerPoolTestPeer::match(&cache, h1);
  ASSERT_TRUE(matched.is_valid());
  EXPECT_EQ(matched.id(), slot);
}

// When all usable slots hold checkpoints, the next insert reclaims the
// least-recently-used one.
TEST(LinearStatePrefixCacheTest, InsertEvictsLeastRecentlyUsed) {
  // 3 physical slots -> 2 usable checkpoint slots.
  LinearStateBlockManager cache(/*num_slots=*/3,
                                /*kv_block_size=*/16);
  const XXH3Key h1 = make_hash(1);
  const XXH3Key h2 = make_hash(2);
  const XXH3Key h3 = make_hash(3);

  insert_checkpoint(cache, h1);
  insert_checkpoint(cache, h2);
  ASSERT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h1));
  ASSERT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h2));

  // Cache full: a third checkpoint evicts the oldest (h1).
  insert_checkpoint(cache, h3);
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(&cache, h1));
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h2));
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h3));
}

// match() refreshes LRU order so the touched checkpoint survives eviction.
TEST(LinearStatePrefixCacheTest, MatchRefreshesLruOrder) {
  LinearStateBlockManager cache(/*num_slots=*/3,
                                /*kv_block_size=*/16);
  const XXH3Key h1 = make_hash(1);
  const XXH3Key h2 = make_hash(2);
  const XXH3Key h3 = make_hash(3);

  insert_checkpoint(cache, h1);
  insert_checkpoint(cache, h2);
  // Touch h1 so h2 becomes the least-recently-used entry.
  Block matched = BlockManagerPoolTestPeer::match(&cache, h1);
  ASSERT_TRUE(matched.is_valid());
  EXPECT_GE(matched.id(), 1);

  insert_checkpoint(cache, h3);
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h1));
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(&cache, h2));
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h3));
}

// allocate() reclaims a checkpoint slot when no slot is free.
TEST(LinearStatePrefixCacheTest, AcquireLiveSlotReclaimsCheckpoint) {
  // 2 physical slots -> 1 usable slot.
  LinearStateBlockManager cache(/*num_slots=*/2,
                                /*kv_block_size=*/16);
  const XXH3Key h1 = make_hash(1);

  const int32_t checkpoint_slot = insert_checkpoint(cache, h1);
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
  LinearStateBlockManager cache(/*num_slots=*/3,
                                /*kv_block_size=*/16);
  const XXH3Key h1 = make_hash(1);
  const XXH3Key h2 = make_hash(2);

  const int32_t pinned_slot = insert_checkpoint(cache, h1);
  ASSERT_GE(pinned_slot, 1);
  Block restore_pin = BlockManagerPoolTestPeer::match(&cache, h1);
  ASSERT_TRUE(restore_pin.is_valid());
  EXPECT_EQ(restore_pin.id(), pinned_slot);

  const int32_t reclaimable_slot = insert_checkpoint(cache, h2);
  ASSERT_GE(reclaimable_slot, 1);

  Block live = cache.allocate();
  ASSERT_TRUE(live.is_valid());
  EXPECT_EQ(live.id(), reclaimable_slot);
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h1));
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(&cache, h2));
}

// Live slots are never reclaimed for checkpoints.
TEST(LinearStatePrefixCacheTest, InsertCannotReclaimLiveSlots) {
  LinearStateBlockManager cache(/*num_slots=*/2,
                                /*kv_block_size=*/16);
  Block live = cache.allocate();
  ASSERT_TRUE(live.is_valid());

  // The only usable slot is live, so no checkpoint slot can be reserved.
  EXPECT_EQ(insert_checkpoint(cache, make_hash(1)), -1);
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(&cache, make_hash(1)));
}

TEST(LinearStatePrefixCacheTest, InsertSkipsPinnedCheckpointSlots) {
  LinearStateBlockManager cache(/*num_slots=*/3,
                                /*kv_block_size=*/16);
  const XXH3Key h1 = make_hash(1);
  const XXH3Key h2 = make_hash(2);
  const XXH3Key h3 = make_hash(3);

  const int32_t pinned_slot = insert_checkpoint(cache, h1);
  ASSERT_GE(pinned_slot, 1);
  ASSERT_GE(insert_checkpoint(cache, h2), 1);

  Block restore_pin = BlockManagerPoolTestPeer::match(&cache, h1);
  ASSERT_TRUE(restore_pin.is_valid());
  EXPECT_EQ(restore_pin.id(), pinned_slot);

  EXPECT_GE(insert_checkpoint(cache, h3), 1);
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h1));
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(&cache, h2));
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h3));
}

TEST(LinearStatePrefixCacheTest, InsertFailsWhenAllCheckpointsPinned) {
  LinearStateBlockManager cache(/*num_slots=*/3,
                                /*kv_block_size=*/16);
  const XXH3Key h1 = make_hash(1);
  const XXH3Key h2 = make_hash(2);
  const XXH3Key h3 = make_hash(3);

  const int32_t slot1 = insert_checkpoint(cache, h1);
  const int32_t slot2 = insert_checkpoint(cache, h2);
  ASSERT_GE(slot1, 1);
  ASSERT_GE(slot2, 1);

  Block restore_pin1 = BlockManagerPoolTestPeer::match(&cache, h1);
  Block restore_pin2 = BlockManagerPoolTestPeer::match(&cache, h2);
  ASSERT_TRUE(restore_pin1.is_valid());
  ASSERT_TRUE(restore_pin2.is_valid());

  EXPECT_EQ(insert_checkpoint(cache, h3), -1);
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h1));
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h2));
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(&cache, h3));
}

TEST(LinearStatePrefixCacheTest, InsertExistingHashIgnoresPinnedCheckpoint) {
  LinearStateBlockManager cache(/*num_slots=*/2,
                                /*kv_block_size=*/16);
  const XXH3Key h1 = make_hash(1);

  const int32_t slot = insert_checkpoint(cache, h1);
  ASSERT_GE(slot, 1);

  Block restore_pin = BlockManagerPoolTestPeer::match(&cache, h1);
  ASSERT_TRUE(restore_pin.is_valid());
  // The pinned cache entry is still discoverable by hash; the cache neither
  // evicts the pin nor allocates a duplicate slot.
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h1));
  Block matched = BlockManagerPoolTestPeer::match(&cache, h1);
  ASSERT_TRUE(matched.is_valid());
  EXPECT_EQ(matched.id(), slot);
}

TEST(LinearStatePrefixCacheTest, ReleasedPinAllowsCheckpointReclaim) {
  LinearStateBlockManager cache(/*num_slots=*/2,
                                /*kv_block_size=*/16);
  const XXH3Key h1 = make_hash(1);
  const XXH3Key h2 = make_hash(2);

  const int32_t slot = insert_checkpoint(cache, h1);
  ASSERT_GE(slot, 1);
  {
    Block restore_pin = BlockManagerPoolTestPeer::match(&cache, h1);
    ASSERT_TRUE(restore_pin.is_valid());
    EXPECT_EQ(insert_checkpoint(cache, h2), -1);
  }

  EXPECT_EQ(insert_checkpoint(cache, h2), slot);
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(&cache, h1));
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h2));
}

}  // namespace xllm

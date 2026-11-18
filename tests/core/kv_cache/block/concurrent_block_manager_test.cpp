/* Copyright 2025-2026 The xLLM Authors.

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

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "core/framework/block/composite_block_manager.h"
#include "core/framework/request/sequence_kv_state.h"
#include "core/kv_cache/block/block_manager_impl.h"
#include "core/kv_cache/block/concurrent_block_manager_impl.h"
#include "core/kv_cache/prefix/block_hasher.h"
#include "core/util/hash_util.h"

namespace xllm {
namespace {

void release_prefix_cache(ConcurrentBlockManagerImpl& manager) {
  std::vector<Block> blocks = manager.allocate(manager.num_total_blocks());
  EXPECT_EQ(blocks.size(), manager.num_total_blocks());
  manager.deallocate(blocks);
}

}  // namespace

TEST(ConcurrentBlockManagerTest, ContinuesPrefixCacheFromExistingBlocks) {
  const uint32_t block_size = 2;
  BlockManager::Options options;
  options.num_blocks(5).block_size(block_size).enable_prefix_cache(true);
  ConcurrentBlockManagerImpl manager(
      std::make_unique<BlockManagerImpl>(options));

  std::vector<int32_t> token_ids = {11, 12, 13, 14};
  std::vector<Block> seed_blocks = manager.allocate(/*num_blocks=*/2);
  ASSERT_EQ(seed_blocks.size(), 2);
  XXH3Key first_hash;
  TextBlockHasher{}.compute(token_ids,
                            /*start_token_idx=*/0,
                            block_size,
                            /*pre_hash_value=*/nullptr,
                            first_hash);
  seed_blocks.front().set_hash_value(first_hash.data);
  manager.cache(token_ids, seed_blocks, /*existed_shared_blocks_num=*/1);

  const int32_t first_block_id = seed_blocks[0].id();
  const int32_t second_block_id = seed_blocks[1].id();

  std::vector<Block> existed_blocks;
  existed_blocks.reserve(1);
  existed_blocks.emplace_back(std::move(seed_blocks[0]));

  manager.release(seed_blocks);

  ASSERT_EQ(manager.num_blocks_in_prefix_cache(), 1);

  std::vector<Block> matched_blocks =
      manager.allocate_shared(token_ids, existed_blocks);

  ASSERT_EQ(matched_blocks.size(), 2u);
  EXPECT_EQ(matched_blocks[0].id(), first_block_id);
  EXPECT_EQ(matched_blocks[1].id(), second_block_id);

  manager.deallocate(matched_blocks);
  manager.deallocate(existed_blocks);
  matched_blocks.clear();
  existed_blocks.clear();
  release_prefix_cache(manager);

  EXPECT_EQ(manager.num_blocks_in_prefix_cache(), 0);
  EXPECT_EQ(manager.num_free_blocks(), manager.num_total_blocks());
}

TEST(ConcurrentBlockManagerTest, AllocatesWhileBlocksReleaseConcurrently) {
  BlockManager::Options options;
  options.num_blocks(65).block_size(2).enable_prefix_cache(false);
  ConcurrentBlockManagerImpl manager(
      std::make_unique<BlockManagerImpl>(options));

  constexpr int32_t kNumThreads = 8;
  constexpr int32_t kNumIterations = 10000;
  std::atomic<bool> start{false};
  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(kNumThreads));

  for (int32_t i = 0; i < kNumThreads; ++i) {
    workers.emplace_back([&manager, &start, kNumIterations]() {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }

      for (int32_t iter = 0; iter < kNumIterations; ++iter) {
        std::vector<Block> blocks = manager.allocate(/*num_blocks=*/1);
        if (blocks.empty()) {
          std::this_thread::yield();
          continue;
        }

        manager.deallocate(blocks);
        blocks.clear();
      }
    });
  }

  start.store(true, std::memory_order_release);
  for (std::thread& worker : workers) {
    worker.join();
  }

  EXPECT_EQ(manager.num_free_blocks(), manager.num_total_blocks());
  EXPECT_EQ(manager.num_used_blocks(), 0);
}

TEST(ConcurrentBlockManagerTest, ReleasesEveryCachedAliasInOneOperation) {
  BlockManager::Options options;
  options.num_blocks(5).block_size(2).enable_prefix_cache(true);
  ConcurrentBlockManagerImpl manager(
      std::make_unique<BlockManagerImpl>(options));
  const std::vector<int32_t> tokens = {11, 12};
  std::vector<Block> blocks = manager.allocate(/*num_blocks=*/1);
  ASSERT_EQ(blocks.size(), 1u);
  manager.cache(tokens, blocks);
  blocks.reserve(2);
  blocks.emplace_back(blocks.front());

  manager.release(blocks);

  EXPECT_FALSE(blocks[0].is_valid());
  EXPECT_FALSE(blocks[1].is_valid());
  EXPECT_EQ(manager.num_used_blocks(), 0u);
  EXPECT_EQ(manager.num_blocks_in_prefix_cache(), 1u);
  EXPECT_EQ(manager.num_free_blocks(), manager.num_total_blocks() - 1);
  release_prefix_cache(manager);
}

TEST(ConcurrentBlockManagerTest, LeaseAndStateReleaseCachedUsageInEitherOrder) {
  for (const bool retire_lease_first : {false, true}) {
    SCOPED_TRACE(retire_lease_first);
    BlockManager::Options options;
    options.num_blocks(5).block_size(2).enable_prefix_cache(true);
    ConcurrentBlockManagerImpl manager(
        std::make_unique<BlockManagerImpl>(options));
    const std::vector<int32_t> tokens = {11, 12};
    std::vector<Block> blocks = manager.allocate(/*num_blocks=*/1);
    ASSERT_EQ(blocks.size(), 1u);
    manager.cache(tokens, blocks);
    auto lease = std::make_unique<KVCacheBlockLease>(blocks);
    KVCacheState state;
    state.add_blocks(BlockType::KV, blocks);
    blocks.clear();

    if (retire_lease_first) {
      lease.reset();
    } else {
      state.reset();
    }
    EXPECT_EQ(manager.num_used_blocks(), 1u);
    EXPECT_EQ(manager.num_blocks_in_prefix_cache(), 1u);

    lease.reset();
    state.reset();
    EXPECT_EQ(manager.num_used_blocks(), 0u);
    EXPECT_EQ(manager.num_blocks_in_prefix_cache(), 1u);
    EXPECT_EQ(manager.num_free_blocks(), manager.num_total_blocks() - 1);
    release_prefix_cache(manager);
  }
}

TEST(ConcurrentBlockManagerTest,
     LeaseAndStateRetainUncachedUsageUntilLastAliasRelease) {
  for (const bool enable_prefix_cache : {false, true}) {
    for (const bool retire_lease_first : {false, true}) {
      SCOPED_TRACE(enable_prefix_cache);
      SCOPED_TRACE(retire_lease_first);
      BlockManager::Options options;
      options.num_blocks(5).block_size(2).enable_prefix_cache(
          enable_prefix_cache);
      ConcurrentBlockManagerImpl manager(
          std::make_unique<BlockManagerImpl>(options));
      std::vector<Block> blocks = manager.allocate(/*num_blocks=*/1);
      ASSERT_EQ(blocks.size(), 1u);
      const int32_t retained_block_id = blocks.front().id();
      auto lease = std::make_unique<KVCacheBlockLease>(blocks);
      KVCacheState state;
      state.add_blocks(BlockType::KV, blocks);
      blocks.clear();

      if (retire_lease_first) {
        lease.reset();
        EXPECT_EQ(state.blocks(BlockType::KV).front().id(), retained_block_id);
      } else {
        state.reset();
      }
      EXPECT_EQ(manager.num_used_blocks(), 1u);
      EXPECT_EQ(manager.num_free_blocks(), manager.num_total_blocks() - 1);

      lease.reset();
      state.reset();
      EXPECT_EQ(manager.num_used_blocks(), 0u);
      EXPECT_EQ(manager.num_blocks_in_prefix_cache(), 0u);
      EXPECT_EQ(manager.num_free_blocks(), manager.num_total_blocks());
    }
  }
}

TEST(ConcurrentBlockManagerTest, BatchRetirementDuringStateResetClearsUsage) {
  BlockManager::Options options;
  options.num_blocks(5).block_size(2).enable_prefix_cache(true);
  ConcurrentBlockManagerImpl manager(
      std::make_unique<BlockManagerImpl>(options));
  const std::vector<int32_t> tokens = {11, 12};

  constexpr int32_t kNumIterations = 256;
  for (int32_t iteration = 0; iteration < kNumIterations; ++iteration) {
    SCOPED_TRACE(iteration);
    std::vector<Block> blocks = manager.allocate(/*num_blocks=*/1);
    ASSERT_EQ(blocks.size(), 1u);
    manager.cache(tokens, blocks);
    auto lease = std::make_unique<KVCacheBlockLease>(blocks);
    KVCacheState state;
    state.add_blocks(BlockType::KV, blocks);
    blocks.clear();
    std::atomic<bool> start{false};
    std::thread completion([&start, lease = std::move(lease)]() mutable {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      lease.reset();
    });

    start.store(true, std::memory_order_release);
    state.reset();
    completion.join();

    EXPECT_EQ(manager.num_used_blocks(), 0u);
    release_prefix_cache(manager);
    EXPECT_EQ(manager.num_blocks_in_prefix_cache(), 0u);
    EXPECT_EQ(manager.num_free_blocks(), manager.num_total_blocks());
  }
}

class LinearConcurrentAllocationTest
    : public ::testing::TestWithParam<int32_t> {};

TEST_P(LinearConcurrentAllocationTest, KeepsLiveSlotsUniqueUntilFinalRelease) {
  constexpr int32_t kNumSlots = 64;
  constexpr int32_t kNumThreads = 8;
  constexpr int32_t kNumIterations = 10000;
  BlockManager::Options options;
  options.num_blocks(128)
      .block_size(128)
      .enable_linear_state(true)
      .linear_state_num_slots(kNumSlots)
      .enable_prefix_cache(false)
      .instance_is_decode(true)
      .enable_disagg_pd(GetParam() == 0)
      .enable_kvcache_store(GetParam() == 1)
      .enable_host_offload(GetParam() == 2);
  auto leaves = build_composite_leaves(options);
  BlockManager* manager = leaves.at(BlockType::LINEAR).leaf.get();
  std::vector<std::atomic<int32_t>> live_ids(kNumSlots + 1);
  for (auto& live : live_ids) {
    live.store(0, std::memory_order_relaxed);
  }
  std::atomic<bool> start{false};
  std::vector<std::thread> workers;
  workers.reserve(kNumThreads);
  for (int32_t thread_index = 0; thread_index < kNumThreads; ++thread_index) {
    workers.emplace_back([&]() {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int32_t iteration = 0; iteration < kNumIterations; ++iteration) {
        std::vector<Block> blocks = manager->allocate(1);
        ASSERT_FALSE(blocks.empty());
        const int32_t block_id = blocks.front().id();
        ASSERT_GT(block_id, 0);
        ASSERT_LE(block_id, kNumSlots);
        EXPECT_EQ(live_ids[block_id].fetch_add(1), 0);
        Block inflight = blocks.front();
        manager->deallocate(blocks);
        blocks.clear();
        EXPECT_EQ(live_ids[block_id].fetch_sub(1), 1);
        inflight = Block();
      }
    });
  }
  start.store(true, std::memory_order_release);
  for (std::thread& worker : workers) {
    worker.join();
  }
  EXPECT_EQ(manager->num_free_blocks(), manager->num_total_blocks());
  EXPECT_EQ(manager->num_used_blocks(), 0u);
}

INSTANTIATE_TEST_SUITE_P(ConcurrentModes,
                         LinearConcurrentAllocationTest,
                         ::testing::Values(0, 1, 2));

TEST(ConcurrentBlockManagerTest, PrefixCacheSharedBlockLifecycleLeak) {
  const uint32_t block_size = 2;
  BlockManager::Options options;
  options.num_blocks(64).block_size(block_size).enable_prefix_cache(true);
  ConcurrentBlockManagerImpl manager(
      std::make_unique<BlockManagerImpl>(options));

  std::vector<int32_t> token_ids = {11, 12, 13, 14, 15, 16, 17, 18};
  std::vector<Block> seed = manager.allocate(/*num_blocks=*/4);
  ASSERT_EQ(seed.size(), 4u);
  manager.cache(token_ids, seed);
  manager.deallocate(seed);
  seed.clear();
  ASSERT_EQ(manager.num_blocks_in_prefix_cache(), 4u);

  constexpr int32_t kNumThreads = 4;
  constexpr int32_t kNumIterations = 50000;
  std::atomic<bool> start{false};
  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(kNumThreads));
  for (int32_t i = 0; i < kNumThreads; ++i) {
    workers.emplace_back([&manager, &token_ids, &start]() {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int32_t iter = 0; iter < kNumIterations; ++iter) {
        std::vector<Block> seq_blocks = manager.allocate_shared(token_ids);
        manager.deallocate(seq_blocks);
        seq_blocks.clear();
      }
    });
  }
  start.store(true, std::memory_order_release);
  for (std::thread& worker : workers) {
    worker.join();
  }

  release_prefix_cache(manager);
  EXPECT_EQ(manager.num_blocks_in_prefix_cache(), 0u);
  EXPECT_EQ(manager.num_free_blocks(), manager.num_total_blocks())
      << "leaked blocks: " << manager.num_blocks_in_prefix_cache()
      << " still pinned in prefix cache; num_free=" << manager.num_free_blocks()
      << " num_used=" << manager.num_used_blocks()
      << " total=" << manager.num_total_blocks();
  EXPECT_EQ(manager.num_used_blocks(), 0u);
}

}  // namespace xllm

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

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include "concurrent_block_manager_impl.h"
#include "framework/prefix_cache/prefix_cache.h"

namespace xllm {
namespace {

void release_prefix_cache(ConcurrentBlockManagerImpl* manager) {
  CHECK(manager != nullptr);
  std::vector<Block> blocks = manager->allocate(manager->num_total_blocks());
  EXPECT_EQ(blocks.size(), manager->num_total_blocks());
  manager->deallocate(blocks);
}

}  // namespace

TEST(ConcurrentBlockManagerTest, ContinuesPrefixCacheFromExistingBlocks) {
  const uint32_t block_size = 2;
  BlockManager::Options options;
  options.num_blocks(5).block_size(block_size).enable_prefix_cache(true);
  ConcurrentBlockManagerImpl manager(options);

  std::vector<int32_t> token_ids = {11, 12, 13, 14};
  std::vector<Block> seed_blocks = manager.allocate(/*num_blocks=*/2);
  ASSERT_EQ(seed_blocks.size(), 2);
  PrefixCache::compute_hash_keys(token_ids, seed_blocks);

  const int32_t first_block_id = seed_blocks[0].id();
  const int32_t second_block_id = seed_blocks[1].id();

  std::vector<Block> existed_blocks;
  existed_blocks.reserve(1);
  existed_blocks.emplace_back(std::move(seed_blocks[0]));

  std::vector<Block> tail_blocks;
  tail_blocks.reserve(1);
  tail_blocks.emplace_back(std::move(seed_blocks[1]));
  manager.cache(tail_blocks);
  manager.deallocate(tail_blocks);
  tail_blocks.clear();
  seed_blocks.clear();

  ASSERT_EQ(manager.num_blocks_in_prefix_cache(), 1);

  std::vector<Block> matched_blocks =
      manager.allocate_shared(token_ids, existed_blocks);

  EXPECT_EQ(matched_blocks.size(), 2);
  if (matched_blocks.size() == 2) {
    EXPECT_EQ(matched_blocks[0].id(), first_block_id);
    EXPECT_EQ(matched_blocks[1].id(), second_block_id);
  }

  manager.deallocate(matched_blocks);
  manager.deallocate(existed_blocks);
  matched_blocks.clear();
  existed_blocks.clear();
  release_prefix_cache(&manager);

  EXPECT_EQ(manager.num_blocks_in_prefix_cache(), 0);
  EXPECT_EQ(manager.num_free_blocks(), manager.num_total_blocks());
}

TEST(ConcurrentBlockManagerTest, AllocatesWhileBlocksReleaseConcurrently) {
  BlockManager::Options options;
  options.num_blocks(65).block_size(2).enable_prefix_cache(false);
  ConcurrentBlockManagerImpl manager(options);

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

}  // namespace xllm

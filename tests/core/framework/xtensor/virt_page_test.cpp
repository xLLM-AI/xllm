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

#include "core/framework/xtensor/virt_page.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace xllm {
namespace {

TEST(VirtPageTest, PaddingBlockIsFirstAndRemovedFromAvailableBlocks) {
  VirtPage page(/*page_id=*/0, /*page_size=*/16);
  page.init(/*block_mem_size=*/4);
  EXPECT_TRUE(page.empty());
  EXPECT_EQ(page.alloc(/*num_blocks=*/1), std::vector<int64_t>({0}));
  EXPECT_FALSE(page.empty());
  EXPECT_EQ(page.num_free_blocks(), 3);
  EXPECT_EQ(page.alloc(/*num_blocks=*/3), std::vector<int64_t>({1, 2, 3}));
  EXPECT_TRUE(page.full());
}

TEST(VirtPageTest, AllocationAndBatchFreeRestorePageCapacity) {
  VirtPage page(/*page_id=*/1, /*page_size=*/16);
  page.init(/*block_mem_size=*/4);
  std::vector<int64_t> blocks = page.alloc(/*num_blocks=*/10);
  EXPECT_EQ(blocks, std::vector<int64_t>({4, 5, 6, 7}));
  EXPECT_TRUE(page.full());
  page.free_batch(blocks);
  EXPECT_TRUE(page.empty());
  EXPECT_EQ(page.num_free_blocks(), 4);
  EXPECT_EQ(page.alloc(/*num_blocks=*/4), blocks);
}

TEST(VirtPageTest, NonDivisibleGeometrySkipsBlocksCrossingPageBoundaries) {
  std::vector<int64_t> allocated_blocks;
  for (int64_t page_id = 0; page_id < 4; ++page_id) {
    VirtPage page(page_id, /*page_size=*/12);
    page.init(/*block_mem_size=*/5);
    std::vector<int64_t> blocks = page.alloc(/*num_blocks=*/12);
    for (int64_t block : blocks) {
      EXPECT_GE(block * 5, page_id * 12);
      EXPECT_LE((block + 1) * 5, (page_id + 1) * 12);
    }
    allocated_blocks.insert(
        allocated_blocks.end(), blocks.begin(), blocks.end());
    page.free_batch(blocks);
    EXPECT_TRUE(page.empty());
  }
  EXPECT_EQ(allocated_blocks, std::vector<int64_t>({0, 1, 3, 5, 6, 8}));
}

TEST(VirtPageTest, SingleFreeMakesAllocatedBlockAvailableAgain) {
  VirtPage page(/*page_id=*/1, /*page_size=*/12);
  page.init(/*block_mem_size=*/5);
  EXPECT_EQ(page.num_free_blocks(), 1);
  EXPECT_EQ(page.alloc(/*num_blocks=*/1), std::vector<int64_t>({3}));
  EXPECT_TRUE(page.full());
  page.free(/*block_id=*/3);
  EXPECT_TRUE(page.empty());
  EXPECT_EQ(page.alloc(/*num_blocks=*/1), std::vector<int64_t>({3}));
}

}  // namespace
}  // namespace xllm

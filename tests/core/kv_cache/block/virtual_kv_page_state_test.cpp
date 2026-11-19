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

#include "core/kv_cache/block/virtual_kv_page_state.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

namespace xllm {
namespace {

const VirtualKVPageState* failure_page = nullptr;
std::vector<int64_t> expected_free_blocks_on_failure;

#if defined(__GNUC__)
__attribute__((noreturn))
#endif
void verify_free_list_before_exit() {
  std::_Exit(failure_page->get_free_blocks() == expected_free_blocks_on_failure
                 ? EXIT_SUCCESS
                 : EXIT_FAILURE);
}

void expect_batch_rejection_without_mutation(
    VirtualKVPageState& page,
    const std::vector<int64_t>& block_ids,
    const char* error_message) {
  EXPECT_EXIT(
      {
        failure_page = &page;
        expected_free_blocks_on_failure = page.get_free_blocks();
        google::InstallFailureFunction(verify_free_list_before_exit);
        page.free_batch(block_ids);
        std::_Exit(EXIT_FAILURE);
      },
      ::testing::ExitedWithCode(EXIT_SUCCESS),
      error_message);
}

TEST(VirtualKVPageStateTest, PaddingBlockIsFirstAndRemovedFromAvailableBlocks) {
  VirtualKVPageState page(/*page_id=*/0, /*page_size=*/16);
  page.init(/*block_mem_size=*/4);
  EXPECT_TRUE(page.empty());
  EXPECT_EQ(page.alloc(/*num_blocks=*/1), std::vector<int64_t>({0}));
  EXPECT_FALSE(page.empty());
  EXPECT_EQ(page.num_free_blocks(), 3);
  EXPECT_EQ(page.alloc(/*num_blocks=*/3), std::vector<int64_t>({1, 2, 3}));
  EXPECT_TRUE(page.full());
}

TEST(VirtualKVPageStateTest, AllocationAndBatchFreeRestorePageCapacity) {
  VirtualKVPageState page(/*page_id=*/1, /*page_size=*/16);
  page.init(/*block_mem_size=*/4);
  std::vector<int64_t> blocks = page.alloc(/*num_blocks=*/10);
  EXPECT_EQ(blocks, std::vector<int64_t>({4, 5, 6, 7}));
  EXPECT_TRUE(page.full());
  page.free_batch(blocks);
  EXPECT_TRUE(page.empty());
  EXPECT_EQ(page.num_free_blocks(), 4);
  EXPECT_EQ(page.alloc(/*num_blocks=*/4), blocks);
}

TEST(VirtualKVPageStateTest,
     NonDivisibleGeometrySkipsBlocksCrossingPageBoundaries) {
  std::vector<int64_t> allocated_blocks;
  for (int64_t page_id = 0; page_id < 4; ++page_id) {
    VirtualKVPageState page(page_id, /*page_size=*/12);
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

TEST(VirtualKVPageStateTest, SingleFreeMakesAllocatedBlockAvailableAgain) {
  VirtualKVPageState page(/*page_id=*/1, /*page_size=*/12);
  page.init(/*block_mem_size=*/5);
  EXPECT_EQ(page.num_free_blocks(), 1);
  EXPECT_EQ(page.alloc(/*num_blocks=*/1), std::vector<int64_t>({3}));
  EXPECT_TRUE(page.full());
  page.free(/*block_id=*/3);
  EXPECT_TRUE(page.empty());
  EXPECT_EQ(page.alloc(/*num_blocks=*/1), std::vector<int64_t>({3}));
}

TEST(VirtualKVPageStateTest, FullPageRejectsAllocationAndReusesFreeOrder) {
  VirtualKVPageState page(/*page_id=*/0, /*page_size=*/16);
  page.init(/*block_mem_size=*/4);
  EXPECT_EQ(page.alloc(/*num_blocks=*/4), std::vector<int64_t>({0, 1, 2, 3}));
  EXPECT_THROW(page.alloc(/*num_blocks=*/1), std::runtime_error);
  EXPECT_TRUE(page.full());

  page.free(/*block_id=*/3);
  page.free_batch(std::vector<int64_t>{1, 0, 2});
  EXPECT_EQ(page.alloc(/*num_blocks=*/4), std::vector<int64_t>({3, 1, 0, 2}));
}

TEST(VirtualKVPageStateTest, SingleFreeRejectsBlocksOutsidePage) {
  VirtualKVPageState page(/*page_id=*/1, /*page_size=*/12);
  page.init(/*block_mem_size=*/5);
  EXPECT_EQ(page.alloc(), std::vector<int64_t>({3}));

  for (int64_t block_id :
       std::vector<int64_t>{-1, 0, 2, 4, std::numeric_limits<int64_t>::max()}) {
    EXPECT_DEATH(page.free(block_id), "does not belong to page 1");
  }
}

TEST(VirtualKVPageStateTest, SingleFreeRejectsUnallocatedAndFreedBlocks) {
  VirtualKVPageState page(/*page_id=*/1, /*page_size=*/16);
  page.init(/*block_mem_size=*/4);
  EXPECT_DEATH(page.free(/*block_id=*/4), "is not allocated on page 1");

  EXPECT_EQ(page.alloc(), std::vector<int64_t>({4}));
  page.free(/*block_id=*/4);
  EXPECT_DEATH(page.free(/*block_id=*/4), "is not allocated on page 1");
}

TEST(VirtualKVPageStateTest, BatchFreeRejectsForeignBlockBeforeMutation) {
  VirtualKVPageState page(/*page_id=*/1, /*page_size=*/16);
  page.init(/*block_mem_size=*/4);
  EXPECT_EQ(page.alloc(/*num_blocks=*/3), std::vector<int64_t>({4, 5, 6}));

  expect_batch_rejection_without_mutation(
      page, std::vector<int64_t>{4, 3}, "does not belong to page 1");
  expect_batch_rejection_without_mutation(
      page, std::vector<int64_t>{4, 8}, "does not belong to page 1");
}

TEST(VirtualKVPageStateTest, BatchFreeRejectsDuplicateBeforeMutation) {
  VirtualKVPageState page(/*page_id=*/1, /*page_size=*/16);
  page.init(/*block_mem_size=*/4);
  EXPECT_EQ(page.alloc(/*num_blocks=*/3), std::vector<int64_t>({4, 5, 6}));

  expect_batch_rejection_without_mutation(
      page, std::vector<int64_t>{4, 5, 4}, "Duplicate blocks in free batch");
}

TEST(VirtualKVPageStateTest,
     BatchFreeRejectsUnallocatedAndFreedBlocksBeforeMutation) {
  VirtualKVPageState page(/*page_id=*/1, /*page_size=*/16);
  page.init(/*block_mem_size=*/4);
  EXPECT_EQ(page.alloc(/*num_blocks=*/3), std::vector<int64_t>({4, 5, 6}));
  expect_batch_rejection_without_mutation(
      page, std::vector<int64_t>{4, 7}, "is not allocated on page 1");

  page.free(/*block_id=*/5);
  expect_batch_rejection_without_mutation(
      page, std::vector<int64_t>{4, 5}, "is not allocated on page 1");
}

TEST(VirtualKVPageStateTest, EmptyBatchDoesNotChangePartialOrFullPage) {
  VirtualKVPageState page(/*page_id=*/0, /*page_size=*/16);
  page.init(/*block_mem_size=*/4);
  EXPECT_EQ(page.alloc(), std::vector<int64_t>({0}));
  page.free_batch({});
  EXPECT_EQ(page.get_free_blocks(), std::vector<int64_t>({1, 2, 3}));

  EXPECT_EQ(page.alloc(/*num_blocks=*/3), std::vector<int64_t>({1, 2, 3}));
  page.free_batch({});
  EXPECT_TRUE(page.full());
}

TEST(VirtualKVPageStateTest, ReallocatedBlocksCanBeFreedAgain) {
  VirtualKVPageState page(/*page_id=*/0, /*page_size=*/16);
  page.init(/*block_mem_size=*/4);
  EXPECT_EQ(page.alloc(/*num_blocks=*/4), std::vector<int64_t>({0, 1, 2, 3}));

  page.free_batch(std::vector<int64_t>{1, 3});
  EXPECT_EQ(page.alloc(/*num_blocks=*/2), std::vector<int64_t>({1, 3}));
  page.free(/*block_id=*/1);
  page.free_batch(std::vector<int64_t>{0, 2, 3});
  EXPECT_TRUE(page.empty());
  EXPECT_EQ(page.alloc(/*num_blocks=*/4), std::vector<int64_t>({1, 0, 2, 3}));
}

TEST(VirtualKVPageStateTest, InitializationRejectsInvalidGeometry) {
  EXPECT_DEATH(VirtualKVPageState::get_block_range(
                   /*page_id=*/-1, /*page_size=*/16, /*block_mem_size=*/4),
               "Page ID must be nonnegative");
  EXPECT_DEATH(VirtualKVPageState::get_block_range(
                   /*page_id=*/0, /*page_size=*/0, /*block_mem_size=*/4),
               "Page size must be positive");
  EXPECT_DEATH(VirtualKVPageState::get_block_range(
                   /*page_id=*/0, /*page_size=*/16, /*block_mem_size=*/0),
               "Block memory size must be positive");
  EXPECT_DEATH(VirtualKVPageState::get_block_range(
                   /*page_id=*/0, /*page_size=*/16, /*block_mem_size=*/17),
               "KV blocks must fit within a page");
  EXPECT_DEATH(VirtualKVPageState::get_num_blocks(
                   /*page_size=*/16, /*block_mem_size=*/0),
               "Block memory size must be positive");
}

TEST(VirtualKVPageStateTest, InitializationRejectsOverflowingRanges) {
  EXPECT_DEATH(
      VirtualKVPageState::get_block_range(std::numeric_limits<int64_t>::max(),
                                          /*page_size=*/2,
                                          /*block_mem_size=*/2),
      "Page byte range overflows");
  EXPECT_DEATH(VirtualKVPageState::get_block_range(
                   /*page_id=*/0,
                   std::numeric_limits<size_t>::max(),
                   /*block_mem_size=*/1),
               "Page block range exceeds block ID limit");
}

TEST(VirtualKVPageStateTest, LargePageOffsetKeepsPageRelativeTracking) {
  int64_t page_id = std::numeric_limits<int64_t>::max() / 2 - 1;
  VirtualKVPageState page(page_id, /*page_size=*/4);
  page.init(/*block_mem_size=*/4);
  EXPECT_EQ(page.alloc(), std::vector<int64_t>({page_id}));
  page.free(page_id);
  EXPECT_TRUE(page.empty());
  EXPECT_EQ(page.alloc(), std::vector<int64_t>({page_id}));
}

TEST(VirtualKVPageStateTest, ReinitializationDoesNotDiscardLiveAllocations) {
  VirtualKVPageState page(/*page_id=*/0, /*page_size=*/16);
  page.init(/*block_mem_size=*/4);
  EXPECT_EQ(page.alloc(), std::vector<int64_t>({0}));
  EXPECT_DEATH(page.init(/*block_mem_size=*/4), "already initialized");
}

}  // namespace
}  // namespace xllm

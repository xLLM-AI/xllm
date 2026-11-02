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

#include "core/kv_cache/layout/virtual_kv_cache_page_layout.h"

#include <gtest/gtest.h>

#include "core/kv_cache/block/virtual_kv_cache_page_state.h"

namespace xllm {
namespace {

constexpr size_t kPageSize = 2 * 1024 * 1024;

TEST(VirtualKVCachePageLayoutTest, MlaPagesCoverIdenticalBlocksWithoutOverlap) {
  const size_t key_bytes = VirtualKVCachePageLayout::tensor_block_bytes(
      {256, 128, 1, 512}, /*element_size=*/2);
  const size_t value_bytes = VirtualKVCachePageLayout::tensor_block_bytes(
      {256, 128, 1, 64}, /*element_size=*/2);
  const VirtualKVCachePageLayout layout({key_bytes, value_bytes}, kPageSize);
  EXPECT_EQ(layout.block_bytes(), 16 * 1024);
  EXPECT_EQ(layout.blocks_per_page(), 128);
  EXPECT_EQ(layout.pages_per_tensor(0), 8);
  EXPECT_EQ(layout.pages_per_tensor(1), 1);
  VirtualKVCachePageState first(/*page_id=*/0, kPageSize);
  VirtualKVCachePageState second(/*page_id=*/1, kPageSize);
  first.init(layout.block_bytes());
  second.init(layout.block_bytes());
  const auto first_blocks = first.alloc(128);
  const auto second_blocks = second.alloc(128);
  ASSERT_EQ(first_blocks.size(), 128);
  ASSERT_EQ(second_blocks.size(), 128);
  EXPECT_EQ(first_blocks.front(), 0);
  EXPECT_EQ(first_blocks.back(), 127);
  EXPECT_EQ(second_blocks.front(), 128);
  EXPECT_EQ(second_blocks.back(), 255);
  for (int64_t block : {0, 15, 16, 127, 128, 255}) {
    const size_t page = block * layout.block_bytes() / kPageSize;
    EXPECT_GE(block * key_bytes, page * 8 * kPageSize);
    EXPECT_LE((block + 1) * key_bytes, (page + 1) * 8 * kPageSize);
    EXPECT_GE(block * value_bytes, page * kPageSize);
    EXPECT_LE((block + 1) * value_bytes, (page + 1) * kPageSize);
  }
  first.free_batch(first_blocks);
  EXPECT_TRUE(first.empty());
  EXPECT_TRUE(second.full());
  EXPECT_EQ(first.alloc(128), first_blocks);
}

TEST(VirtualKVCachePageLayoutTest, SymmetricCacheKeepsOnePagePerTensor) {
  const VirtualKVCachePageLayout layout({128 * 1024, 128 * 1024}, kPageSize);
  EXPECT_EQ(layout.blocks_per_page(), 16);
  EXPECT_EQ(layout.pages_per_tensor(0), 1);
  EXPECT_EQ(layout.pages_per_tensor(1), 1);
}

TEST(VirtualKVCachePageLayoutTest, TargetAndDraftShareOneLogicalPageGeometry) {
  // The target alone covers 16 blocks per page and the draft covers 64.
  // Their shared group must cover the same 64 block IDs in every tensor.
  const VirtualKVCachePageLayout layout(
      {128 * 1024, 128 * 1024, 32 * 1024, 32 * 1024}, kPageSize);
  EXPECT_EQ(layout.block_bytes(), 32 * 1024);
  EXPECT_EQ(layout.blocks_per_page(), 64);
  EXPECT_EQ(layout.physical_pages({8, 8, 1, 1}), 66);
  EXPECT_EQ(layout.block_capacity(/*physical_page_budget=*/65, {8, 8, 1, 1}),
            0);
  EXPECT_EQ(layout.block_capacity(/*physical_page_budget=*/66, {8, 8, 1, 1}),
            64);
  EXPECT_EQ(layout.block_capacity(/*physical_page_budget=*/131, {8, 8, 1, 1}),
            64);
  EXPECT_EQ(layout.block_capacity(/*physical_page_budget=*/132, {8, 8, 1, 1}),
            128);
  for (size_t block : {size_t{0}, size_t{63}, size_t{64}, size_t{127}}) {
    const size_t logical_page = block / layout.blocks_per_page();
    for (size_t tensor_index = 0; tensor_index < 4; ++tensor_index) {
      const size_t tensor_bytes = tensor_index < 2 ? 128 * 1024 : 32 * 1024;
      const size_t pages = layout.pages_per_tensor(tensor_index);
      EXPECT_GE(block * tensor_bytes, logical_page * pages * kPageSize);
      EXPECT_LE((block + 1) * tensor_bytes,
                (logical_page + 1) * pages * kPageSize);
    }
  }
}

TEST(VirtualKVCachePageLayoutTest,
     CombinedBudgetFitsOnlyCompleteMappingGroups) {
  const VirtualKVCachePageLayout target({128 * 1024, 128 * 1024}, kPageSize);
  const VirtualKVCachePageLayout combined(
      {128 * 1024, 128 * 1024, 32 * 1024, 32 * 1024}, kPageSize);
  constexpr size_t kPoolPages = 131;
  // Target-only budgeting would advertise 128 blocks, requiring 132 pages
  // once the draft tensors are included. The pair must expose just 64.
  EXPECT_EQ(target.block_capacity(kPoolPages, {8, 8}), 128);
  const size_t blocks = combined.block_capacity(kPoolPages, {8, 8, 1, 1});
  ASSERT_EQ(blocks, 64);
  const size_t groups = blocks / combined.blocks_per_page();
  const size_t pages = combined.physical_pages({8, 8, 1, 1});
  EXPECT_LE(groups * pages, kPoolPages);
  EXPECT_GT((groups + 1) * pages, kPoolPages);
}

TEST(VirtualKVCachePageLayoutTest, Glm53CountsOnlyOwnedIndexerLayers) {
  const size_t key_bytes = VirtualKVCachePageLayout::tensor_block_bytes(
      {256, 128, 1, 512}, /*element_size=*/2);
  const size_t value_bytes = VirtualKVCachePageLayout::tensor_block_bytes(
      {256, 128, 1, 64}, /*element_size=*/2);
  const size_t index_bytes = VirtualKVCachePageLayout::tensor_block_bytes(
      {256, 1, 128, 128}, /*element_size=*/2);
  const VirtualKVCachePageLayout layout({key_bytes, value_bytes, index_bytes},
                                        kPageSize);
  EXPECT_EQ(layout.blocks_per_page(), 128);
  EXPECT_EQ(layout.pages_per_tensor(2), 2);
  EXPECT_EQ(layout.physical_pages({78, 78, 21}), 744);
  EXPECT_EQ(layout.physical_pages({78, 78, 0}), 702);
  EXPECT_EQ(layout.physical_pages({78, 78, 78}), 858);
  // Every role covers the same block IDs even at its hardware-page boundary.
  for (size_t block : {size_t{0},
                       size_t{63},
                       size_t{64},
                       size_t{127},
                       size_t{128},
                       size_t{255}}) {
    const size_t logical_page = block / layout.blocks_per_page();
    EXPECT_GE(block * index_bytes, logical_page * 2 * kPageSize);
    EXPECT_LE((block + 1) * index_bytes, (logical_page + 1) * 2 * kPageSize);
  }
}

TEST(VirtualKVCachePageLayoutTest, PhysicalPageBudgetRejectsOverflow) {
  const VirtualKVCachePageLayout layout({128 * 1024, 16 * 1024, 32 * 1024},
                                        kPageSize);
  EXPECT_DEATH(layout.physical_pages({78, 78}), "tensor_counts.size");
  EXPECT_DEATH(
      layout.physical_pages({std::numeric_limits<size_t>::max(), 0, 0}),
      "Check failed");
}

TEST(VirtualKVCachePageLayoutTest, NzAndNdLayoutsHaveIdenticalBlockBytes) {
  EXPECT_EQ(VirtualKVCachePageLayout::tensor_block_bytes({129, 128, 1, 512},
                                                         /*element_size=*/2),
            VirtualKVCachePageLayout::tensor_block_bytes({129, 32, 128, 16},
                                                         /*element_size=*/2));
}

TEST(VirtualKVCachePageLayoutTest, RejectsBlocksThatCannotTileHardwarePages) {
  EXPECT_DEATH((VirtualKVCachePageLayout({72 * 1024, 72 * 1024}, kPageSize)),
               "common block size");
}

}  // namespace
}  // namespace xllm

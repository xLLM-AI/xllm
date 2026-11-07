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

#include "core/kv_cache/block/static_kv_cache_block_pool.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "core/kv_cache/layout/static_kv_cache_layout.h"

namespace xllm {
namespace {

StaticKVCacheLayoutSpec make_spec(size_t num_slabs = 9,
                                  size_t checkpoint_stride = 2) {
  StaticKVCacheLayoutSpec spec;
  spec.num_slots = 64;
  spec.num_slabs = num_slabs;
  spec.full_row_bytes = 16;
  spec.conv_row_bytes = 8;
  spec.ssm_row_bytes = 32;
  spec.checkpoint_stride = checkpoint_stride;
  spec.dtype = torch::kFloat32;
  spec.ssm_dtype = torch::kFloat32;
  spec.full_layer_ids = {0, 3};
  spec.linear_layer_ids = {1, 2};
  return spec;
}

struct ByteExtent {
  size_t begin;
  size_t end;
};

TEST(StaticKVCacheBlockPoolTest, PacksMixedResourcesWithoutOverlappingBytes) {
  const StaticKVCacheLayout layout(make_spec());
  StaticKVCacheBlockPool pool(/*logical_capacity=*/64, layout);
  const auto full = pool.allocate(/*num_blocks=*/3, BlockType::KV);
  const auto linear = pool.allocate(/*num_blocks=*/1, BlockType::LINEAR);
  ASSERT_EQ(full.size(), 3);
  ASSERT_EQ(linear.size(), 1);
  // The two stable logical ID spaces are independent.
  EXPECT_EQ(full.front(), linear.front());
  std::vector<ByteExtent> extents;
  extents.reserve(full.size() * layout.full_layer_count() +
                  2 * layout.linear_layer_count());
  for (const int32_t id : full) {
    std::vector<int32_t> rows;
    pool.append_full_block_rows(id, &rows);
    ASSERT_EQ(rows.size(), layout.full_layer_count());
    for (const int32_t row : rows) {
      const size_t begin = row * layout.full_row_bytes();
      extents.emplace_back(ByteExtent{begin, begin + layout.full_pair_bytes()});
      EXPECT_EQ(begin % layout.full_pair_bytes(), 0);
    }
  }
  std::vector<int32_t> state_rows;
  pool.append_linear_state_rows(linear.front(), &state_rows);
  ASSERT_EQ(state_rows.size(), 2 * layout.linear_layer_count());
  for (size_t layer = 0; layer < layout.linear_layer_count(); ++layer) {
    const size_t conv_begin = state_rows[2 * layer] * layout.conv_row_bytes();
    const size_t ssm_begin = state_rows[2 * layer + 1] * layout.ssm_row_bytes();
    extents.emplace_back(
        ByteExtent{conv_begin, conv_begin + layout.conv_row_bytes()});
    extents.emplace_back(
        ByteExtent{ssm_begin, ssm_begin + layout.ssm_bundle_bytes()});
    EXPECT_EQ(ssm_begin % layout.ssm_bundle_bytes(), 0);
    EXPECT_LE(ssm_begin + layout.ssm_bundle_bytes(), layout.bytes_used());
  }
  std::sort(extents.begin(),
            extents.end(),
            [](const ByteExtent& left, const ByteExtent& right) {
              return left.begin < right.begin;
            });
  EXPECT_GE(extents.front().begin, layout.slab_bytes());
  for (size_t extent = 1; extent < extents.size(); ++extent) {
    EXPECT_LE(extents[extent - 1].end, extents[extent].begin);
  }
  EXPECT_EQ(pool.slots_per_block(BlockType::KV), 64);
  EXPECT_EQ(pool.slots_per_block(BlockType::LINEAR), 144);
  EXPECT_EQ(pool.num_used_slots(), 3 * 64 + 144);
  EXPECT_EQ(pool.num_free_slots(), 8 * 64 - (3 * 64 + 144));
}

TEST(StaticKVCacheBlockPoolTest, ReusesCompletelyFreedSlabsForAnotherType) {
  const StaticKVCacheLayout layout(make_spec(/*num_slabs=*/4));
  StaticKVCacheBlockPool pool(/*logical_capacity=*/64, layout);
  const auto full = pool.allocate(/*num_blocks=*/3, BlockType::KV);
  ASSERT_EQ(full.size(), 3);
  EXPECT_EQ(pool.num_free_blocks(BlockType::KV), 0);
  EXPECT_TRUE(pool.allocate(/*num_blocks=*/1, BlockType::LINEAR).empty());
  for (const int32_t id : full) {
    pool.release(id, BlockType::KV);
  }
  EXPECT_EQ(pool.num_free_slots(), pool.num_total_slots());
  const auto linear = pool.allocate(/*num_blocks=*/1, BlockType::LINEAR);
  ASSERT_EQ(linear.size(), 1);
  EXPECT_EQ(pool.num_free_blocks(BlockType::KV), 0);
  pool.release(linear.front(), BlockType::LINEAR);
  EXPECT_EQ(pool.num_free_blocks(BlockType::KV), 3);
}

TEST(StaticKVCacheBlockPoolTest, RollsBackCoupledLinearAllocationOnExhaustion) {
  const StaticKVCacheLayout layout(make_spec(/*num_slabs=*/3));
  StaticKVCacheBlockPool pool(/*logical_capacity=*/64, layout);
  EXPECT_EQ(pool.num_free_blocks(BlockType::LINEAR), 0);
  EXPECT_TRUE(pool.allocate(/*num_blocks=*/1, BlockType::LINEAR).empty());
  EXPECT_EQ(pool.num_used_slots(), 0);
  EXPECT_EQ(pool.num_free_slots(), pool.num_total_slots());
  EXPECT_EQ(pool.num_free_blocks(BlockType::KV), 2);
  const auto full = pool.allocate(/*num_blocks=*/2, BlockType::KV);
  ASSERT_EQ(full.size(), 2);
  EXPECT_EQ(full, (std::vector<int32_t>{1, 2}));
}

TEST(StaticKVCacheBlockPoolTest, RollsBackWholeFullBatchOnExhaustion) {
  const StaticKVCacheLayout layout(
      make_spec(/*num_slabs=*/2, /*checkpoint_stride=*/4));
  StaticKVCacheBlockPool pool(/*logical_capacity=*/64, layout);
  EXPECT_EQ(pool.num_free_blocks(BlockType::KV), 2);
  EXPECT_TRUE(pool.allocate(/*num_blocks=*/3, BlockType::KV).empty());
  EXPECT_EQ(pool.num_free_blocks(BlockType::KV), 2);
  EXPECT_EQ(pool.num_free_slots(), pool.num_total_slots());
  const auto full = pool.allocate(/*num_blocks=*/2, BlockType::KV);
  ASSERT_EQ(full.size(), 2);
  EXPECT_EQ(full, (std::vector<int32_t>{1, 2}));
  EXPECT_EQ(pool.num_free_slots(), 0);
}

TEST(StaticKVCacheBlockPoolTest, InactiveOwnershipKeepsPhysicalRowsPinned) {
  const StaticKVCacheLayout layout(
      make_spec(/*num_slabs=*/2, /*checkpoint_stride=*/4));
  StaticKVCacheBlockPool pool(/*logical_capacity=*/64, layout);
  const auto first = pool.allocate(/*num_blocks=*/1, BlockType::KV);
  ASSERT_EQ(first.size(), 1);
  std::vector<int32_t> original_rows;
  pool.append_full_block_rows(first.front(), &original_rows);
  pool.mark_inactive(first.front(), BlockType::KV);
  pool.mark_inactive(first.front(), BlockType::KV);
  EXPECT_EQ(pool.num_used_slots(), 0);
  EXPECT_EQ(pool.num_free_slots(), 64);
  const auto second = pool.allocate(/*num_blocks=*/1, BlockType::KV);
  ASSERT_EQ(second.size(), 1);
  EXPECT_TRUE(pool.allocate(/*num_blocks=*/1, BlockType::KV).empty());
  std::vector<int32_t> owned_rows;
  pool.append_full_block_rows(first.front(), &owned_rows);
  EXPECT_EQ(owned_rows, original_rows);
  pool.mark_active(first.front(), BlockType::KV);
  pool.mark_active(first.front(), BlockType::KV);
  EXPECT_EQ(pool.num_used_slots(), 128);
  pool.release(first.front(), BlockType::KV);
  EXPECT_EQ(pool.num_used_slots(), 64);
  EXPECT_EQ(pool.num_free_slots(), 64);
  pool.release(second.front(), BlockType::KV);
  EXPECT_EQ(pool.num_used_slots(), 0);
  EXPECT_EQ(pool.num_free_slots(), pool.num_total_slots());
}

TEST(StaticKVCacheBlockPoolTest,
     CountsAllocatableBlocksWithTypedFragmentation) {
  const StaticKVCacheLayout layout(make_spec());
  StaticKVCacheBlockPool pool(/*logical_capacity=*/64, layout);
  EXPECT_EQ(pool.num_free_blocks(BlockType::KV), 8);
  EXPECT_EQ(pool.num_free_blocks(BlockType::LINEAR), 3);
  const auto full = pool.allocate(/*num_blocks=*/2, BlockType::KV);
  ASSERT_EQ(full.size(), 2);
  EXPECT_EQ(pool.num_free_blocks(BlockType::LINEAR), 2);
  const auto linear = pool.allocate(/*num_blocks=*/2, BlockType::LINEAR);
  ASSERT_EQ(linear.size(), 2);
  EXPECT_EQ(pool.num_free_blocks(BlockType::LINEAR), 0);
  EXPECT_EQ(pool.num_free_blocks(BlockType::KV), 1);
  pool.release(linear.front(), BlockType::LINEAR);
  EXPECT_EQ(pool.num_free_blocks(BlockType::LINEAR), 1);
}

TEST(StaticKVCacheBlockPoolTest, PreservesScratchAndNegativeSentinelMappings) {
  const StaticKVCacheLayout layout(make_spec());
  StaticKVCacheBlockPool pool(/*logical_capacity=*/64, layout);
  std::vector<int32_t> full_rows;
  pool.append_full_block_rows(/*block_id=*/0, &full_rows);
  pool.append_full_block_rows(/*block_id=*/-1, &full_rows);
  EXPECT_EQ(full_rows, (std::vector<int32_t>{0, 0, -1, -1}));
  std::vector<int32_t> linear_rows;
  pool.append_linear_state_rows(/*state_id=*/0, &linear_rows);
  pool.append_linear_state_slots(/*state_id=*/-2, &linear_rows);
  EXPECT_EQ(linear_rows, (std::vector<int32_t>{0, 0, 0, 0, -2, -2, -2, -2}));
}

TEST(StaticKVCacheBlockPoolTest, CapacityQueriesMatchAllocationAfterRetyping) {
  const StaticKVCacheLayout layout(
      make_spec(/*num_slabs=*/13, /*checkpoint_stride=*/4));
  StaticKVCacheBlockPool pool(/*logical_capacity=*/64, layout);
  std::vector<int32_t> full = pool.allocate(/*num_blocks=*/3, BlockType::KV);
  std::vector<int32_t> linear =
      pool.allocate(/*num_blocks=*/2, BlockType::LINEAR);
  ASSERT_EQ(full.size(), 3);
  ASSERT_EQ(linear.size(), 2);
  const auto check_capacity = [&pool]() {
    for (const BlockType type : {BlockType::KV, BlockType::LINEAR}) {
      const size_t capacity = pool.num_free_blocks(type);
      const auto probe = pool.allocate(capacity, type);
      ASSERT_EQ(probe.size(), capacity);
      EXPECT_TRUE(pool.allocate(/*num_blocks=*/1, type).empty());
      for (const int32_t id : probe) {
        pool.release(id, type);
      }
    }
  };
  check_capacity();
  pool.release(full.back(), BlockType::KV);
  full.pop_back();
  pool.release(linear.back(), BlockType::LINEAR);
  linear.pop_back();
  check_capacity();
  for (const int32_t id : full) {
    pool.release(id, BlockType::KV);
  }
  for (const int32_t id : linear) {
    pool.release(id, BlockType::LINEAR);
  }
  check_capacity();
  EXPECT_EQ(pool.num_free_slots(), pool.num_total_slots());
  EXPECT_EQ(pool.num_used_slots(), 0);
}

TEST(StaticKVCacheBlockPoolTest,
     AccountsPayloadAboveFourGiBWithoutArenaMemory) {
  auto spec = make_spec(/*num_slabs=*/17);
  spec.full_row_bytes = size_t{1} << 27;
  spec.conv_row_bytes = size_t{1} << 26;
  spec.ssm_row_bytes = size_t{1} << 28;
  const StaticKVCacheLayout layout(spec);
  StaticKVCacheBlockPool pool(/*logical_capacity=*/64, layout);
  EXPECT_EQ(pool.num_total_slots(), size_t{8} << 30);
  const auto full = pool.allocate(/*num_blocks=*/9, BlockType::KV);
  ASSERT_EQ(full.size(), 9);
  EXPECT_EQ(pool.num_used_slots(), size_t{9} << 29);
  EXPECT_EQ(pool.num_free_slots(), size_t{7} << 29);
  for (const int32_t id : full) {
    pool.release(id, BlockType::KV);
  }
  EXPECT_EQ(pool.num_free_slots(), size_t{8} << 30);
  EXPECT_EQ(pool.num_used_slots(), 0);
}

}  // namespace
}  // namespace xllm

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

class StaticKVCacheBlockPoolTestPeer final {
 public:
  static size_t num_request_placements(const StaticKVCacheBlockPool& pool) {
    std::lock_guard<std::mutex> lock(pool.mutex_);
    return pool.request_placements_.size();
  }
};

namespace {

StaticKVCacheLayoutSpec make_spec(size_t num_slabs = 10,
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

size_t full_block_slab(const StaticKVCacheBlockPool& pool,
                       const StaticKVCacheLayout& layout,
                       int32_t block_id) {
  std::vector<int32_t> rows;
  pool.append_full_block_rows(block_id, &rows);
  return rows.front() * layout.full_row_bytes() / layout.slab_bytes();
}

TEST(StaticKVCacheBlockPoolTest,
     FullLeaseEpochChangesWhenTheSamePhysicalRowIsReallocated) {
  StaticKVCacheLayoutSpec spec = make_spec(/*num_slabs=*/10,
                                           /*checkpoint_stride=*/4);
  spec.full_layer_ids = {0};
  const StaticKVCacheLayout layout(spec);
  StaticKVCacheBlockPool pool(/*logical_capacity=*/64, layout);
  const auto full = pool.allocate(/*num_blocks=*/2, BlockType::KV);
  ASSERT_EQ(full.size(), 2u);
  ASSERT_EQ(full_block_slab(pool, layout, full[0]),
            full_block_slab(pool, layout, full[1]));
  std::vector<int32_t> original_rows;
  std::vector<uint64_t> epochs;
  pool.append_full_block_rows(full[0], &original_rows, &epochs);
  EXPECT_EQ(epochs, (std::vector<uint64_t>{1}));

  pool.mark_inactive(full[0], BlockType::KV);
  pool.mark_active(full[0], BlockType::KV);
  std::vector<int32_t> retained_rows;
  pool.append_full_block_rows(full[0], &retained_rows, &epochs);
  EXPECT_EQ(retained_rows, original_rows);
  EXPECT_EQ(epochs, (std::vector<uint64_t>{1, 1}));

  // Keep the other Full object live so the slab cannot change generation.
  pool.release(full[0], BlockType::KV);
  const auto reused = pool.allocate(/*num_blocks=*/1, BlockType::KV);
  ASSERT_EQ(reused.size(), 1u);
  ASSERT_EQ(reused.front(), full[0]);
  std::vector<int32_t> reused_rows;
  pool.append_full_block_rows(reused.front(), &reused_rows, &epochs);
  EXPECT_EQ(reused_rows, original_rows);
  EXPECT_EQ(epochs, (std::vector<uint64_t>{1, 1, 2}));
  std::vector<int32_t> other_rows;
  pool.append_full_block_rows(full[1], &other_rows, &epochs);
  EXPECT_EQ(epochs.back(), 1u);
}

TEST(StaticKVCacheBlockPoolTest,
     FullLeaseEpochSurvivesCrossTypeReuseAndFailedAllocation) {
  StaticKVCacheLayoutSpec spec = make_spec();
  spec.full_layer_ids = {0};
  const StaticKVCacheLayout layout(spec);
  StaticKVCacheBlockPool pool(/*logical_capacity=*/64, layout);
  const size_t capacity = pool.num_free_blocks(BlockType::KV);
  EXPECT_TRUE(pool.allocate(capacity + 1, BlockType::KV).empty());
  const auto full = pool.allocate(/*num_blocks=*/1, BlockType::KV);
  ASSERT_EQ(full.size(), 1u);
  std::vector<int32_t> rows;
  std::vector<uint64_t> epochs;
  pool.append_full_block_rows(full.front(), &rows, &epochs);
  EXPECT_EQ(epochs, (std::vector<uint64_t>{1}));
  const size_t old_slab = full_block_slab(pool, layout, full.front());
  pool.release(full.front(), BlockType::KV);

  const auto linear = pool.allocate(/*num_blocks=*/1, BlockType::LINEAR);
  ASSERT_EQ(linear.size(), 1u);
  std::vector<int32_t> linear_rows;
  pool.append_linear_state_rows(linear.front(), &linear_rows);
  EXPECT_EQ(linear_rows.front() * layout.conv_row_bytes() / layout.slab_bytes(),
            old_slab);
  pool.release(linear.front(), BlockType::LINEAR);
  const auto reused = pool.allocate(/*num_blocks=*/1, BlockType::KV);
  ASSERT_EQ(reused.size(), 1u);
  ASSERT_EQ(reused.front(), full.front());
  rows.clear();
  pool.append_full_block_rows(reused.front(), &rows, &epochs);
  EXPECT_EQ(epochs, (std::vector<uint64_t>{1, 2}));
  pool.append_full_block_rows(/*block_id=*/0, &rows, &epochs);
  pool.append_full_block_rows(/*block_id=*/-1, &rows, &epochs);
  EXPECT_EQ(epochs, (std::vector<uint64_t>{1, 2, 0, 0}));
}

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
  EXPECT_GE(extents.front().begin,
            layout.num_reserved_slabs() * layout.slab_bytes());
  for (size_t extent = 1; extent < extents.size(); ++extent) {
    EXPECT_LE(extents[extent - 1].end, extents[extent].begin);
  }
  EXPECT_EQ(pool.slots_per_block(BlockType::KV), 64);
  EXPECT_EQ(pool.slots_per_block(BlockType::LINEAR), 144);
  EXPECT_EQ(pool.num_used_slots(), 3 * 64 + 144);
  EXPECT_EQ(pool.num_free_slots(), 8 * 64 - (3 * 64 + 144));
}

TEST(StaticKVCacheBlockPoolTest, ReusesCompletelyFreedSlabsForAnotherType) {
  const StaticKVCacheLayout layout(make_spec(/*num_slabs=*/5));
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
  const StaticKVCacheLayout layout(make_spec(/*num_slabs=*/4));
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
      make_spec(/*num_slabs=*/3, /*checkpoint_stride=*/4));
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
      make_spec(/*num_slabs=*/3, /*checkpoint_stride=*/4));
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
  EXPECT_GT(layout.full_read_padding_row(), 0);
  std::vector<int32_t> linear_rows;
  pool.append_linear_state_rows(/*state_id=*/0, &linear_rows);
  pool.append_linear_state_slots(/*state_id=*/-2, &linear_rows);
  EXPECT_EQ(linear_rows, (std::vector<int32_t>{0, 0, 0, 0, -2, -2, -2, -2}));
}

TEST(StaticKVCacheBlockPoolTest, ExcludesReadPaddingWhenSlabsChangeType) {
  const StaticKVCacheLayout layout(make_spec());
  StaticKVCacheBlockPool pool(/*logical_capacity=*/64, layout);
  const size_t allocatable_begin =
      layout.num_reserved_slabs() * layout.slab_bytes();
  const size_t padding_end =
      layout.full_read_padding_row() * layout.full_row_bytes() +
      layout.full_pair_bytes();
  ASSERT_LE(padding_end, allocatable_begin);
  const auto full =
      pool.allocate(pool.num_free_blocks(BlockType::KV), BlockType::KV);
  ASSERT_FALSE(full.empty());
  for (const int32_t id : full) {
    std::vector<int32_t> rows;
    pool.append_full_block_rows(id, &rows);
    for (const int32_t row : rows) {
      EXPECT_GE(row * layout.full_row_bytes(), allocatable_begin);
    }
    pool.release(id, BlockType::KV);
  }
  const auto linear =
      pool.allocate(pool.num_free_blocks(BlockType::LINEAR), BlockType::LINEAR);
  ASSERT_FALSE(linear.empty());
  for (const int32_t id : linear) {
    std::vector<int32_t> rows;
    pool.append_linear_state_rows(id, &rows);
    for (size_t layer = 0; layer < layout.linear_layer_count(); ++layer) {
      EXPECT_GE(rows[2 * layer] * layout.conv_row_bytes(), allocatable_begin);
      EXPECT_GE(rows[2 * layer + 1] * layout.ssm_row_bytes(),
                allocatable_begin);
    }
    pool.release(id, BlockType::LINEAR);
  }
  EXPECT_EQ(pool.num_free_slots(), pool.num_total_slots());
}

TEST(StaticKVCacheBlockPoolTest, AccountsForReadPaddingBeyondScratchSlab) {
  const StaticKVCacheLayout layout(
      make_spec(/*num_slabs=*/4, /*checkpoint_stride=*/2));
  StaticKVCacheBlockPool pool(/*logical_capacity=*/64, layout);
  EXPECT_EQ(layout.num_reserved_slabs(), 2);
  EXPECT_EQ(layout.full_read_padding_row(), 4);
  EXPECT_EQ(pool.num_total_slots(), 2 * layout.slab_bytes());
  EXPECT_EQ(pool.num_free_blocks(BlockType::KV), 2);
  EXPECT_TRUE(layout.can_fit(/*full_blocks=*/2, /*linear_states=*/0));
  EXPECT_FALSE(layout.can_fit(/*full_blocks=*/3, /*linear_states=*/0));
  EXPECT_FALSE(layout.can_fit(/*full_blocks=*/0, /*linear_states=*/1));
  EXPECT_EQ(pool.allocate(/*num_blocks=*/2, BlockType::KV).size(), 2);
  EXPECT_TRUE(pool.allocate(/*num_blocks=*/1, BlockType::KV).empty());
}

TEST(StaticKVCacheBlockPoolTest, CapacityQueriesMatchAllocationAfterRetyping) {
  const StaticKVCacheLayout layout(
      make_spec(/*num_slabs=*/14, /*checkpoint_stride=*/4));
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
     RejectsHybridFootprintsNeedingTooManyTypedSlabs) {
  StaticKVCacheLayoutSpec spec;
  spec.num_slots = 32;
  spec.num_slabs = 4;
  spec.full_row_bytes = 16;
  spec.conv_row_bytes = 6;
  spec.ssm_row_bytes = 4;
  spec.full_layer_ids = {1, 3};
  spec.linear_layer_ids = {0, 2};
  const StaticKVCacheLayout layout(spec);
  StaticKVCacheBlockPool pool(/*logical_capacity=*/32, layout);
  ASSERT_EQ(layout.slab_bytes(), 96);
  EXPECT_TRUE(layout.can_fit(/*full_blocks=*/1, /*linear_states=*/1));
  EXPECT_FALSE(layout.can_fit(/*full_blocks=*/3, /*linear_states=*/1));
  EXPECT_FALSE(layout.can_fit(/*full_blocks=*/2, /*linear_states=*/2));
  const auto full = pool.allocate(/*num_blocks=*/3, BlockType::KV);
  ASSERT_EQ(full.size(), 3);
  EXPECT_LT(pool.slots_per_block(BlockType::KV) * full.size() +
                pool.slots_per_block(BlockType::LINEAR),
            pool.num_total_slots());
  EXPECT_TRUE(pool.allocate(/*num_blocks=*/1, BlockType::LINEAR).empty());
  for (const int32_t id : full) {
    pool.release(id, BlockType::KV);
  }
  const auto one_full = pool.allocate(/*num_blocks=*/1, BlockType::KV);
  const auto one_linear = pool.allocate(/*num_blocks=*/1, BlockType::LINEAR);
  ASSERT_EQ(one_full.size(), 1);
  ASSERT_EQ(one_linear.size(), 1);
  pool.release(one_full.front(), BlockType::KV);
  pool.release(one_linear.front(), BlockType::LINEAR);
  EXPECT_EQ(pool.num_free_slots(), pool.num_total_slots());
}

TEST(StaticKVCacheBlockPoolTest,
     AccountsPayloadAboveFourGiBWithoutArenaMemory) {
  auto spec = make_spec(/*num_slabs=*/18);
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

TEST(StaticKVCacheBlockPoolTest, ResumedRequestsPreferTheirOwnNonfullSlabs) {
  const StaticKVCacheLayout layout(
      make_spec(/*num_slabs=*/10, /*checkpoint_stride=*/8));
  StaticKVCacheBlockPool pool(/*logical_capacity=*/64, layout);
  const int32_t request_a = 1;
  const int32_t request_b = 2;
  auto first = pool.allocate(2, BlockType::KV, &request_a);
  auto second = pool.allocate(4, BlockType::KV, &request_b);
  ASSERT_EQ(first.size(), 2);
  ASSERT_EQ(second.size(), 4);
  const size_t first_slab = full_block_slab(pool, layout, first.back());
  const size_t second_slab = full_block_slab(pool, layout, second.back());
  ASSERT_NE(first_slab, second_slab);
  pool.release(first.front(), BlockType::KV);
  first.erase(first.begin());

  // The newly opened global hole belongs to A, but B resumes its own slab.
  const auto second_resume = pool.allocate(1, BlockType::KV, &request_b);
  ASSERT_EQ(second_resume.size(), 1);
  EXPECT_EQ(full_block_slab(pool, layout, second_resume.front()), second_slab);
  const auto first_resume = pool.allocate(1, BlockType::KV, &request_a);
  ASSERT_EQ(first_resume.size(), 1);
  EXPECT_EQ(full_block_slab(pool, layout, first_resume.front()), first_slab);
  EXPECT_EQ(StaticKVCacheBlockPoolTestPeer::num_request_placements(pool), 2);
  pool.release(first.front(), BlockType::KV);
  EXPECT_EQ(StaticKVCacheBlockPoolTestPeer::num_request_placements(pool), 2);
  pool.release(first_resume.front(), BlockType::KV);
  EXPECT_EQ(StaticKVCacheBlockPoolTestPeer::num_request_placements(pool), 1);
  for (const int32_t id : second) {
    pool.release(id, BlockType::KV);
  }
  pool.release(second_resume.front(), BlockType::KV);
  EXPECT_EQ(StaticKVCacheBlockPoolTestPeer::num_request_placements(pool), 0);
  EXPECT_EQ(pool.num_free_slots(), pool.num_total_slots());
}

TEST(StaticKVCacheBlockPoolTest,
     RetypingInvalidatesPriorRequestSlabGeneration) {
  const StaticKVCacheLayout layout(
      make_spec(/*num_slabs=*/7, /*checkpoint_stride=*/8));
  StaticKVCacheBlockPool pool(/*logical_capacity=*/64, layout);
  const int32_t request_a = 1;
  const int32_t request_b = 2;
  const auto first = pool.allocate(2, BlockType::KV, &request_a);
  const auto state = pool.allocate(1, BlockType::LINEAR, &request_a);
  ASSERT_EQ(first.size(), 2);
  ASSERT_EQ(state.size(), 1);
  const size_t old_slab = full_block_slab(pool, layout, first.front());
  for (const int32_t id : first) {
    pool.release(id, BlockType::KV);
  }
  // A's live state keeps its request hint while its old KV slab changes
  // from KV to SSM and back to KV under another request.
  const auto other_state = pool.allocate(1, BlockType::LINEAR, &request_b);
  ASSERT_EQ(other_state.size(), 1);
  pool.release(other_state.front(), BlockType::LINEAR);
  const auto other_full = pool.allocate(8, BlockType::KV, &request_b);
  ASSERT_EQ(other_full.size(), 8);
  ASSERT_EQ(full_block_slab(pool, layout, other_full[4]), old_slab);
  const size_t global_slab = full_block_slab(pool, layout, other_full[0]);
  ASSERT_NE(global_slab, old_slab);
  pool.release(other_full[4], BlockType::KV);
  pool.release(other_full[0], BlockType::KV);
  const auto resumed = pool.allocate(1, BlockType::KV, &request_a);
  ASSERT_EQ(resumed.size(), 1);
  EXPECT_EQ(full_block_slab(pool, layout, resumed.front()), global_slab);
  for (size_t index = 0; index < other_full.size(); ++index) {
    if (index != 0 && index != 4) {
      pool.release(other_full[index], BlockType::KV);
    }
  }
  pool.release(resumed.front(), BlockType::KV);
  pool.release(state.front(), BlockType::LINEAR);
  EXPECT_EQ(StaticKVCacheBlockPoolTestPeer::num_request_placements(pool), 0);
  EXPECT_EQ(pool.num_free_slots(), pool.num_total_slots());
}

TEST(StaticKVCacheBlockPoolTest,
     FailedRequestAllocationRollsBackHintOwnership) {
  const StaticKVCacheLayout layout(
      make_spec(/*num_slabs=*/5, /*checkpoint_stride=*/8));
  StaticKVCacheBlockPool pool(/*logical_capacity=*/64, layout);
  const int32_t request_a = 1;
  const int32_t request_b = 2;
  const auto state = pool.allocate(1, BlockType::LINEAR, &request_a);
  ASSERT_EQ(state.size(), 1);
  std::vector<int32_t> before;
  pool.append_linear_state_rows(state.front(), &before);
  pool.mark_inactive(state.front(), BlockType::LINEAR);
  EXPECT_EQ(pool.num_used_slots(), 0);
  EXPECT_EQ(StaticKVCacheBlockPoolTestPeer::num_request_placements(pool), 1);
  const size_t free_bytes = pool.num_free_slots();
  EXPECT_TRUE(pool.allocate(1, BlockType::LINEAR, &request_b).empty());
  EXPECT_EQ(pool.num_free_slots(), free_bytes);
  EXPECT_EQ(StaticKVCacheBlockPoolTestPeer::num_request_placements(pool), 1);
  std::vector<int32_t> after;
  pool.append_linear_state_rows(state.front(), &after);
  EXPECT_EQ(after, before);
  pool.release(state.front(), BlockType::LINEAR);
  EXPECT_EQ(StaticKVCacheBlockPoolTestPeer::num_request_placements(pool), 0);
  const auto full = pool.allocate(12, BlockType::KV, &request_b);
  ASSERT_EQ(full.size(), 12);
  for (const int32_t id : full) {
    pool.release(id, BlockType::KV);
  }
  EXPECT_EQ(StaticKVCacheBlockPoolTestPeer::num_request_placements(pool), 0);
  EXPECT_EQ(pool.num_free_slots(), pool.num_total_slots());
}

TEST(StaticKVCacheBlockPoolTest, ScoresOnlyCompletelyCoveredInactiveSlabs) {
  const StaticKVCacheLayout layout(
      make_spec(/*num_slabs=*/10, /*checkpoint_stride=*/8));
  StaticKVCacheBlockPool pool(/*logical_capacity=*/64, layout);
  const auto full = pool.allocate(4, BlockType::KV);
  const auto linear = pool.allocate(2, BlockType::LINEAR);
  ASSERT_EQ(full.size(), 4);
  ASSERT_EQ(linear.size(), 2);
  for (const int32_t id : full) {
    pool.mark_inactive(id, BlockType::KV);
  }
  EXPECT_EQ(pool.reclaimable_slab_count(BlockType::KV, {full[0]}), 0);
  EXPECT_EQ(pool.reclaimable_slab_count(BlockType::KV, full), 1);
  pool.mark_active(full[0], BlockType::KV);
  EXPECT_EQ(pool.reclaimable_slab_count(BlockType::KV, full), 0);
  pool.mark_inactive(full[0], BlockType::KV);
  pool.mark_inactive(linear[0], BlockType::LINEAR);
  EXPECT_EQ(pool.reclaimable_slab_count(BlockType::LINEAR, {linear[0]}), 2);
  pool.mark_inactive(linear[1], BlockType::LINEAR);
  const size_t free_bytes = pool.num_free_slots();
  EXPECT_EQ(pool.reclaimable_slab_count(BlockType::LINEAR, linear), 5);
  EXPECT_EQ(pool.reclaimable_slab_count(BlockType::KV, {-1, 0, 63}), 0);
  EXPECT_EQ(pool.num_free_slots(), free_bytes);
  for (const int32_t id : full) {
    pool.release(id, BlockType::KV);
  }
  pool.release(linear[0], BlockType::LINEAR);
  EXPECT_EQ(pool.reclaimable_slab_count(BlockType::LINEAR, linear), 3);
  pool.release(linear[1], BlockType::LINEAR);
  EXPECT_EQ(pool.num_free_slots(), pool.num_total_slots());
  StaticKVCacheBlockPool legacy(/*num_slots=*/8, /*linear_state_slots=*/1);
  EXPECT_EQ(legacy.reclaimable_slab_count(BlockType::KV, {}), 0);
}

}  // namespace
}  // namespace xllm

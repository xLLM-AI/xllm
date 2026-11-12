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

#include "core/kv_cache/storage/static_kv_cache_tensor_allocator.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "core/kv_cache/block/static_kv_cache_block_pool.h"

namespace xllm {
namespace {

StaticKVCacheLayoutSpec make_spec(size_t num_slots = 64,
                                  size_t checkpoint_stride = 2,
                                  size_t num_slabs = 12) {
  StaticKVCacheLayoutSpec spec;
  spec.num_slots = num_slots;
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

struct CacheViews {
  std::vector<torch::Tensor> keys;
  std::vector<torch::Tensor> values;
  std::vector<torch::Tensor> conv;
  std::vector<torch::Tensor> ssm;
};

CacheViews allocate_views(StaticKVCacheTensorAllocator& allocator,
                          const StaticKVCacheLayoutSpec& spec) {
  CacheViews views;
  views.keys.reserve(spec.full_layer_ids.size());
  views.values.reserve(spec.full_layer_ids.size());
  views.conv.reserve(spec.linear_layer_ids.size());
  views.ssm.reserve(spec.linear_layer_ids.size());
  const int64_t slots = static_cast<int64_t>(spec.num_slots);
  const int64_t row_elements = static_cast<int64_t>(
      spec.full_row_bytes / torch::elementSize(spec.dtype));
  const int64_t conv_elements = static_cast<int64_t>(
      spec.conv_row_bytes / torch::elementSize(spec.dtype));
  const int64_t ssm_elements = static_cast<int64_t>(
      spec.ssm_row_bytes / torch::elementSize(spec.ssm_dtype));
  for (const int64_t layer : spec.full_layer_ids) {
    allocator.begin_layer(layer);
    views.keys.emplace_back(allocator.allocate(KVCacheTensorRole::KEY,
                                               {slots, row_elements},
                                               spec.dtype,
                                               torch::kCPU));
    views.values.emplace_back(allocator.allocate(KVCacheTensorRole::VALUE,
                                                 {slots, row_elements},
                                                 spec.dtype,
                                                 torch::kCPU));
  }
  for (const int64_t layer : spec.linear_layer_ids) {
    allocator.begin_layer(layer);
    views.conv.emplace_back(allocator.allocate(KVCacheTensorRole::CONV,
                                               {slots, conv_elements},
                                               spec.dtype,
                                               torch::kCPU));
    views.ssm.emplace_back(allocator.allocate(
        KVCacheTensorRole::SSM,
        {slots * static_cast<int64_t>(spec.checkpoint_stride), ssm_elements},
        spec.ssm_dtype,
        torch::kCPU));
  }
  return views;
}

struct FullPublication {
  std::vector<int32_t> rows;
  std::vector<uint64_t> epochs;
};

FullPublication full_publication(const StaticKVCacheBlockPool& pool,
                                 const StaticKVCacheLayout& layout,
                                 const std::vector<int32_t>& ids) {
  FullPublication publication;
  publication.rows.reserve(ids.size() * (layout.full_layer_count() + 1));
  publication.epochs.reserve(ids.size());
  for (const int32_t id : ids) {
    publication.rows.emplace_back(id);
    pool.append_full_block_rows(id, &publication.rows, &publication.epochs);
  }
  return publication;
}

TEST(StaticKVCacheTensorAllocatorTest, PlansGeometryWithoutAllocatingCapacity) {
  const StaticKVCacheLayout layout(make_spec(
      /*num_slots=*/0, /*checkpoint_stride=*/2, /*num_slabs=*/0));
  EXPECT_EQ(layout.bank_count(), 2);
  EXPECT_EQ(layout.full_layers_per_bank(), 1);
  EXPECT_EQ(layout.linear_layers_per_bank(), 1);
  EXPECT_EQ(layout.slice_bytes(), 64);
  EXPECT_EQ(layout.slab_bytes(), 128);
  EXPECT_EQ(layout.bank_bytes(), 0);
  EXPECT_EQ(layout.num_reserved_slabs(), 2);
  EXPECT_EQ(layout.full_read_padding_page(), 4);
  EXPECT_EQ(layout.scratch_full_page(), 0);
  EXPECT_EQ(layout.full_pages_per_slab(), 4);
  EXPECT_EQ(layout.objects_per_slab(KVCacheTensorRole::KEY), 2);
  EXPECT_EQ(layout.objects_per_slab(KVCacheTensorRole::CONV), 8);
  EXPECT_EQ(layout.objects_per_slab(KVCacheTensorRole::SSM), 1);
  EXPECT_EQ(layout.layer_bank_index(0), 0);
  EXPECT_EQ(layout.layer_bank_index(3), 1);
  EXPECT_EQ(layout.layer_bank_index(1), 0);
  EXPECT_EQ(layout.layer_bank_index(2), 1);
  EXPECT_EQ(layout.linear_group(2, KVCacheTensorRole::SSM), 3);
  EXPECT_EQ(layout.bytes_used(), 0);
}

TEST(StaticKVCacheTensorAllocatorTest,
     LogicalCapacityDoesNotInflatePhysicalTensorViews) {
  const StaticKVCacheLayoutSpec small_spec = make_spec(/*num_slots=*/8);
  StaticKVCacheLayoutSpec large_spec = small_spec;
  large_spec.num_slots = 8192;
  const StaticKVCacheLayout small_layout(small_spec);
  const StaticKVCacheLayout large_layout(large_spec);
  StaticKVCacheTensorAllocator small_allocator(small_layout, torch::kCPU);
  StaticKVCacheTensorAllocator large_allocator(large_layout, torch::kCPU);
  const CacheViews small = allocate_views(small_allocator, small_spec);
  const CacheViews large = allocate_views(large_allocator, large_spec);

  EXPECT_EQ(small_layout.bytes_used(), large_layout.bytes_used());
  EXPECT_EQ(small_layout.bank_bytes(), large_layout.bank_bytes());
  EXPECT_EQ(small.keys.front().sizes(), large.keys.front().sizes());
  EXPECT_EQ(small.values.front().sizes(), large.values.front().sizes());
  EXPECT_EQ(small.conv.front().sizes(), large.conv.front().sizes());
  EXPECT_EQ(small.ssm.front().sizes(), large.ssm.front().sizes());
  EXPECT_GT(small.keys.front().size(0), small_spec.num_slots);
  EXPECT_LT(large.keys.front().size(0), large_spec.num_slots);
}

TEST(StaticKVCacheTensorAllocatorTest, UsesGcdBanksForGeneralLayerRatios) {
  struct Geometry {
    size_t full_layers;
    size_t linear_layers;
    size_t banks;
    size_t full_per_bank;
    size_t linear_per_bank;
    size_t slice_bytes;
  };
  for (const Geometry geometry : {Geometry{2, 2, 2, 1, 1, 64},
                                  Geometry{3, 2, 1, 3, 2, 192},
                                  Geometry{6, 4, 2, 3, 2, 192},
                                  Geometry{5, 7, 1, 5, 7, 320}}) {
    SCOPED_TRACE(geometry.full_layers);
    SCOPED_TRACE(geometry.linear_layers);
    StaticKVCacheLayoutSpec spec = make_spec();
    spec.full_layer_ids.clear();
    spec.linear_layer_ids.clear();
    spec.full_layer_ids.reserve(geometry.full_layers);
    spec.linear_layer_ids.reserve(geometry.linear_layers);
    for (size_t layer = 0; layer < geometry.full_layers; ++layer) {
      spec.full_layer_ids.emplace_back(static_cast<int64_t>(layer));
    }
    for (size_t layer = 0; layer < geometry.linear_layers; ++layer) {
      spec.linear_layer_ids.emplace_back(
          static_cast<int64_t>(geometry.full_layers + layer));
    }
    const StaticKVCacheLayout layout(spec);
    EXPECT_EQ(layout.bank_count(), geometry.banks);
    EXPECT_EQ(layout.full_layers_per_bank(), geometry.full_per_bank);
    EXPECT_EQ(layout.linear_layers_per_bank(), geometry.linear_per_bank);
    EXPECT_EQ(layout.slice_bytes(), geometry.slice_bytes);
    EXPECT_EQ(layout.slab_bytes(), geometry.banks * geometry.slice_bytes);
    EXPECT_EQ(layout.bytes_used(), spec.num_slabs * layout.slab_bytes());
    for (const int64_t layer : spec.full_layer_ids) {
      const auto& key = layout.tensor_layout(layer, KVCacheTensorRole::KEY);
      const auto& value = layout.tensor_layout(layer, KVCacheTensorRole::VALUE);
      EXPECT_EQ(value.offset_bytes - key.offset_bytes, spec.full_row_bytes);
      EXPECT_LE(value.offset_bytes + value.view_rows * value.row_bytes,
                (key.bank + 1) * layout.bank_bytes());
    }
  }
}

TEST(StaticKVCacheTensorAllocatorTest, BoundsRealQwen38FinalFiaPagesPerBank) {
  StaticKVCacheLayoutSpec spec = make_spec(
      /*num_slots=*/65536, /*checkpoint_stride=*/5, /*num_slabs=*/0);
  spec.full_row_bytes = 128 * 2 * 256 * 2;
  spec.conv_row_bytes = 7 * (2 * 8 * 128 + 24 * 128) * 2;
  spec.ssm_row_bytes = 24 * 128 * 128 * 4;
  spec.dtype = torch::kBFloat16;
  spec.full_layer_ids.clear();
  spec.linear_layer_ids.clear();
  spec.full_layer_ids.reserve(16);
  spec.linear_layer_ids.reserve(48);
  for (int64_t layer = 0; layer < 16; ++layer) {
    spec.full_layer_ids.emplace_back(layer);
  }
  for (int64_t layer = 16; layer < 64; ++layer) {
    spec.linear_layer_ids.emplace_back(layer);
  }
  const StaticKVCacheLayout planned(spec);
  EXPECT_EQ(planned.bank_count(), 16);
  EXPECT_EQ(planned.full_layers_per_bank(), 1);
  EXPECT_EQ(planned.linear_layers_per_bank(), 3);
  EXPECT_EQ(planned.slice_bytes(), 7864320);
  EXPECT_EQ(planned.slab_bytes(), 125829120);
  EXPECT_EQ(planned.full_pages_per_slab(), 60);
  EXPECT_EQ(planned.objects_per_slab(KVCacheTensorRole::KEY), 30);
  EXPECT_EQ(planned.objects_per_slab(KVCacheTensorRole::CONV), 108);
  EXPECT_EQ(planned.num_reserved_slabs(), 2);
  ASSERT_EQ(planned.max_slab_count(), 1092);
  spec.num_slabs = planned.max_slab_count();
  const StaticKVCacheLayout layout(spec);
  EXPECT_GT(layout.bytes_used(), uint64_t{1} << 32);
  StaticKVCacheBlockPool pool(/*logical_capacity=*/65536, layout);
  const auto full =
      pool.allocate(pool.num_free_blocks(BlockType::KV), BlockType::KV);
  ASSERT_FALSE(full.empty());
  constexpr uint64_t kFiaElementLimit = uint64_t{1} << 32;
  constexpr uint64_t kRowElements = 128 * 2 * 256;
  for (const int32_t id : full) {
    const int32_t page = pool.final_full_page(id);
    EXPECT_GE(page, 120);
    EXPECT_LT((static_cast<uint64_t>(page) + 1) * kRowElements - 1,
              kFiaElementLimit);
    std::vector<int32_t> rows;
    pool.append_full_block_rows(id, &rows);
    EXPECT_EQ(rows, std::vector<int32_t>(16, page));
  }
  for (const int64_t layer : spec.full_layer_ids) {
    const auto& key = layout.tensor_layout(layer, KVCacheTensorRole::KEY);
    EXPECT_LE(key.view_rows * kRowElements, kFiaElementLimit);
  }
  spec.num_slabs += 1;
  EXPECT_DEATH((void)StaticKVCacheLayout(spec),
               "paged attention address range");
}

TEST(StaticKVCacheTensorAllocatorTest,
     SameFinalPageKeepsAllLayerPayloadsSeparate) {
  StaticKVCacheLayoutSpec spec = make_spec();
  spec.full_layer_ids = {0, 1, 2, 3, 4, 5};
  spec.linear_layer_ids = {6, 7, 8, 9};
  const StaticKVCacheLayout layout(spec);
  StaticKVCacheTensorAllocator allocator(layout, torch::kCPU);
  const CacheViews views = allocate_views(allocator, spec);
  StaticKVCacheBlockPool pool(/*logical_capacity=*/64, layout);
  const auto full = pool.allocate(/*num_blocks=*/2, BlockType::KV);
  ASSERT_EQ(full.size(), 2);
  const int32_t page = pool.final_full_page(full.front());
  struct ByteExtent {
    const uint8_t* begin;
    const uint8_t* end;
  };
  std::vector<ByteExtent> extents;
  extents.reserve(2 * views.keys.size());
  for (size_t layer = 0; layer < views.keys.size(); ++layer) {
    views.keys[layer][page].fill_(static_cast<int64_t>(layer + 1));
    views.values[layer][page].fill_(static_cast<int64_t>(layer + 101));
    for (const torch::Tensor& tensor :
         {views.keys[layer], views.values[layer]}) {
      EXPECT_TRUE(tensor.is_contiguous());
      const auto* begin = static_cast<const uint8_t*>(tensor[page].data_ptr());
      extents.emplace_back(ByteExtent{begin, begin + spec.full_row_bytes});
    }
  }
  std::sort(extents.begin(),
            extents.end(),
            [](const ByteExtent& left, const ByteExtent& right) {
              return left.begin < right.begin;
            });
  for (size_t index = 1; index < extents.size(); ++index) {
    EXPECT_LE(extents[index - 1].end, extents[index].begin);
  }
  for (size_t layer = 0; layer < views.keys.size(); ++layer) {
    EXPECT_TRUE(views.keys[layer][page].eq(layer + 1).all().item<bool>());
    EXPECT_TRUE(views.values[layer][page].eq(layer + 101).all().item<bool>());
  }
  EXPECT_NE(views.keys[0].data_ptr(), views.keys[1].data_ptr());
  EXPECT_NE(views.keys[0].data_ptr(), views.keys[3].data_ptr());
}

TEST(StaticKVCacheTensorAllocatorTest, FreeGroupsReuseFullConvAndSsmBytes) {
  const StaticKVCacheLayoutSpec spec = make_spec(
      /*num_slots=*/64, /*checkpoint_stride=*/2, /*num_slabs=*/8);
  const StaticKVCacheLayout layout(spec);
  StaticKVCacheTensorAllocator allocator(layout, torch::kCPU);
  const CacheViews views = allocate_views(allocator, spec);
  StaticKVCacheBlockPool pool(/*logical_capacity=*/64, layout);
  const auto original =
      pool.allocate(pool.num_free_blocks(BlockType::KV), BlockType::KV);
  std::vector<const uint8_t*> previous_pairs;
  previous_pairs.reserve(original.size() * views.keys.size());
  for (const int32_t id : original) {
    const int32_t page = pool.final_full_page(id);
    for (const torch::Tensor& key : views.keys) {
      previous_pairs.emplace_back(
          static_cast<const uint8_t*>(key[page].data_ptr()));
    }
    pool.release(id, BlockType::KV);
  }
  const auto linear = pool.allocate(/*num_blocks=*/1, BlockType::LINEAR);
  ASSERT_EQ(linear.size(), 1);
  std::vector<int32_t> rows;
  pool.append_linear_state_rows(linear.front(), &rows);
  for (size_t layer = 0; layer < views.conv.size(); ++layer) {
    const torch::Tensor conv = views.conv[layer][rows[2 * layer]];
    const torch::Tensor ssm = views.ssm[layer].narrow(
        /*dim=*/0,
        /*start=*/rows[2 * layer + 1],
        /*length=*/static_cast<int64_t>(spec.checkpoint_stride));
    for (const torch::Tensor& tensor : {conv, ssm}) {
      const auto* address = static_cast<const uint8_t*>(tensor.data_ptr());
      EXPECT_TRUE(std::any_of(previous_pairs.begin(),
                              previous_pairs.end(),
                              [address, &layout](const uint8_t* begin) {
                                return address >= begin &&
                                       address <
                                           begin + layout.full_pair_bytes();
                              }));
    }
    conv.fill_(17);
    ssm.fill_(19);
  }
  const auto live = pool.allocate(/*num_blocks=*/1, BlockType::KV);
  ASSERT_EQ(live.size(), 1);
  const int32_t live_page = pool.final_full_page(live.front());
  for (size_t layer = 0; layer < views.keys.size(); ++layer) {
    views.keys[layer][live_page].fill_(31);
    views.values[layer][live_page].fill_(37);
  }
  pool.release(linear.front(), BlockType::LINEAR);
  const auto reused =
      pool.allocate(pool.num_free_blocks(BlockType::KV), BlockType::KV);
  const FullPublication publication = full_publication(pool, layout, reused);
  allocator.initialize_full_blocks(publication.rows, publication.epochs);
  for (const int32_t id : reused) {
    const int32_t page = pool.final_full_page(id);
    for (size_t layer = 0; layer < views.keys.size(); ++layer) {
      EXPECT_TRUE(views.keys[layer][page].eq(0).all().item<bool>());
      EXPECT_TRUE(views.values[layer][page].eq(0).all().item<bool>());
    }
  }
  for (size_t layer = 0; layer < views.keys.size(); ++layer) {
    EXPECT_TRUE(views.keys[layer][live_page].eq(31).all().item<bool>());
    EXPECT_TRUE(views.values[layer][live_page].eq(37).all().item<bool>());
  }
}

TEST(StaticKVCacheTensorAllocatorTest,
     NewEpochClearsEveryBankAndPreservesLiveGroups) {
  StaticKVCacheLayoutSpec spec = make_spec();
  spec.full_layer_ids = {0, 1, 2, 3};
  spec.linear_layer_ids = {4, 5};
  const StaticKVCacheLayout layout(spec);
  StaticKVCacheTensorAllocator allocator(layout, torch::kCPU);
  const CacheViews views = allocate_views(allocator, spec);
  StaticKVCacheBlockPool pool(/*logical_capacity=*/64, layout);
  const auto full = pool.allocate(/*num_blocks=*/2, BlockType::KV);
  const auto linear = pool.allocate(/*num_blocks=*/1, BlockType::LINEAR);
  ASSERT_EQ(full.size(), 2);
  ASSERT_EQ(linear.size(), 1);
  const int32_t renewed_page = pool.final_full_page(full[0]);
  const int32_t prefix_page = pool.final_full_page(full[1]);
  const FullPublication initial = full_publication(pool, layout, full);
  allocator.initialize_full_blocks(initial.rows, initial.epochs);
  for (size_t layer = 0; layer < views.keys.size(); ++layer) {
    views.keys[layer][renewed_page].fill_(
        std::numeric_limits<float>::quiet_NaN());
    views.values[layer][renewed_page].fill_(
        std::numeric_limits<float>::quiet_NaN());
    views.keys[layer][prefix_page].fill_(41);
    views.values[layer][prefix_page].fill_(43);
  }
  std::vector<int32_t> state_rows;
  pool.append_linear_state_rows(linear.front(), &state_rows);
  for (size_t layer = 0; layer < views.conv.size(); ++layer) {
    views.conv[layer][state_rows[2 * layer]].fill_(47);
    views.ssm[layer].narrow(0, state_rows[2 * layer + 1], 2).fill_(53);
  }
  pool.mark_inactive(full[1], BlockType::KV);
  pool.mark_active(full[1], BlockType::KV);
  pool.release(full[0], BlockType::KV);
  const auto renewed = pool.allocate(/*num_blocks=*/1, BlockType::KV);
  ASSERT_EQ(renewed.size(), 1);
  ASSERT_EQ(renewed.front(), full[0]);
  ASSERT_EQ(pool.final_full_page(renewed.front()), renewed_page);
  const FullPublication publication = full_publication(pool, layout, renewed);
  EXPECT_GT(publication.epochs.front(), initial.epochs.front());
  EXPECT_TRUE(allocator.needs_full_block_initialization(publication.rows,
                                                        publication.epochs));
  allocator.initialize_full_blocks(publication.rows, publication.epochs);
  EXPECT_FALSE(allocator.needs_full_block_initialization(publication.rows,
                                                         publication.epochs));
  for (size_t layer = 0; layer < views.keys.size(); ++layer) {
    EXPECT_TRUE(views.keys[layer][renewed_page].eq(0).all().item<bool>());
    EXPECT_TRUE(views.values[layer][renewed_page].eq(0).all().item<bool>());
    EXPECT_TRUE(views.keys[layer][prefix_page].eq(41).all().item<bool>());
    EXPECT_TRUE(views.values[layer][prefix_page].eq(43).all().item<bool>());
    views.keys[layer][renewed_page].fill_(59);
  }
  allocator.initialize_full_blocks(publication.rows, publication.epochs);
  for (const torch::Tensor& key : views.keys) {
    EXPECT_TRUE(key[renewed_page].eq(59).all().item<bool>());
  }
  for (size_t layer = 0; layer < views.conv.size(); ++layer) {
    EXPECT_TRUE(
        views.conv[layer][state_rows[2 * layer]].eq(47).all().item<bool>());
    EXPECT_TRUE(views.ssm[layer]
                    .narrow(0, state_rows[2 * layer + 1], 2)
                    .eq(53)
                    .all()
                    .item<bool>());
  }
}

TEST(StaticKVCacheTensorAllocatorTest,
     ReallocatedLogicalIdClearsItsNewPhysicalGroup) {
  const StaticKVCacheLayoutSpec spec = make_spec();
  const StaticKVCacheLayout layout(spec);
  StaticKVCacheTensorAllocator allocator(layout, torch::kCPU);
  const CacheViews views = allocate_views(allocator, spec);
  StaticKVCacheBlockPool pool(/*logical_capacity=*/64, layout);
  const auto original = pool.allocate(/*num_blocks=*/1, BlockType::KV);
  ASSERT_EQ(original.size(), 1);
  const FullPublication initial = full_publication(pool, layout, original);
  allocator.initialize_full_blocks(initial.rows, initial.epochs);
  const int32_t old_page = pool.final_full_page(original.front());
  pool.release(original.front(), BlockType::KV);
  const auto linear = pool.allocate(/*num_blocks=*/1, BlockType::LINEAR);
  ASSERT_EQ(linear.size(), 1);
  const auto renewed = pool.allocate(/*num_blocks=*/1, BlockType::KV);
  ASSERT_EQ(renewed.size(), 1);
  ASSERT_EQ(renewed.front(), original.front());
  const int32_t new_page = pool.final_full_page(renewed.front());
  ASSERT_NE(new_page, old_page);
  for (size_t layer = 0; layer < views.keys.size(); ++layer) {
    views.keys[layer][new_page].fill_(std::numeric_limits<float>::quiet_NaN());
    views.values[layer][new_page].fill_(
        std::numeric_limits<float>::quiet_NaN());
  }
  const FullPublication publication = full_publication(pool, layout, renewed);
  EXPECT_GT(publication.epochs.front(), initial.epochs.front());
  allocator.initialize_full_blocks(publication.rows, publication.epochs);
  for (size_t layer = 0; layer < views.keys.size(); ++layer) {
    EXPECT_TRUE(views.keys[layer][new_page].eq(0).all().item<bool>());
    EXPECT_TRUE(views.values[layer][new_page].eq(0).all().item<bool>());
  }
}

TEST(StaticKVCacheTensorAllocatorTest,
     DisjointFreshGroupsPreserveLiveLinearAndPrefixPayloads) {
  StaticKVCacheLayoutSpec spec = make_spec();
  spec.full_layer_ids = {0, 1, 2, 3};
  spec.linear_layer_ids = {4, 5};
  const StaticKVCacheLayout layout(spec);
  StaticKVCacheTensorAllocator allocator(layout, torch::kCPU);
  const CacheViews views = allocate_views(allocator, spec);
  StaticKVCacheBlockPool pool(/*logical_capacity=*/64, layout);
  const auto first = pool.allocate(/*num_blocks=*/1, BlockType::KV);
  const auto linear = pool.allocate(/*num_blocks=*/1, BlockType::LINEAR);
  const auto prefix = pool.allocate(/*num_blocks=*/1, BlockType::KV);
  const auto last = pool.allocate(/*num_blocks=*/1, BlockType::KV);
  ASSERT_EQ(first.size(), 1);
  ASSERT_EQ(linear.size(), 1);
  ASSERT_EQ(prefix.size(), 1);
  ASSERT_EQ(last.size(), 1);
  const FullPublication prefix_publication =
      full_publication(pool, layout, prefix);
  allocator.initialize_full_blocks(prefix_publication.rows,
                                   prefix_publication.epochs);
  const int32_t first_page = pool.final_full_page(first.front());
  const int32_t prefix_page = pool.final_full_page(prefix.front());
  const int32_t last_page = pool.final_full_page(last.front());
  for (size_t layer = 0; layer < views.keys.size(); ++layer) {
    for (const int32_t page : {first_page, last_page}) {
      views.keys[layer][page].fill_(std::numeric_limits<float>::quiet_NaN());
      views.values[layer][page].fill_(std::numeric_limits<float>::quiet_NaN());
    }
    views.keys[layer][prefix_page].fill_(61);
    views.values[layer][prefix_page].fill_(67);
  }
  std::vector<int32_t> state_rows;
  pool.append_linear_state_rows(linear.front(), &state_rows);
  for (size_t layer = 0; layer < views.conv.size(); ++layer) {
    views.conv[layer][state_rows[2 * layer]].fill_(71);
    views.ssm[layer].narrow(0, state_rows[2 * layer + 1], 2).fill_(73);
  }
  pool.mark_inactive(prefix.front(), BlockType::KV);
  const FullPublication publication = full_publication(
      pool, layout, {first.front(), prefix.front(), last.front()});
  allocator.initialize_full_blocks(publication.rows, publication.epochs);
  for (size_t layer = 0; layer < views.keys.size(); ++layer) {
    for (const int32_t page : {first_page, last_page}) {
      EXPECT_TRUE(views.keys[layer][page].eq(0).all().item<bool>());
      EXPECT_TRUE(views.values[layer][page].eq(0).all().item<bool>());
    }
    EXPECT_TRUE(views.keys[layer][prefix_page].eq(61).all().item<bool>());
    EXPECT_TRUE(views.values[layer][prefix_page].eq(67).all().item<bool>());
  }
  for (size_t layer = 0; layer < views.conv.size(); ++layer) {
    EXPECT_TRUE(
        views.conv[layer][state_rows[2 * layer]].eq(71).all().item<bool>());
    EXPECT_TRUE(views.ssm[layer]
                    .narrow(0, state_rows[2 * layer + 1], 2)
                    .eq(73)
                    .all()
                    .item<bool>());
  }
}

TEST(StaticKVCacheTensorAllocatorTest,
     ReadPaddingSurvivesAllWritableScratchRoles) {
  const StaticKVCacheLayoutSpec spec = make_spec();
  const StaticKVCacheLayout layout(spec);
  StaticKVCacheTensorAllocator allocator(layout, torch::kCPU);
  const CacheViews views = allocate_views(allocator, spec);
  for (size_t layer = 0; layer < views.keys.size(); ++layer) {
    views.keys[layer][layout.scratch_full_page()].fill_(
        std::numeric_limits<float>::quiet_NaN());
    views.values[layer][layout.scratch_full_page()].fill_(
        std::numeric_limits<float>::quiet_NaN());
  }
  for (size_t layer = 0; layer < views.conv.size(); ++layer) {
    views.conv[layer][0].fill_(std::numeric_limits<float>::quiet_NaN());
    views.ssm[layer].narrow(0, 0, 2).fill_(
        std::numeric_limits<float>::quiet_NaN());
  }
  for (size_t layer = 0; layer < views.keys.size(); ++layer) {
    EXPECT_TRUE(views.keys[layer][layout.full_read_padding_page()]
                    .eq(0)
                    .all()
                    .item<bool>());
    EXPECT_TRUE(views.values[layer][layout.full_read_padding_page()]
                    .eq(0)
                    .all()
                    .item<bool>());
  }
  EXPECT_NE(views.keys[0][layout.full_read_padding_page()].data_ptr(),
            views.keys[1][layout.full_read_padding_page()].data_ptr());
}

TEST(StaticKVCacheTensorAllocatorTest, TensorViewRetainsArenaOwnership) {
  torch::Tensor retained;
  {
    const StaticKVCacheLayoutSpec spec = make_spec();
    StaticKVCacheTensorAllocator allocator(StaticKVCacheLayout(spec),
                                           torch::kCPU);
    retained = allocate_views(allocator, spec).keys.front();
    retained[retained.size(0) - 1].fill_(43);
  }
  EXPECT_TRUE(retained[retained.size(0) - 1].eq(43).all().item<bool>());
}

TEST(StaticKVCacheTensorAllocatorTest,
     DraftLocalLayersShareArenaAndFinalPagesWithTarget) {
  StaticKVCacheLayoutSpec spec = make_spec();
  spec.draft_layer_count = 1;
  auto arena = std::make_shared<StaticKVCacheTensorAllocator>(
      StaticKVCacheLayout(spec), torch::kCPU);
  const CacheViews target = allocate_views(*arena, spec);
  const auto draft = create_static_draft_kv_tensor_allocator(arena);
  draft->begin_layer(/*layer_id=*/0);
  const torch::Tensor key = draft->allocate(
      KVCacheTensorRole::KEY, {64, 4}, torch::kFloat32, torch::kCPU);
  const torch::Tensor value = draft->allocate(
      KVCacheTensorRole::VALUE, {64, 4}, torch::kFloat32, torch::kCPU);
  EXPECT_TRUE(key.is_alias_of(target.keys.front()));
  EXPECT_TRUE(value.is_alias_of(target.values.front()));
  EXPECT_EQ(key.sizes(), target.keys.front().sizes());
  const StaticKVCacheLayout& layout = arena->layout();
  StaticKVCacheBlockPool pool(/*logical_capacity=*/64, layout);
  const auto blocks = pool.allocate(/*num_blocks=*/2, BlockType::KV);
  ASSERT_EQ(blocks.size(), 2u);
  const int32_t page = pool.final_full_page(blocks[0]);
  const int32_t prefix_page = pool.final_full_page(blocks[1]);
  const FullPublication publication = full_publication(pool, layout, blocks);
  arena->initialize_full_blocks(publication.rows, publication.epochs);
  target.keys.front()[page].fill_(17);
  key[page].fill_(19);
  value[page].fill_(23);
  key[prefix_page].fill_(29);
  value[prefix_page].fill_(31);
  EXPECT_TRUE(target.keys.front()[page].eq(17).all().item<bool>());
  EXPECT_TRUE(key[page].eq(19).all().item<bool>());
  EXPECT_TRUE(value[page].eq(23).all().item<bool>());
  EXPECT_NE(key[page].data_ptr(), target.keys.front()[page].data_ptr());
  EXPECT_NE(key[page].data_ptr(), value[page].data_ptr());

  // A repeated publication must preserve draft writes in the shared lease.
  arena->initialize_full_blocks(publication.rows, publication.epochs);
  EXPECT_TRUE(key[page].eq(19).all().item<bool>());
  pool.release(blocks[0], BlockType::KV);
  const auto renewed = pool.allocate(/*num_blocks=*/1, BlockType::KV);
  ASSERT_EQ(renewed.size(), 1u);
  ASSERT_EQ(pool.final_full_page(renewed.front()), page);
  const FullPublication next = full_publication(pool, layout, renewed);
  arena->initialize_full_blocks(next.rows, next.epochs);
  EXPECT_TRUE(target.keys.front()[page].eq(0).all().item<bool>());
  EXPECT_TRUE(key[page].eq(0).all().item<bool>());
  EXPECT_TRUE(value[page].eq(0).all().item<bool>());
  EXPECT_TRUE(key[prefix_page].eq(29).all().item<bool>());
  EXPECT_TRUE(value[prefix_page].eq(31).all().item<bool>());
  std::weak_ptr<StaticKVCacheTensorAllocator> retained_arena = arena;
  arena.reset();
  EXPECT_FALSE(retained_arena.expired());
  key[page].fill_(37);
  EXPECT_TRUE(key[page].eq(37).all().item<bool>());
}

TEST(StaticKVCacheTensorAllocatorTest,
     DraftRejectsRecurrentRolesAndBadGeometry) {
  StaticKVCacheLayoutSpec spec = make_spec();
  spec.draft_layer_count = 1;
  const auto arena = std::make_shared<StaticKVCacheTensorAllocator>(
      StaticKVCacheLayout(spec), torch::kCPU);
  const auto draft = create_static_draft_kv_tensor_allocator(arena);
  EXPECT_DEATH(draft->begin_layer(/*layer_id=*/1), "layer_id");
  draft->begin_layer(/*layer_id=*/0);
  EXPECT_DEATH(
      draft->allocate(
          KVCacheTensorRole::KEY, {64, 6}, torch::kFloat32, torch::kCPU),
      "inner row geometry");
  EXPECT_DEATH(
      draft->allocate(
          KVCacheTensorRole::CONV, {64, 2}, torch::kFloat32, torch::kCPU),
      "draft cache supports only KEY and VALUE");
}

TEST(StaticKVCacheTensorAllocatorTest, RejectsInvalidLayerAndRowGeometry) {
  StaticKVCacheTensorAllocator allocator(StaticKVCacheLayout(make_spec()),
                                         torch::kCPU);
  const auto allocate_key = [&allocator](int64_t slots, int64_t row_elements) {
    return allocator.allocate(KVCacheTensorRole::KEY,
                              {slots, row_elements},
                              torch::kFloat32,
                              torch::kCPU);
  };
  EXPECT_DEATH(allocate_key(/*slots=*/64, /*row_elements=*/4),
               "explicit layer");
  EXPECT_DEATH(allocator.begin_layer(7), "Unknown static cache layer");
  allocator.begin_layer(0);
  EXPECT_DEATH(allocate_key(/*slots=*/63, /*row_elements=*/4),
               "logical capacity");
  EXPECT_DEATH(allocate_key(/*slots=*/64, /*row_elements=*/6),
               "inner row geometry");
  EXPECT_DEATH(
      allocator.allocate(
          KVCacheTensorRole::CONV, {64, 2}, torch::kFloat32, torch::kCPU),
      "has no conv");
  allocate_key(/*slots=*/64, /*row_elements=*/4);
  EXPECT_DEATH(allocate_key(/*slots=*/64, /*row_elements=*/4),
               "role cannot be allocated twice");
  EXPECT_DEATH(allocator.begin_layer(0), "layer cannot be allocated twice");
}

}  // namespace
}  // namespace xllm

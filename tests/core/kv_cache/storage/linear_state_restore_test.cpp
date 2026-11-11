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

#include "core/kv_cache/storage/linear_state_restore.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/kv_cache/block/static_kv_cache_block_pool.h"
#include "core/kv_cache/storage/linear_state_slot_table.h"
#include "core/kv_cache/storage/static_kv_cache_tensor_allocator.h"

namespace xllm {
namespace {

TEST(LinearStateSlotTableTest, SkipsInitiallyZeroAndUnchangedMappings) {
  LinearStateSlotTable table(/*num_slots=*/4, /*num_groups=*/2, torch::kCPU);
  const int32_t* storage = table.device_slots().data_ptr<int32_t>();
  const int64_t initial_version = table.device_slots()._version();

  table.update({});
  table.update({0, 0, 0, 3, 0, 0});
  EXPECT_EQ(table.device_slots()._version(), initial_version);
  EXPECT_EQ(table.slot(3, 0), 0);
  EXPECT_EQ(table.slot(3, 1), 0);

  table.update({1, 5, 7, 2, 11, 13});
  const int64_t published_version = table.device_slots()._version();
  EXPECT_GT(published_version, initial_version);
  table.update({2, 11, 13, 1, 5, 7, 0, 0, 0});

  EXPECT_EQ(table.device_slots()._version(), published_version);
  EXPECT_EQ(table.device_slots().data_ptr<int32_t>(), storage);
  EXPECT_TRUE(torch::equal(
      table.device_slots(),
      torch::tensor({{0, 5, 11, 0}, {0, 7, 13, 0}}, torch::kInt32)));
}

TEST(LinearStateSlotTableTest, PublishesChangedRowsWithoutRelocatingStorage) {
  LinearStateSlotTable table(/*num_slots=*/5, /*num_groups=*/3, torch::kCPU);
  table.update({1, 3, 5, 7, 2, 11, 13, 17});
  const int32_t* storage = table.device_slots().data_ptr<int32_t>();

  table.update({1, 3, 5, 7, 2, 11, 19, 17, 3, 23, 29, 31});

  EXPECT_EQ(table.device_slots().data_ptr<int32_t>(), storage);
  EXPECT_EQ(table.slot(1, 1), 5);
  EXPECT_EQ(table.slot(2, 1), 19);
  EXPECT_EQ(table.slot(3, 2), 31);
  EXPECT_TRUE(torch::equal(
      table.device_slots(),
      torch::tensor({{0, 3, 11, 23, 0}, {0, 5, 19, 29, 0}, {0, 7, 17, 31, 0}},
                    torch::kInt32)));
}

TEST(LinearStateSlotTableTest, ReusedLogicalIdsPublishNewPhysicalRows) {
  LinearStateSlotTable table(/*num_slots=*/4, /*num_groups=*/2, torch::kCPU);
  table.update({1, 3, 5, 2, 7, 11});
  const int32_t* storage = table.device_slots().data_ptr<int32_t>();

  table.update({1, 0, 0});
  EXPECT_EQ(table.slot(1, 0), 0);
  EXPECT_EQ(table.device_slots()[1][1].item<int32_t>(), 0);
  table.update({1, 13, 17});

  EXPECT_EQ(table.slot(1, 0), 13);
  EXPECT_EQ(table.slot(1, 1), 17);
  EXPECT_EQ(table.device_slots().data_ptr<int32_t>(), storage);
  EXPECT_TRUE(torch::equal(
      table.device_slots(),
      torch::tensor({{0, 13, 7, 0}, {0, 17, 11, 0}}, torch::kInt32)));
}

TEST(LinearStateSlotTableTest, RepeatedIdsUseOnlyTheirFinalMapping) {
  LinearStateSlotTable table(/*num_slots=*/4, /*num_groups=*/2, torch::kCPU);
  table.update({1, 3, 5, 1, 7, 11, 2, 13, 17, 1, 19, 23});
  const int64_t published_version = table.device_slots()._version();

  table.update({1, 29, 31, 2, 37, 41, 1, 19, 23, 2, 13, 17});
  EXPECT_EQ(table.device_slots()._version(), published_version);
  EXPECT_EQ(table.slot(1, 0), 19);
  EXPECT_EQ(table.slot(2, 1), 17);
  EXPECT_TRUE(torch::equal(
      table.device_slots(),
      torch::tensor({{0, 19, 13, 0}, {0, 23, 17, 0}}, torch::kInt32)));

  table.update({2, 43, 47, 1, 53, 59, 2, 61, 67});
  EXPECT_EQ(table.slot(1, 1), 59);
  EXPECT_EQ(table.slot(2, 0), 61);
  EXPECT_TRUE(torch::equal(
      table.device_slots(),
      torch::tensor({{0, 53, 61, 0}, {0, 59, 67, 0}}, torch::kInt32)));
}

TEST(LinearStateSlotTableTest, ValidatesAllRowsBeforeFilteringChanges) {
  LinearStateSlotTable table(/*num_slots=*/4, /*num_groups=*/2, torch::kCPU);

  EXPECT_DEATH(table.update({1, 0}), "Linear state slot rows must be");
  EXPECT_DEATH(table.update({-1, 0, 0}), "state_id");
  EXPECT_DEATH(table.update({4, 0, 0}), "state_id");
  EXPECT_DEATH(table.update({1, -1, 0, 1, 0, 0}), "entry");
}

TEST(LinearStateRestoreTest, BuildsColdMaskForUncachedRow) {
  EXPECT_EQ(build_linear_state_mask(/*cached_tokens=*/{0}, /*active_rows=*/1),
            std::vector<int64_t>({0}));
}

TEST(LinearStateRestoreTest, BuildsWarmMaskForCachedRow) {
  EXPECT_EQ(build_linear_state_mask(/*cached_tokens=*/{8}, /*active_rows=*/1),
            std::vector<int64_t>({1}));
}

TEST(LinearStateRestoreTest, BuildsMixedWarmMask) {
  EXPECT_EQ(
      build_linear_state_mask(/*cached_tokens=*/{0, 8, -1}, /*active_rows=*/3),
      std::vector<int64_t>({0, 1, 0}));
}

TEST(LinearStateRestoreTest, RepeatsLogicalRowsForActiveRows) {
  EXPECT_EQ(build_linear_state_mask(/*cached_tokens=*/{0, 8},
                                    /*active_rows=*/6),
            std::vector<int64_t>({0, 0, 0, 1, 1, 1}));
}

TEST(LinearStateRestoreTest, RejectsEmptyCachedTokens) {
  EXPECT_DEATH(build_linear_state_mask(/*cached_tokens=*/{},
                                       /*active_rows=*/1),
               "cached_tokens must not be empty");
}

TEST(LinearStateRestoreTest, RejectsNonPositiveActiveRows) {
  EXPECT_DEATH(build_linear_state_mask(/*cached_tokens=*/{0},
                                       /*active_rows=*/0),
               "active_rows must be positive");
}

TEST(LinearStateRestoreTest, RejectsNonDivisibleActiveRows) {
  EXPECT_DEATH(build_linear_state_mask(/*cached_tokens=*/{0, 8},
                                       /*active_rows=*/3),
               "logical rows must evenly divide active rows");
}

struct LinearStateTestCache {
  std::vector<KVCache> kv_caches;
  torch::Tensor conv_cache;
  torch::Tensor ssm_cache;
};

LinearStateTestCache make_cache(int64_t num_slots = 4,
                                int64_t checkpoint_stride = 1) {
  LinearStateTestCache cache;
  cache.conv_cache = torch::full({num_slots, 2, 3}, 7.0, torch::kFloat32);
  cache.ssm_cache =
      torch::full({num_slots * checkpoint_stride, 2, 2}, 11.0, torch::kFloat32);
  cache.kv_caches.emplace_back(
      LinearAttentionKVCacheTensors{cache.conv_cache, cache.ssm_cache});
  return cache;
}

TEST(LinearStateRestoreTest, PartialLinearCacheLayoutFailsClosed) {
  LinearStateTestCache cache = make_cache();
  cache.kv_caches.emplace_back(LinearAttentionKVCacheTensors{
      torch::zeros({4, 2, 3}, torch::kFloat32), torch::Tensor()});

  std::vector<int64_t> validity_mask = {0};
  EXPECT_DEATH(restore_linear_state_slots(
                   cache.kv_caches, {2}, {1}, validity_mask, false),
               "must provide both conv and ssm caches");
}

TEST(LinearStateRestoreTest, EmptySsmCheckpointLayoutFailsClosed) {
  LinearStateTestCache cache = make_cache();
  cache.kv_caches.clear();
  cache.kv_caches.emplace_back(
      LinearAttentionKVCacheTensors{torch::zeros({4, 2, 3}, torch::kFloat32),
                                    torch::zeros({0, 2, 2}, torch::kFloat32)});

  std::vector<int64_t> validity_mask = {0};
  EXPECT_DEATH(restore_linear_state_slots(
                   cache.kv_caches, {2}, {1}, validity_mask, false),
               "ssm cache must contain checkpoint rows");
}

TEST(LinearStateRestoreTest, RejectsNonDivisibleCheckpointLayout) {
  LinearStateTestCache cache = make_cache();
  cache.kv_caches.emplace_back(
      LinearAttentionKVCacheTensors{torch::zeros({4, 2, 3}, torch::kFloat32),
                                    torch::zeros({5, 2, 2}, torch::kFloat32)});
  std::vector<int64_t> validity_mask = {0};
  EXPECT_DEATH(restore_linear_state_slots(
                   cache.kv_caches, {2}, {1}, validity_mask, false),
               "physical rows must be divisible");
}

TEST(LinearStateRestoreTest, CopiesOnlyDestinationAcrossLayers) {
  LinearStateTestCache cache = make_cache(6, 3);
  cache.conv_cache[1].fill_(19.0);
  cache.ssm_cache.narrow(0, 3, 3).fill_(23.0);
  torch::Tensor second_conv = torch::full({6, 2, 3}, 29.0, torch::kFloat32);
  torch::Tensor second_ssm = torch::full({12, 2, 2}, 31.0, torch::kFloat32);
  second_conv[1].fill_(37.0);
  second_ssm.narrow(0, 2, 2).fill_(41.0);
  cache.kv_caches.emplace_back();
  cache.kv_caches.emplace_back(
      LinearAttentionKVCacheTensors{second_conv, second_ssm});
  torch::Tensor expected_conv = cache.conv_cache.clone();
  torch::Tensor expected_ssm = cache.ssm_cache.clone();
  torch::Tensor expected_second_conv = second_conv.clone();
  torch::Tensor expected_second_ssm = second_ssm.clone();
  expected_conv[4].copy_(expected_conv[1]);
  expected_ssm.narrow(0, 12, 3).copy_(expected_ssm.narrow(0, 3, 3));
  expected_second_conv[4].copy_(expected_second_conv[1]);
  expected_second_ssm.narrow(0, 8, 2).copy_(
      expected_second_ssm.narrow(0, 2, 2));

  std::vector<int64_t> validity_mask = {0};
  restore_linear_state_slots(cache.kv_caches, {4}, {1}, validity_mask, false);
  EXPECT_EQ(validity_mask, std::vector<int64_t>({1}));

  EXPECT_TRUE(torch::equal(cache.conv_cache, expected_conv));
  EXPECT_TRUE(torch::equal(cache.ssm_cache, expected_ssm));
  EXPECT_TRUE(torch::equal(second_conv, expected_second_conv));
  EXPECT_TRUE(torch::equal(second_ssm, expected_second_ssm));
}

TEST(LinearStateRestoreTest, BatchDirectReadPreservesAllSlots) {
  LinearStateTestCache cache = make_cache(6, 3);
  const torch::Tensor original_conv = cache.conv_cache.clone();
  const torch::Tensor original_ssm = cache.ssm_cache.clone();
  std::vector<int64_t> validity_mask = {0, 0, 1, 0};

  restore_linear_state_slots(
      cache.kv_caches, {1, 2, 0, 4}, {3, 2, 0, 4}, validity_mask, true);

  EXPECT_EQ(validity_mask, std::vector<int64_t>({1, 0, 0, 0}));
  EXPECT_TRUE(torch::equal(cache.conv_cache, original_conv));
  EXPECT_TRUE(torch::equal(cache.ssm_cache, original_ssm));
}

TEST(LinearStateRestoreTest, BatchInPlaceRestoresExpandedMixedRows) {
  LinearStateTestCache cache = make_cache(6, 3);
  cache.conv_cache[3].fill_(19.0);
  cache.ssm_cache.narrow(0, 9, 3).fill_(23.0);
  cache.conv_cache[5].fill_(29.0);
  cache.ssm_cache.narrow(0, 15, 3).fill_(31.0);
  torch::Tensor expected_conv = cache.conv_cache.clone();
  torch::Tensor expected_ssm = cache.ssm_cache.clone();
  expected_conv[1].copy_(expected_conv[3]);
  expected_conv[4].copy_(expected_conv[5]);
  expected_ssm.narrow(0, 3, 3).copy_(expected_ssm.narrow(0, 9, 3));
  expected_ssm.narrow(0, 12, 3).copy_(expected_ssm.narrow(0, 15, 3));
  std::vector<int64_t> validity_mask = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};

  restore_linear_state_slots(
      cache.kv_caches, {1, 2, 0, 4}, {3, 2, 0, 5}, validity_mask, false);

  EXPECT_EQ(validity_mask,
            std::vector<int64_t>({1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1}));
  EXPECT_TRUE(torch::equal(cache.conv_cache, expected_conv));
  EXPECT_TRUE(torch::equal(cache.ssm_cache, expected_ssm));
}

TEST(LinearStateRestoreTest, BatchMissingReadIdsPreservesColdAndWarmRows) {
  for (const bool reads_distinct_state : {false, true}) {
    SCOPED_TRACE(reads_distinct_state);
    LinearStateTestCache cache = make_cache();
    const torch::Tensor original_conv = cache.conv_cache.clone();
    const torch::Tensor original_ssm = cache.ssm_cache.clone();
    std::vector<int64_t> validity_mask = {0, 1, 1};

    restore_linear_state_slots(
        cache.kv_caches, {1, 2, 0}, {}, validity_mask, reads_distinct_state);

    EXPECT_EQ(validity_mask, std::vector<int64_t>({0, 1, 0}));
    EXPECT_TRUE(torch::equal(cache.conv_cache, original_conv));
    EXPECT_TRUE(torch::equal(cache.ssm_cache, original_ssm));
  }
}

TEST(LinearStateRestoreTest, BatchRejectsMismatchedRows) {
  LinearStateTestCache cache = make_cache();
  std::vector<int64_t> validity_mask = {0, 1};
  EXPECT_DEATH(restore_linear_state_slots(
                   cache.kv_caches, {1, 2}, {1}, validity_mask, true),
               "read/write rows must match");
}

TEST(LinearStateRestoreTest, BatchRejectsInvalidMaskShape) {
  LinearStateTestCache cache = make_cache();
  std::vector<int64_t> validity_mask = {1};
  EXPECT_DEATH(restore_linear_state_slots(
                   cache.kv_caches, {1, 2}, {}, validity_mask, true),
               "must cover every logical row");
  validity_mask = {0, 1, 1};
  EXPECT_DEATH(restore_linear_state_slots(
                   cache.kv_caches, {1, 2}, {}, validity_mask, true),
               "must evenly expand logical rows");
}

TEST(LinearStateRestoreTest, BatchRejectsNonBinaryMask) {
  LinearStateTestCache cache = make_cache();
  std::vector<int64_t> validity_mask = {2};
  EXPECT_DEATH(restore_linear_state_slots(
                   cache.kv_caches, {1}, {1}, validity_mask, true),
               "validity entries must be 0 or 1");
}

TEST(LinearStateRestoreTest, BatchRejectsInvalidSlotIds) {
  LinearStateTestCache cache = make_cache();
  std::vector<int64_t> validity_mask = {1};
  EXPECT_DEATH(restore_linear_state_slots(
                   cache.kv_caches, {-1}, {1}, validity_mask, true),
               "write_ids");
  EXPECT_DEATH(restore_linear_state_slots(
                   cache.kv_caches, {4}, {1}, validity_mask, true),
               "write_ids");
  EXPECT_DEATH(restore_linear_state_slots(
                   cache.kv_caches, {1}, {-1}, validity_mask, true),
               "source_ids");
  EXPECT_DEATH(restore_linear_state_slots(
                   cache.kv_caches, {1}, {4}, validity_mask, true),
               "source_ids");
  EXPECT_DEATH(restore_linear_state_slots(
                   cache.kv_caches, {0}, {1}, validity_mask, false),
               "padding must not be used as a real linear-state");
  EXPECT_DEATH(restore_linear_state_slots(
                   cache.kv_caches, {1}, {0}, validity_mask, false),
               "padding must not be used as a real linear-state");
}

TEST(LinearStateRestoreTest, BatchRejectsMissingCache) {
  std::vector<KVCache> caches;
  std::vector<int64_t> validity_mask = {1};
  EXPECT_DEATH(
      restore_linear_state_slots(caches, {1}, {2}, validity_mask, true),
      "requires an allocated recurrent cache");
}

TEST(LinearStateRestoreTest, BatchValidatesAllLayersBeforeDirectRead) {
  LinearStateTestCache cache = make_cache();
  cache.kv_caches.emplace_back(
      LinearAttentionKVCacheTensors{torch::zeros({5, 2, 3}, torch::kFloat32),
                                    torch::zeros({5, 2, 2}, torch::kFloat32)});
  std::vector<int64_t> validity_mask = {0};
  EXPECT_DEATH(restore_linear_state_slots(
                   cache.kv_caches, {1}, {2}, validity_mask, true),
               "slot count must match across layers");
}

TEST(LinearStateRestoreTest, KPoolTailFollowsCheckpointForkAndSlotReset) {
  LinearStateTestCache cache = make_cache();
  IndexedKVCacheTensors tensors;
  tensors.kpool_tail = torch::arange(4 * 2 * 16 * 8, torch::kFloat32)
                           .reshape({4, 2, 16, 8})
                           .to(torch::kBFloat16);
  const torch::Tensor checkpoint = tensors.kpool_tail[1].clone();
  cache.kv_caches.emplace_back(tensors);
  std::vector<int64_t> validity_mask = {0};
  restore_linear_state_slots(cache.kv_caches, {2}, {1}, validity_mask, false);
  EXPECT_EQ(validity_mask, std::vector<int64_t>({1}));
  EXPECT_TRUE(torch::equal(tensors.kpool_tail[2], checkpoint));
  tensors.kpool_tail[2].fill_(-1);
  EXPECT_TRUE(torch::equal(tensors.kpool_tail[1], checkpoint));
  const torch::Tensor original_conv = cache.conv_cache.clone();
  const torch::Tensor original_ssm = cache.ssm_cache.clone();
  validity_mask = {0};
  restore_linear_state_slots(cache.kv_caches, {2}, {2}, validity_mask, false);
#if defined(USE_NPU)
  EXPECT_EQ(tensors.kpool_tail[2][0].count_nonzero().item<int64_t>(), 0);
  EXPECT_TRUE(torch::isneginf(tensors.kpool_tail[2][1]).all().item<bool>());
#else
  EXPECT_EQ(tensors.kpool_tail[2].count_nonzero().item<int64_t>(), 0);
#endif
  EXPECT_TRUE(torch::equal(tensors.kpool_tail[1], checkpoint));
  EXPECT_EQ(validity_mask, std::vector<int64_t>({0}));
  EXPECT_TRUE(torch::equal(cache.conv_cache, original_conv));
  EXPECT_TRUE(torch::equal(cache.ssm_cache, original_ssm));
}

TEST(LinearStateRestoreTest, KPoolOnlyCachePreservesWarmAndPaddingSlots) {
  IndexedKVCacheTensors tensors;
  tensors.kpool_tail = torch::full({7, 2, 16, 8}, 7.0, torch::kBFloat16);
  tensors.kpool_tail[1].fill_(11);
  const torch::Tensor checkpoint = tensors.kpool_tail[1].clone();
  const torch::Tensor padding = tensors.kpool_tail[0].clone();
  std::vector<KVCache> caches;
  caches.emplace_back(tensors);
  std::vector<int64_t> validity_mask = {0, 1};

  restore_linear_state_slots(caches, {2, 0}, {1, 0}, validity_mask, false);
  EXPECT_TRUE(torch::equal(tensors.kpool_tail[2], checkpoint));
  EXPECT_EQ(validity_mask, std::vector<int64_t>({1, 0}));

  restore_linear_state_slots(caches, {2, 0}, {2, 0}, validity_mask, false);
  EXPECT_TRUE(torch::equal(tensors.kpool_tail[2], checkpoint));
  EXPECT_TRUE(torch::equal(tensors.kpool_tail[0], padding));

  validity_mask = {0, 0};
  restore_linear_state_slots(caches, {2, 0}, {2, 0}, validity_mask, false);
#if defined(USE_NPU)
  EXPECT_EQ(tensors.kpool_tail[2][0].count_nonzero().item<int64_t>(), 0);
  EXPECT_TRUE(torch::isneginf(tensors.kpool_tail[2][1]).all().item<bool>());
#else
  EXPECT_EQ(tensors.kpool_tail[2].count_nonzero().item<int64_t>(), 0);
#endif
  EXPECT_TRUE(torch::equal(tensors.kpool_tail[1], checkpoint));
  EXPECT_TRUE(torch::equal(tensors.kpool_tail[0], padding));

  validity_mask = {0, 0, 0, 0, 1, 1};
  restore_linear_state_slots(caches,
                             {4, 3, 5, 6, 1, 0},
                             {4, 3, 1, 6, 1, 0},
                             validity_mask,
                             /*reads_distinct_state=*/false);
  EXPECT_EQ(validity_mask, std::vector<int64_t>({0, 0, 1, 0, 1, 0}));
  const torch::Tensor reset = tensors.kpool_tail.index_select(
      0, torch::tensor({3, 4, 6}, torch::kInt64));
#if defined(USE_NPU)
  EXPECT_EQ(reset.select(1, 0).count_nonzero().item<int64_t>(), 0);
  EXPECT_TRUE(torch::isneginf(reset.select(1, 1)).all().item<bool>());
#else
  EXPECT_EQ(reset.count_nonzero().item<int64_t>(), 0);
#endif
  EXPECT_TRUE(torch::equal(tensors.kpool_tail[5], checkpoint));
  EXPECT_TRUE(torch::equal(tensors.kpool_tail[1], checkpoint));
  EXPECT_TRUE(torch::equal(tensors.kpool_tail[0], padding));
}

struct StaticSharedTestCache {
  std::shared_ptr<StaticKVCacheLayout> layout;
  std::unique_ptr<StaticKVCacheBlockPool> pool;
  std::vector<KVCache> caches;

  std::vector<int32_t> allocate(size_t count, BlockType type) {
    std::vector<int32_t> ids = pool->allocate(count, type);
    const size_t groups = type == BlockType::KV ? layout->full_layer_count()
                                                : layout->num_linear_groups();
    std::vector<int32_t> rows;
    rows.reserve(ids.size() * (groups + 1));
    for (const int32_t id : ids) {
      rows.emplace_back(id);
      if (type == BlockType::KV) {
        pool->append_full_block_rows(id, &rows);
      } else {
        pool->append_linear_state_rows(id, &rows);
      }
    }
    if (type == BlockType::KV) {
      caches.front().update_full_block_rows(rows);
    } else {
      update_linear_state_slots(caches, rows);
    }
    return ids;
  }
};

StaticSharedTestCache make_static_shared_cache(int64_t n_layers = 2,
                                               int64_t checkpoint_stride = 2,
                                               int64_t conv_state_len = 2) {
  KVCacheCapacity capacity;
  capacity.n_blocks(32)
      .shared_slab_count(20)
      .block_size(2)
      .num_linear_state_blocks(32)
      .linear_conv_state_len(conv_state_len)
      .linear_ssm_checkpoint_stride(checkpoint_stride);
  ModelArgs args;
  args.model_type("qwen3_next")
      .n_layers(n_layers)
      .n_heads(1)
      .n_kv_heads(1)
      .head_dim(2)
      .full_attention_interval(2)
      .linear_num_key_heads(1)
      .linear_num_value_heads(1)
      .linear_key_head_dim(2)
      .linear_value_head_dim(2)
      .linear_conv_kernel_dim(3);
  const KVCacheShape shape(capacity, args, /*world_size=*/1);
  KVCacheCreateOptions options;
  options.device(torch::kCPU)
      .dtype(torch::kFloat32)
      .ssm_dtype(torch::kFloat32)
      .num_layers(n_layers)
      .model_type(args.model_type())
      .enable_linear_attention(true)
      .full_attention_interval(2);
  StaticSharedTestCache cache;
  cache.layout = std::make_shared<StaticKVCacheLayout>(shape, options);
  cache.pool = std::make_unique<StaticKVCacheBlockPool>(32, *cache.layout);
  options.static_layout(cache.layout)
      .linear_state_slot_table(std::make_shared<LinearStateSlotTable>(
          cache.layout->num_slots(),
          cache.layout->num_linear_groups(),
          torch::kCPU))
      .full_block_slot_table(std::make_shared<LinearStateSlotTable>(
          cache.layout->num_slots(),
          cache.layout->full_layer_count(),
          torch::kCPU))
      .tensor_allocator(std::make_shared<StaticKVCacheTensorAllocator>(
          *cache.layout, torch::kCPU));
  allocate_kv_caches(cache.caches, shape, options);
  return cache;
}

TEST(LinearStateRestoreTest, StaticCopyFollowsEachLayerAndRoleMapping) {
  auto fixture = make_static_shared_cache(/*n_layers=*/4);
  auto& caches = fixture.caches;
  const auto ids = fixture.allocate(2, BlockType::LINEAR);
  ASSERT_EQ(ids.size(), 2u);
  const int32_t source = ids[0];
  const int32_t destination = ids[1];
  std::vector<int32_t> source_rows;
  std::vector<int32_t> destination_rows;
  fixture.pool->append_linear_state_rows(source, &source_rows);
  fixture.pool->append_linear_state_rows(destination, &destination_rows);
  ASSERT_EQ(source_rows.size(), 4u);
  ASSERT_EQ(destination_rows.size(), 4u);
  EXPECT_NE(source_rows[0], source_rows[2]);
  EXPECT_NE(source_rows[1], source_rows[3]);
  EXPECT_NE(source_rows[0], source);

  for (size_t layer = 0; layer < 2; ++layer) {
    const KVCache& cache = caches[layer * 2];
    EXPECT_EQ(cache.cache_row_stride(KVCacheTensorRole::CONV), 1);
    EXPECT_EQ(cache.cache_row_stride(KVCacheTensorRole::SSM), 1);
    EXPECT_EQ(cache.linear_state_slot_row().sizes().vec(),
              (std::vector<int64_t>{2, 32}));
    EXPECT_EQ(cache.linear_state_slot_row()[0][source].item<int32_t>(),
              source_rows[layer * 2]);
    EXPECT_EQ(cache.linear_state_slot_row()[1][source].item<int32_t>(),
              source_rows[layer * 2 + 1]);
    EXPECT_EQ(cache.physical_row_index(source, KVCacheTensorRole::CONV),
              source_rows[layer * 2]);
    EXPECT_EQ(cache.physical_row_index(source, KVCacheTensorRole::SSM),
              source_rows[layer * 2 + 1]);
    EXPECT_TRUE(torch::equal(
        cache.physical_row_indices(
            torch::tensor({-2, -1, 0, source, destination}, torch::kInt32),
            KVCacheTensorRole::SSM),
        torch::tensor({-2,
                       -1,
                       0,
                       source_rows[layer * 2 + 1],
                       destination_rows[layer * 2 + 1]},
                      torch::kInt32)));
    cache.get_conv_cache()[source_rows[layer * 2]].fill_(11 + layer);
    cache.get_ssm_cache()
        .narrow(0, source_rows[layer * 2 + 1], 2)
        .fill_(21 + layer);
  }
  std::vector<int64_t> validity_mask = {0};
  restore_linear_state_slots(caches,
                             {destination},
                             {source},
                             validity_mask,
                             /*reads_distinct_state=*/false);

  EXPECT_EQ(validity_mask, std::vector<int64_t>({1}));
  for (size_t layer = 0; layer < 2; ++layer) {
    const KVCache& cache = caches[layer * 2];
    EXPECT_TRUE(cache.get_conv_cache()[destination_rows[layer * 2]]
                    .eq(11 + layer)
                    .all()
                    .item<bool>());
    EXPECT_TRUE(cache.get_ssm_cache()
                    .narrow(0, destination_rows[layer * 2 + 1], 2)
                    .eq(21 + layer)
                    .all()
                    .item<bool>());
  }
}

TEST(LinearStateRestoreTest, StaticColdReuseClearsOnlyItsOwnRecurrentRows) {
  auto fixture = make_static_shared_cache();
  auto& caches = fixture.caches;
  const KVCache& linear = caches[0];
  const KVCache& full = caches[1];
  ASSERT_TRUE(linear.is_static_shared());
  EXPECT_EQ(linear.linear_state_slot_count(), 32);
  EXPECT_EQ(linear.linear_checkpoint_stride(), 2);
  const auto old_full_ids = fixture.allocate(1, BlockType::KV);
  ASSERT_EQ(old_full_ids.size(), 1u);
  const int64_t released_row =
      full.physical_row_index(old_full_ids[0], KVCacheTensorRole::KEY);
  full.get_k_cache()[released_row].fill_(17);
  full.get_v_cache()[released_row].fill_(19);
  fixture.pool->release(old_full_ids[0], BlockType::KV);

  // The empty Full slab changes type and its bytes back a cold Conv state.
  const auto states = fixture.allocate(2, BlockType::LINEAR);
  ASSERT_EQ(states.size(), 2u);
  const int64_t cold_conv =
      linear.physical_row_index(states[0], KVCacheTensorRole::CONV);
  const int64_t cold_ssm =
      linear.physical_row_index(states[0], KVCacheTensorRole::SSM);
  const int64_t warm_conv =
      linear.physical_row_index(states[1], KVCacheTensorRole::CONV);
  const int64_t warm_ssm =
      linear.physical_row_index(states[1], KVCacheTensorRole::SSM);
  ASSERT_EQ(full.get_k_cache()[released_row].data_ptr(),
            linear.get_conv_cache()[cold_conv].data_ptr());
  ASSERT_GT(linear.get_conv_cache()[cold_conv].count_nonzero().item<int64_t>(),
            0);
  linear.get_conv_cache()[cold_conv].fill_(23);
  linear.get_ssm_cache().narrow(0, cold_ssm, 2).fill_(29);
  linear.get_conv_cache()[warm_conv].fill_(37);
  linear.get_ssm_cache().narrow(0, warm_ssm, 2).fill_(41);
  const auto live_full_ids = fixture.allocate(1, BlockType::KV);
  ASSERT_EQ(live_full_ids.size(), 1u);
  const int64_t live_row =
      full.physical_row_index(live_full_ids[0], KVCacheTensorRole::KEY);
  full.get_k_cache()[live_row].fill_(43);
  full.get_v_cache()[live_row].fill_(47);
  const int64_t scratch_rows = static_cast<int64_t>(
      fixture.layout->rows_per_slab(KVCacheTensorRole::KEY));
  full.get_k_cache().narrow(0, 0, scratch_rows).fill_(53);
  const torch::Tensor scratch =
      full.get_k_cache().narrow(0, 0, scratch_rows).clone();
  std::vector<int64_t> validity_mask = {0, 1, 1};

  restore_linear_state_slots(caches,
                             {states[0], states[1], 0},
                             {},
                             validity_mask,
                             /*reads_distinct_state=*/true);

  EXPECT_EQ(validity_mask, std::vector<int64_t>({0, 1, 0}));
  EXPECT_EQ(linear.get_conv_cache()[cold_conv].count_nonzero().item<int64_t>(),
            0);
  EXPECT_EQ(linear.get_ssm_cache()
                .narrow(0, cold_ssm, 2)
                .count_nonzero()
                .item<int64_t>(),
            0);
  EXPECT_TRUE(linear.get_conv_cache()[warm_conv].eq(37).all().item<bool>());
  EXPECT_TRUE(
      linear.get_ssm_cache().narrow(0, warm_ssm, 2).eq(41).all().item<bool>());
  EXPECT_TRUE(full.get_k_cache()[live_row].eq(43).all().item<bool>());
  EXPECT_TRUE(full.get_v_cache()[live_row].eq(47).all().item<bool>());
  EXPECT_TRUE(
      torch::equal(full.get_k_cache().narrow(0, 0, scratch_rows), scratch));
}

TEST(LinearStateRestoreTest, StaticFullIndicesPreserveOffsetsAndSentinels) {
  auto fixture = make_static_shared_cache(/*n_layers=*/4);
  const auto ids = fixture.allocate(3, BlockType::KV);
  ASSERT_EQ(ids.size(), 3u);
  std::vector<int32_t> first_rows;
  fixture.pool->append_full_block_rows(ids[0], &first_rows);
  ASSERT_EQ(first_rows.size(), 2u);
  EXPECT_NE(first_rows[0], first_rows[1]);

  for (size_t layer = 0; layer < 2; ++layer) {
    const KVCache& full = fixture.caches[layer * 2 + 1];
    std::vector<int64_t> expected_rows = {-2, -1, 0};
    std::vector<int64_t> logical_rows = {-2, -1, 0};
    std::vector<int64_t> expected_tokens = {-2, -1, 0, 1};
    std::vector<int64_t> logical_tokens = {-2, -1, 0, 1};
    for (const int32_t id : ids) {
      std::vector<int32_t> rows;
      fixture.pool->append_full_block_rows(id, &rows);
      const int64_t row = rows[layer];
      logical_rows.emplace_back(id);
      expected_rows.emplace_back(row);
      logical_tokens.emplace_back(id * 2);
      logical_tokens.emplace_back(id * 2 + 1);
      expected_tokens.emplace_back(row * 2);
      expected_tokens.emplace_back(row * 2 + 1);
      EXPECT_EQ(full.physical_row_index(id, KVCacheTensorRole::KEY), row);
      EXPECT_EQ(full.physical_row_index(id, KVCacheTensorRole::VALUE), row);
      EXPECT_EQ(full.full_block_rows()[id].item<int32_t>(), row);
      EXPECT_EQ(full.get_v_cache()[row].data_ptr(),
                full.get_k_cache()[row + 1].data_ptr());
    }
    EXPECT_EQ(full.full_block_rows().sizes().vec(), (std::vector<int64_t>{32}));
    EXPECT_EQ(full.full_block_rows()[0].item<int32_t>(), 0);
    EXPECT_EQ(full.physical_row_index(-1, KVCacheTensorRole::KEY), -1);
    EXPECT_EQ(full.cache_row_stride(KVCacheTensorRole::KEY), 1);
    EXPECT_TRUE(torch::equal(
        full.physical_row_indices(torch::tensor(logical_rows, torch::kInt64),
                                  KVCacheTensorRole::KEY),
        torch::tensor(expected_rows, torch::kInt64)));
    EXPECT_TRUE(torch::equal(
        full.physical_row_indices(
            torch::tensor(logical_rows, torch::kInt32).reshape({2, 3}),
            KVCacheTensorRole::KEY),
        torch::tensor(expected_rows, torch::kInt32).reshape({2, 3})));
    EXPECT_TRUE(torch::equal(
        full.physical_token_slots(torch::tensor(logical_tokens, torch::kInt64),
                                  /*block_size=*/2),
        torch::tensor(expected_tokens, torch::kInt64)));
  }
}

TEST(LinearStateRestoreTest, StaticRestoreCopiesEverySpeculativeCheckpoint) {
  auto fixture = make_static_shared_cache(/*n_layers=*/4,
                                          /*checkpoint_stride=*/5,
                                          /*conv_state_len=*/7);
  auto& caches = fixture.caches;
  const auto states = fixture.allocate(3, BlockType::LINEAR);
  ASSERT_EQ(states.size(), 3u);
  const auto full_ids = fixture.allocate(1, BlockType::KV);
  ASSERT_EQ(full_ids.size(), 1u);
  for (size_t layer = 0; layer < 2; ++layer) {
    const KVCache& linear = caches[layer * 2];
    const KVCache& full = caches[layer * 2 + 1];
    const int64_t source_conv =
        linear.physical_row_index(states[0], KVCacheTensorRole::CONV);
    const int64_t source_ssm =
        linear.physical_row_index(states[0], KVCacheTensorRole::SSM);
    const int64_t cold_conv =
        linear.physical_row_index(states[2], KVCacheTensorRole::CONV);
    const int64_t cold_ssm =
        linear.physical_row_index(states[2], KVCacheTensorRole::SSM);
    EXPECT_GT(source_ssm, linear.linear_state_slot_count());
    linear.get_conv_cache()[source_conv].fill_(11 + layer);
    for (int64_t checkpoint = 0; checkpoint < 5; ++checkpoint) {
      linear.get_ssm_cache()[source_ssm + checkpoint].fill_(21 + layer * 10 +
                                                            checkpoint);
    }
    linear.get_conv_cache()[cold_conv].fill_(41);
    linear.get_ssm_cache().narrow(0, cold_ssm, 5).fill_(43);
    const int64_t full_row =
        full.physical_row_index(full_ids[0], KVCacheTensorRole::KEY);
    full.get_k_cache()[full_row].fill_(51 + layer);
    full.get_v_cache()[full_row].fill_(61 + layer);
  }
  std::vector<int64_t> validity_mask = {0, 0};
  restore_linear_state_slots(caches,
                             {states[1], states[2]},
                             {states[0], states[2]},
                             validity_mask,
                             /*reads_distinct_state=*/false);

  EXPECT_EQ(validity_mask, std::vector<int64_t>({1, 0}));
  for (size_t layer = 0; layer < 2; ++layer) {
    const KVCache& linear = caches[layer * 2];
    const KVCache& full = caches[layer * 2 + 1];
    const int64_t source_ssm =
        linear.physical_row_index(states[0], KVCacheTensorRole::SSM);
    const int64_t destination_conv =
        linear.physical_row_index(states[1], KVCacheTensorRole::CONV);
    const int64_t destination_ssm =
        linear.physical_row_index(states[1], KVCacheTensorRole::SSM);
    const int64_t cold_conv =
        linear.physical_row_index(states[2], KVCacheTensorRole::CONV);
    const int64_t cold_ssm =
        linear.physical_row_index(states[2], KVCacheTensorRole::SSM);
    EXPECT_TRUE(linear.get_conv_cache()[destination_conv]
                    .eq(11 + layer)
                    .all()
                    .item<bool>());
    for (int64_t checkpoint = 0; checkpoint < 5; ++checkpoint) {
      EXPECT_TRUE(linear.get_ssm_cache()[destination_ssm + checkpoint]
                      .eq(21 + layer * 10 + checkpoint)
                      .all()
                      .item<bool>());
      EXPECT_TRUE(linear.get_ssm_cache()[source_ssm + checkpoint]
                      .eq(21 + layer * 10 + checkpoint)
                      .all()
                      .item<bool>());
    }
    EXPECT_EQ(
        linear.get_conv_cache()[cold_conv].count_nonzero().item<int64_t>(), 0);
    EXPECT_EQ(linear.get_ssm_cache()
                  .narrow(0, cold_ssm, 5)
                  .count_nonzero()
                  .item<int64_t>(),
              0);
    const int64_t full_row =
        full.physical_row_index(full_ids[0], KVCacheTensorRole::KEY);
    EXPECT_TRUE(full.get_k_cache()[full_row].eq(51 + layer).all().item<bool>());
    EXPECT_TRUE(full.get_v_cache()[full_row].eq(61 + layer).all().item<bool>());
    linear.get_ssm_cache()[destination_ssm].fill_(71);
    EXPECT_TRUE(linear.get_ssm_cache()[source_ssm]
                    .eq(21 + layer * 10)
                    .all()
                    .item<bool>());
  }
}

}  // namespace
}  // namespace xllm

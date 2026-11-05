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

#include <cstdint>
#include <memory>
#include <vector>

#include "core/kv_cache/storage/linear_state_slot_table.h"
#include "core/kv_cache/storage/static_kv_cache_tensor_allocator.h"

namespace xllm {
namespace {

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

std::vector<KVCache> make_static_shared_caches(int64_t n_layers = 2) {
  KVCacheCapacity capacity;
  capacity.n_blocks(32)
      .block_size(2)
      .num_linear_state_blocks(32)
      .linear_conv_state_len(2)
      .linear_ssm_checkpoint_stride(2);
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
  auto layout = std::make_shared<StaticKVCacheLayout>(shape, options);
  options.static_layout(layout)
      .linear_state_slot_table(std::make_shared<LinearStateSlotTable>(
          layout->num_slots(), layout->num_linear_groups(), torch::kCPU))
      .tensor_allocator(
          std::make_shared<StaticKVCacheTensorAllocator>(*layout, torch::kCPU));
  std::vector<KVCache> caches;
  allocate_kv_caches(caches, shape, options);
  return caches;
}

TEST(LinearStateRestoreTest, StaticCopyFollowsEachLinearGroupSlot) {
  // Linear layers 0 and 2 share one lane and form groups 0 and 1.
  auto caches = make_static_shared_caches(/*n_layers=*/4);
  ASSERT_EQ(caches.size(), 4u);
  const KVCache& first = caches[0];
  const KVCache& second = caches[2];
  ASSERT_EQ(first.cache_row_stride(KVCacheTensorRole::SSM), 9);
  // State 3 holds slots {3, 7}; state 5 holds slots {5, 9}.
  update_linear_state_slots(caches, {3, 3, 7, 5, 5, 9});
  EXPECT_EQ(first.physical_row_index(5, KVCacheTensorRole::SSM), 45);
  EXPECT_EQ(second.physical_row_index(5, KVCacheTensorRole::SSM), 81);
  EXPECT_TRUE(torch::equal(
      second.physical_row_indices(torch::tensor({-1, 0, 3, 5}, torch::kInt32),
                                  KVCacheTensorRole::SSM),
      torch::tensor({-1, 0, 63, 81}, torch::kInt32)));
  first.get_ssm_cache().narrow(0, 27, 2).fill_(11);
  second.get_ssm_cache().narrow(0, 63, 2).fill_(13);
  std::vector<int64_t> validity_mask = {0};

  restore_linear_state_slots(caches,
                             {5},
                             {3},
                             validity_mask,
                             /*reads_distinct_state=*/false);

  EXPECT_EQ(validity_mask, std::vector<int64_t>({1}));
  EXPECT_TRUE(first.get_ssm_cache().narrow(0, 45, 2).eq(11).all().item<bool>());
  EXPECT_TRUE(
      second.get_ssm_cache().narrow(0, 81, 2).eq(13).all().item<bool>());
}

TEST(LinearStateRestoreTest, StaticColdReuseClearsOnlyItsOwnRecurrentRows) {
  auto caches = make_static_shared_caches();
  ASSERT_EQ(caches.size(), 2u);
  const KVCache& linear = caches[0];
  const KVCache& full = caches[1];
  ASSERT_TRUE(linear.is_static_shared());
  ASSERT_EQ(linear.linear_state_slot_count(), 32);
  ASSERT_EQ(linear.linear_checkpoint_stride(), 2);
  ASSERT_EQ(linear.cache_row_stride(KVCacheTensorRole::CONV), 3);
  ASSERT_EQ(linear.cache_row_stride(KVCacheTensorRole::SSM), 9);

  // A released Full logical page at page 4 becomes a recurrent logical page.
  full.get_k_cache()[36].fill_(17);
  full.get_v_cache()[36].fill_(19);
  ASSERT_EQ(full.get_k_cache()[36].data_ptr(),
            linear.get_conv_cache()[12].data_ptr());
  ASSERT_GT(linear.get_conv_cache()[12].count_nonzero().item<int64_t>(), 0);
  linear.get_ssm_cache().narrow(0, 36, 2).fill_(23);
  full.get_k_cache()[63].fill_(29);
  full.get_v_cache()[63].fill_(31);
  linear.get_conv_cache()[30].fill_(37);
  linear.get_ssm_cache().narrow(0, 90, 2).fill_(41);
  std::vector<int64_t> validity_mask = {0, 1};

  restore_linear_state_slots(caches,
                             {4, 10},
                             {},
                             validity_mask,
                             /*reads_distinct_state=*/true);

  EXPECT_EQ(validity_mask, std::vector<int64_t>({0, 1}));
  EXPECT_EQ(linear.get_conv_cache()[12].count_nonzero().item<int64_t>(), 0);
  EXPECT_EQ(
      linear.get_ssm_cache().narrow(0, 36, 2).count_nonzero().item<int64_t>(),
      0);
  EXPECT_TRUE(full.get_k_cache()[63].eq(29).all().item<bool>());
  EXPECT_TRUE(full.get_v_cache()[63].eq(31).all().item<bool>());
  EXPECT_TRUE(linear.get_conv_cache()[30].eq(37).all().item<bool>());
  EXPECT_TRUE(
      linear.get_ssm_cache().narrow(0, 90, 2).eq(41).all().item<bool>());
}

TEST(LinearStateRestoreTest, StaticCheckpointCopyUsesIndependentRoleStrides) {
  auto caches = make_static_shared_caches();
  const KVCache& linear = caches[0];
  const KVCache& full = caches[1];
  linear.get_conv_cache()[12].fill_(11);
  linear.get_ssm_cache()[36].fill_(13);
  linear.get_ssm_cache()[37].fill_(17);
  full.get_k_cache()[126].fill_(23);
  full.get_v_cache()[126].fill_(29);
  std::vector<int64_t> validity_mask = {0};

  restore_linear_state_slots(caches,
                             {10},
                             {4},
                             validity_mask,
                             /*reads_distinct_state=*/false);

  EXPECT_EQ(validity_mask, std::vector<int64_t>({1}));
  EXPECT_TRUE(linear.get_conv_cache()[30].eq(11).all().item<bool>());
  EXPECT_TRUE(linear.get_ssm_cache()[90].eq(13).all().item<bool>());
  EXPECT_TRUE(linear.get_ssm_cache()[91].eq(17).all().item<bool>());
  linear.get_ssm_cache()[90].fill_(31);
  EXPECT_TRUE(linear.get_ssm_cache()[36].eq(13).all().item<bool>());
  EXPECT_TRUE(full.get_k_cache()[126].eq(23).all().item<bool>());
  EXPECT_TRUE(full.get_v_cache()[126].eq(29).all().item<bool>());
}

TEST(LinearStateRestoreTest, StaticFullIndicesPreserveOffsetsAndSentinels) {
  auto caches = make_static_shared_caches();
  const KVCache& full = caches[1];
  const torch::Tensor rows = torch::tensor({-2, -1, 0, 2, 30}, torch::kInt32);
  EXPECT_TRUE(
      torch::equal(full.physical_row_indices(rows, KVCacheTensorRole::KEY),
                   torch::tensor({-2, -1, 0, 18, 270}, torch::kInt32)));
  const torch::Tensor slots =
      torch::tensor({-2, -1, 0, 1, 2, 3, 8, 9, 61}, torch::kInt64);
  EXPECT_TRUE(torch::equal(
      full.physical_token_slots(slots, /*block_size=*/2),
      torch::tensor({-2, -1, 0, 1, 18, 19, 72, 73, 541}, torch::kInt64)));
  EXPECT_EQ(full.physical_row_index(-1, KVCacheTensorRole::KEY), -1);
  EXPECT_EQ(full.physical_row_index(30, KVCacheTensorRole::KEY), 270);
}

TEST(LinearStateRestoreTest, StaticCeilingConvMappingRestoresEveryCheckpoint) {
  KVCacheCapacity capacity;
  capacity.n_blocks(8)
      .block_size(2)
      .num_linear_state_blocks(8)
      .linear_conv_state_len(7)
      .linear_ssm_checkpoint_stride(5);
  ModelArgs args;
  args.model_type("qwen3_next")
      .n_layers(2)
      .n_heads(1)
      .n_kv_heads(1)
      .head_dim(2)
      .full_attention_interval(2)
      .linear_num_key_heads(1)
      .linear_num_value_heads(1)
      .linear_key_head_dim(128)
      .linear_value_head_dim(128)
      .linear_conv_kernel_dim(4);
  const KVCacheShape shape(capacity, args, /*world_size=*/1);
  KVCacheCreateOptions options;
  options.device(torch::kCPU)
      .dtype(torch::kFloat32)
      .ssm_dtype(torch::kFloat32)
      .num_layers(2)
      .model_type(args.model_type())
      .enable_linear_attention(true)
      .full_attention_interval(2);
  auto layout = std::make_shared<StaticKVCacheLayout>(shape, options);
  ASSERT_EQ(layout->slot_bytes(), 393216);
  options.static_layout(layout)
      .linear_state_slot_table(std::make_shared<LinearStateSlotTable>(
          layout->num_slots(), layout->num_linear_groups(), torch::kCPU))
      .tensor_allocator(
          std::make_shared<StaticKVCacheTensorAllocator>(*layout, torch::kCPU));
  std::vector<KVCache> caches;
  allocate_kv_caches(caches, shape, options);
  const KVCache& linear = caches[0];
  const KVCache& full = caches[1];
  const torch::Tensor logical =
      torch::tensor({-2, -1, 0, 1, 2, 6, 7, 10000000}, torch::kInt32);
  EXPECT_TRUE(torch::equal(
      linear.physical_row_indices(logical, KVCacheTensorRole::CONV),
      torch::tensor({-2, -1, 0, 37, 74, 220, 256, 365714286}, torch::kInt32)));
  EXPECT_EQ(linear.physical_row_index(10000000, KVCacheTensorRole::CONV),
            365714286);
  linear.get_conv_cache()[37].fill_(11);
  for (int64_t checkpoint = 0; checkpoint < 5; ++checkpoint) {
    linear.get_ssm_cache()[6 + checkpoint].fill_(13 + checkpoint);
  }
  linear.get_conv_cache()[110].fill_(41);
  linear.get_ssm_cache().narrow(0, 18, 5).fill_(43);
  full.get_k_cache()[98304].fill_(29);
  full.get_v_cache()[98304].fill_(31);
  std::vector<int64_t> validity_mask = {0, 0};

  restore_linear_state_slots(caches,
                             {2, 3},
                             {1, 3},
                             validity_mask,
                             /*reads_distinct_state=*/false);

  EXPECT_EQ(validity_mask, std::vector<int64_t>({1, 0}));
  EXPECT_TRUE(linear.get_conv_cache()[74].eq(11).all().item<bool>());
  EXPECT_TRUE(torch::equal(linear.get_ssm_cache().narrow(0, 12, 5),
                           linear.get_ssm_cache().narrow(0, 6, 5)));
  EXPECT_EQ(linear.get_conv_cache()[110].count_nonzero().item<int64_t>(), 0);
  EXPECT_EQ(
      linear.get_ssm_cache().narrow(0, 18, 5).count_nonzero().item<int64_t>(),
      0);
  EXPECT_TRUE(full.get_k_cache()[98304].eq(29).all().item<bool>());
  EXPECT_TRUE(full.get_v_cache()[98304].eq(31).all().item<bool>());
  linear.get_ssm_cache()[12].fill_(47);
  EXPECT_TRUE(linear.get_ssm_cache()[6].eq(13).all().item<bool>());
}

}  // namespace
}  // namespace xllm

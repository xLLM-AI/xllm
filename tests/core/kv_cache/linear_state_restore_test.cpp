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

#include "core/kv_cache/linear_state_restore.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "framework/model/model_params_view.h"

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
               "checkpoint layout mismatch");
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

}  // namespace
}  // namespace xllm

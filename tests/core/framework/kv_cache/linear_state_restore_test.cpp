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

#include "framework/kv_cache/linear_state_restore.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

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

TEST(LinearStateRestoreTest, ModelInputConversionPreservesValidityMask) {
  ModelInputParams input_params;
  input_params.linear_state_validity_mask = {0, 1, 1, 0};

  ModelInputParams converted = input_params.to(torch::Device(torch::kCPU));

  EXPECT_EQ(converted.linear_state_validity_mask,
            std::vector<int64_t>({0, 1, 1, 0}));
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

TEST(LinearStateRestoreTest, ColdStartClearsOnlyLiveSlotAndMarksItCold) {
  LinearStateTestCache cache = make_cache(/*num_slots=*/4,
                                          /*checkpoint_stride=*/2);
  const torch::Tensor untouched_conv = cache.conv_cache.select(0, 1).clone();
  const torch::Tensor untouched_ssm = cache.ssm_cache.narrow(0, 2, 2).clone();

  LinearStateCacheOp reset;
  reset.linear_state_id = 2;
  reset.reset_requested = true;
  std::vector<int64_t> validity_mask = {1};
  restore_linear_state_slots(cache.kv_caches, {reset}, validity_mask);

  EXPECT_EQ(validity_mask[0], 0);
  EXPECT_EQ(cache.conv_cache.select(0, 2).count_nonzero().item<int64_t>(), 0);
  EXPECT_EQ(cache.ssm_cache.narrow(0, 4, 2).count_nonzero().item<int64_t>(), 0);
  EXPECT_TRUE(torch::equal(cache.conv_cache.select(0, 1), untouched_conv));
  EXPECT_TRUE(torch::equal(cache.ssm_cache.narrow(0, 2, 2), untouched_ssm));
}

TEST(LinearStateRestoreTest, ColdStartsBatchContiguousSlotsAcrossLayers) {
  LinearStateTestCache cache = make_cache(/*num_slots=*/6,
                                          /*checkpoint_stride=*/2);
  torch::Tensor second_conv = torch::full({6, 2, 3}, 13.0, torch::kFloat32);
  torch::Tensor second_ssm = torch::full({12, 2, 2}, 17.0, torch::kFloat32);
  cache.kv_caches.emplace_back(
      LinearAttentionKVCacheTensors{second_conv, second_ssm});
  const torch::Tensor untouched_conv = cache.conv_cache.select(0, 5).clone();
  const torch::Tensor untouched_ssm = cache.ssm_cache.narrow(0, 10, 2).clone();
  const torch::Tensor untouched_second_conv = second_conv.select(0, 1).clone();
  const torch::Tensor untouched_second_ssm = second_ssm.narrow(0, 2, 2).clone();

  std::vector<LinearStateCacheOp> resets(3);
  for (size_t i = 0; i < resets.size(); ++i) {
    resets[i].linear_state_id = static_cast<int32_t>(i + 2);
    resets[i].reset_requested = true;
  }
  std::vector<int64_t> validity_mask = {1, 1, 1};
  restore_linear_state_slots(cache.kv_caches, resets, validity_mask);

  EXPECT_EQ(validity_mask, std::vector<int64_t>({0, 0, 0}));
  EXPECT_EQ(cache.conv_cache.narrow(0, 2, 3).count_nonzero().item<int64_t>(),
            0);
  EXPECT_EQ(cache.ssm_cache.narrow(0, 4, 6).count_nonzero().item<int64_t>(), 0);
  EXPECT_EQ(second_conv.narrow(0, 2, 3).count_nonzero().item<int64_t>(), 0);
  EXPECT_EQ(second_ssm.narrow(0, 4, 6).count_nonzero().item<int64_t>(), 0);
  EXPECT_TRUE(torch::equal(cache.conv_cache.select(0, 5), untouched_conv));
  EXPECT_TRUE(torch::equal(cache.ssm_cache.narrow(0, 10, 2), untouched_ssm));
  EXPECT_TRUE(torch::equal(second_conv.select(0, 1), untouched_second_conv));
  EXPECT_TRUE(torch::equal(second_ssm.narrow(0, 2, 2), untouched_second_ssm));
}

TEST(LinearStateRestoreTest, ColdStartsBatchDisjointRanges) {
  LinearStateTestCache cache = make_cache(/*num_slots=*/7,
                                          /*checkpoint_stride=*/2);
  const torch::Tensor untouched_conv_two =
      cache.conv_cache.select(0, 2).clone();
  const torch::Tensor untouched_conv_five =
      cache.conv_cache.select(0, 5).clone();
  const torch::Tensor untouched_ssm_two =
      cache.ssm_cache.narrow(0, 4, 2).clone();
  const torch::Tensor untouched_ssm_five =
      cache.ssm_cache.narrow(0, 10, 2).clone();

  std::vector<LinearStateCacheOp> resets(4);
  const std::vector<int32_t> reset_slots = {1, 3, 4, 6};
  for (size_t i = 0; i < resets.size(); ++i) {
    resets[i].linear_state_id = reset_slots[i];
    resets[i].reset_requested = true;
  }
  std::vector<int64_t> validity_mask = {1, 1, 1, 1};
  restore_linear_state_slots(cache.kv_caches, resets, validity_mask);

  EXPECT_EQ(validity_mask, std::vector<int64_t>({0, 0, 0, 0}));
  for (const int32_t slot_id : reset_slots) {
    EXPECT_EQ(
        cache.conv_cache.select(0, slot_id).count_nonzero().item<int64_t>(), 0);
    EXPECT_EQ(cache.ssm_cache.narrow(0, slot_id * 2, 2)
                  .count_nonzero()
                  .item<int64_t>(),
              0);
  }
  EXPECT_TRUE(torch::equal(cache.conv_cache.select(0, 2), untouched_conv_two));
  EXPECT_TRUE(torch::equal(cache.conv_cache.select(0, 5), untouched_conv_five));
  EXPECT_TRUE(torch::equal(cache.ssm_cache.narrow(0, 4, 2), untouched_ssm_two));
  EXPECT_TRUE(
      torch::equal(cache.ssm_cache.narrow(0, 10, 2), untouched_ssm_five));
}

TEST(LinearStateRestoreTest, RestoreCopiesCheckpointAndMarksItWarm) {
  LinearStateTestCache cache = make_cache(/*num_slots=*/4,
                                          /*checkpoint_stride=*/2);
  cache.conv_cache.select(0, 1).fill_(17.0);
  cache.conv_cache.select(0, 2).zero_();
  cache.ssm_cache.narrow(0, 2, 2).fill_(23.0);
  cache.ssm_cache.narrow(0, 4, 2).zero_();

  LinearStateCacheOp restore;
  restore.linear_state_id = 2;
  restore.restore_requested = true;
  restore.restore_src_slot_id = 1;
  std::vector<int64_t> validity_mask = {0};
  restore_linear_state_slots(cache.kv_caches, {restore}, validity_mask);

  EXPECT_EQ(validity_mask[0], 1);
  EXPECT_TRUE(torch::equal(cache.conv_cache.select(0, 2),
                           cache.conv_cache.select(0, 1)));
  EXPECT_TRUE(torch::equal(cache.ssm_cache.narrow(0, 4, 2),
                           cache.ssm_cache.narrow(0, 2, 2)));
}

TEST(LinearStateRestoreTest, MixedResetAndRestorePreserveMetadata) {
  LinearStateTestCache cache = make_cache(/*num_slots=*/6,
                                          /*checkpoint_stride=*/2);
  cache.conv_cache.select(0, 1).fill_(17.0);
  cache.ssm_cache.narrow(0, 2, 2).fill_(23.0);

  LinearStateCacheOp reset_two;
  reset_two.linear_state_id = 2;
  reset_two.reset_requested = true;
  LinearStateCacheOp restore_three;
  restore_three.linear_state_id = 3;
  restore_three.restore_requested = true;
  restore_three.restore_src_slot_id = 1;
  LinearStateCacheOp reset_four;
  reset_four.linear_state_id = 4;
  reset_four.reset_requested = true;
  std::vector<int64_t> validity_mask = {1, 0, 1};

  restore_linear_state_slots(
      cache.kv_caches, {reset_two, restore_three, reset_four}, validity_mask);

  EXPECT_EQ(validity_mask, std::vector<int64_t>({0, 1, 0}));
  EXPECT_EQ(cache.conv_cache.select(0, 2).count_nonzero().item<int64_t>(), 0);
  EXPECT_EQ(cache.ssm_cache.narrow(0, 4, 2).count_nonzero().item<int64_t>(), 0);
  EXPECT_TRUE(torch::equal(cache.conv_cache.select(0, 3),
                           cache.conv_cache.select(0, 1)));
  EXPECT_TRUE(torch::equal(cache.ssm_cache.narrow(0, 6, 2),
                           cache.ssm_cache.narrow(0, 2, 2)));
  EXPECT_EQ(cache.conv_cache.select(0, 4).count_nonzero().item<int64_t>(), 0);
  EXPECT_EQ(cache.ssm_cache.narrow(0, 8, 2).count_nonzero().item<int64_t>(), 0);
}

TEST(LinearStateRestoreTest, ResetAndRestoreTogetherFailsClosed) {
  LinearStateTestCache cache = make_cache();
  LinearStateCacheOp invalid;
  invalid.linear_state_id = 2;
  invalid.reset_requested = true;
  invalid.restore_requested = true;
  invalid.restore_src_slot_id = 1;
  std::vector<int64_t> validity_mask = {1};

  EXPECT_DEATH(
      restore_linear_state_slots(cache.kv_caches, {invalid}, validity_mask),
      "reset and restore are mutually exclusive");
}

TEST(LinearStateRestoreTest, ContinuedRequestPreservesStateAndMetadata) {
  LinearStateTestCache cache = make_cache();
  const torch::Tensor original_conv = cache.conv_cache.clone();
  const torch::Tensor original_ssm = cache.ssm_cache.clone();

  LinearStateCacheOp continued;
  continued.linear_state_id = 2;
  std::vector<int64_t> validity_mask = {1};
  restore_linear_state_slots(cache.kv_caches, {continued}, validity_mask);

  EXPECT_EQ(validity_mask[0], 1);
  EXPECT_TRUE(torch::equal(cache.conv_cache, original_conv));
  EXPECT_TRUE(torch::equal(cache.ssm_cache, original_ssm));
}

TEST(LinearStateRestoreTest, MissingRestoreSourceFailsClosed) {
  LinearStateTestCache cache = make_cache();
  LinearStateCacheOp restore;
  restore.linear_state_id = 2;
  restore.restore_requested = true;
  std::vector<int64_t> validity_mask = {1};

  EXPECT_DEATH(
      restore_linear_state_slots(cache.kv_caches, {restore}, validity_mask),
      "restore source must be a real non-padding slot");
}

TEST(LinearStateRestoreTest, PaddingLiveSlotFailsClosed) {
  LinearStateTestCache cache = make_cache();
  LinearStateCacheOp reset;
  reset.linear_state_id = 0;
  reset.reset_requested = true;
  std::vector<int64_t> validity_mask = {0};

  EXPECT_DEATH(
      restore_linear_state_slots(cache.kv_caches, {reset}, validity_mask),
      "live slot must be a real non-padding slot");
}

TEST(LinearStateRestoreTest, PaddingRestoreSourceFailsClosed) {
  LinearStateTestCache cache = make_cache();
  LinearStateCacheOp restore;
  restore.linear_state_id = 2;
  restore.restore_requested = true;
  restore.restore_src_slot_id = 0;
  std::vector<int64_t> validity_mask = {0};

  EXPECT_DEATH(
      restore_linear_state_slots(cache.kv_caches, {restore}, validity_mask),
      "restore source must be a real non-padding slot");
}

TEST(LinearStateRestoreTest, OutOfRangeSlotFailsClosed) {
  LinearStateTestCache cache = make_cache();
  LinearStateCacheOp continued;
  continued.linear_state_id = 4;
  std::vector<int64_t> validity_mask = {1};

  EXPECT_DEATH(
      restore_linear_state_slots(cache.kv_caches, {continued}, validity_mask),
      "live slot must be a real non-padding slot");
}

TEST(LinearStateRestoreTest, PartialLinearCacheLayoutFailsClosed) {
  LinearStateTestCache cache = make_cache();
  cache.kv_caches.emplace_back(LinearAttentionKVCacheTensors{
      torch::zeros({4, 2, 3}, torch::kFloat32), torch::Tensor()});
  LinearStateCacheOp continued;
  continued.linear_state_id = 2;
  std::vector<int64_t> validity_mask = {1};

  EXPECT_DEATH(
      restore_linear_state_slots(cache.kv_caches, {continued}, validity_mask),
      "must provide both conv and ssm caches");
}

TEST(LinearStateRestoreTest, EmptySsmCheckpointLayoutFailsClosed) {
  LinearStateTestCache cache = make_cache();
  cache.kv_caches.clear();
  cache.kv_caches.emplace_back(
      LinearAttentionKVCacheTensors{torch::zeros({4, 2, 3}, torch::kFloat32),
                                    torch::zeros({0, 2, 2}, torch::kFloat32)});
  LinearStateCacheOp reset;
  reset.linear_state_id = 2;
  reset.reset_requested = true;
  std::vector<int64_t> validity_mask = {0};

  EXPECT_DEATH(
      restore_linear_state_slots(cache.kv_caches, {reset}, validity_mask),
      "ssm cache must contain checkpoint rows");
}

}  // namespace
}  // namespace xllm

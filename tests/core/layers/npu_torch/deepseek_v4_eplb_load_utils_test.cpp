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

#include "core/layers/npu_torch/deepseek_v4_eplb_load_utils.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

namespace xllm::layer {
namespace {

TEST(DeepseekV4EplbLoadUtilsTest, ExcludesSyntheticEmptyRankPadding) {
  torch::Tensor global_mask = torch::tensor({true, true}, torch::kBool);
  torch::Tensor active_mask = dsv4_eplb::select_ep2_active_token_mask(
      global_mask, 1, {2, 0}, 1, /*all_dp_ranks_decode=*/true);

  EXPECT_TRUE(active_mask.defined());
  EXPECT_TRUE(torch::equal(active_mask, torch::tensor({false}, torch::kBool)));
}

TEST(DeepseekV4EplbLoadUtilsTest, KeepsMixedPhaseDispatchUnmasked) {
  torch::Tensor global_mask = torch::tensor({false, true, true}, torch::kBool);
  torch::Tensor active_mask = dsv4_eplb::select_ep2_active_token_mask(
      global_mask, 2, {1, 2}, 1, /*all_dp_ranks_decode=*/false);

  EXPECT_FALSE(active_mask.defined());
}

TEST(DeepseekV4EplbLoadUtilsTest, AppliesDecodeMaskPerTopKRoute) {
  torch::Tensor valid_weights = torch::ones({6}, torch::kInt64);
  torch::Tensor decode_mask = torch::tensor({false, true, true}, torch::kBool);

  EXPECT_TRUE(torch::equal(
      dsv4_eplb::apply_decode_token_mask(valid_weights, decode_mask, 2),
      torch::tensor({0, 0, 1, 1, 1, 1}, torch::kInt64)));
}

TEST(DeepseekV4EplbLoadUtilsTest,
     GraphLoadMaskExcludesPaddingWhenPrefillLoadIsEnabled) {
  torch::Tensor graph_mask_with_padding =
      torch::tensor({true, true, false, false}, torch::kBool);

  torch::Tensor graph_mask = dsv4_eplb::select_recorded_load_token_mask(
      graph_mask_with_padding,
      /*routed_token_count=*/4,
      /*dp_token_counts=*/{2},
      /*dp_rank=*/0,
      /*routed_tokens_are_dp_gathered=*/false,
      /*decode_only=*/false,
      /*enable_graph=*/true);
  EXPECT_TRUE(torch::equal(
      graph_mask, torch::tensor({true, true, false, false}, torch::kBool)));

  torch::Tensor eager_mask_without_padding =
      torch::tensor({true, true}, torch::kBool);
  torch::Tensor eager_mask = dsv4_eplb::select_recorded_load_token_mask(
      eager_mask_without_padding,
      /*routed_token_count=*/2,
      /*dp_token_counts=*/{2},
      /*dp_rank=*/0,
      /*routed_tokens_are_dp_gathered=*/false,
      /*decode_only=*/false,
      /*enable_graph=*/false);
  EXPECT_FALSE(eager_mask.defined());
}

TEST(DeepseekV4EplbLoadUtilsTest, WarmupDoesNotRecordDispatchLoad) {
  torch::Tensor expert_load_data = torch::full({1, 4}, 7, torch::kInt64);
  torch::Tensor receiver_counts = torch::tensor({2, 1, 3, 4}, torch::kInt32);

  dsv4_eplb::record_dispatch_expert_load(
      receiver_counts, expert_load_data, 0, /*is_graph_warmup=*/true);

  EXPECT_TRUE(
      torch::equal(expert_load_data, torch::full_like(expert_load_data, 7)));
}

}  // namespace
}  // namespace xllm::layer

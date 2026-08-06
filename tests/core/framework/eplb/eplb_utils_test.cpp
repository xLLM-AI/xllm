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

#include "core/framework/eplb/eplb_utils.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

namespace xllm {
namespace {

TEST(EplbUtilsTest, UsesWorkerNumWhenEpCoversAllWorkers) {
  EXPECT_EQ(eplb::effective_device_num(/*worker_num=*/8, /*ep_size=*/8), 8);
  EXPECT_EQ(eplb::local_physical_experts_num(/*num_logical_experts=*/256,
                                             /*eplb_device_num=*/8,
                                             /*redundant_experts_num=*/2),
            34);
}

TEST(EplbUtilsTest, MapsWorkerRanksDirectlyToEpRanks) {
  const int32_t worker_num = 8;
  const int32_t eplb_device_num = 8;

  EXPECT_EQ(eplb::eplb_rank_from_worker_rank(0, worker_num, eplb_device_num),
            0);
  EXPECT_EQ(eplb::eplb_rank_from_worker_rank(1, worker_num, eplb_device_num),
            1);
  EXPECT_EQ(eplb::eplb_rank_from_worker_rank(2, worker_num, eplb_device_num),
            2);
  EXPECT_EQ(eplb::eplb_rank_from_worker_rank(7, worker_num, eplb_device_num),
            7);
}

TEST(EplbUtilsTest, CountsOnlyCrossDeviceExpertTransfers) {
  const std::vector<int64_t> previous = {0, 1, 2, 3, 4, 5};
  const std::vector<int64_t> local_reorder = {1, 0, 3, 2, 5, 4};
  EXPECT_EQ(eplb::count_cross_device_migrations(
                previous, local_reorder, /*device_experts_num=*/2),
            0);

  const std::vector<int64_t> cross_device = {2, 1, 4, 3, 0, 5};
  EXPECT_EQ(eplb::count_cross_device_migrations(
                previous, cross_device, /*device_experts_num=*/2),
            3);
}

TEST(EplbUtilsTest, BuildsGlobalDecodeMaskForEveryDpRank) {
  const std::vector<torch::Tensor> local_masks = {
      torch::tensor({true, false}, torch::kBool),
      torch::empty({0}, torch::kBool),
      torch::tensor({true}, torch::kBool),
  };

  const torch::Tensor global_mask = eplb::build_global_decode_token_mask(
      local_masks, /*dp_token_counts=*/{2, 0, 1});

  EXPECT_TRUE(torch::equal(global_mask,
                           torch::tensor({true, false, true}, torch::kBool)));
}

TEST(EplbUtilsTest, ExpandsDecodeMaskForSpeculativeValidationRows) {
  const torch::Tensor decode_mask = torch::tensor({true, false}, torch::kBool);

  const torch::Tensor expanded_mask =
      eplb::expand_decode_token_mask(decode_mask, /*tokens_per_row=*/2);

  EXPECT_TRUE(torch::equal(
      expanded_mask, torch::tensor({true, true, false, false}, torch::kBool)));
}

}  // namespace
}  // namespace xllm

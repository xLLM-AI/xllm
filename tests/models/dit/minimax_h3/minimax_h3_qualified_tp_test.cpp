/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <vector>

#include "core/framework/parallel_state/rank_generator.h"
#include "models/dit/schedulers/minimax_h3_scheduler.h"
#include "models/dit/transformers/minimax_h3_tp_layout.h"
#include "models/dit/transformers/minimax_h3_tp_weights.h"

namespace xllm {
namespace {

TEST(QualifiedTpContract, AcceptsOnlyA3SixteenCardTp2U8) {
  EXPECT_TRUE(minimax_h3_topology_is_valid({16, 1, 2, 8, 1, 1, 1}));
  EXPECT_FALSE(minimax_h3_topology_is_valid({16, 1, 16, 1, 1, 1, 1}));
  EXPECT_FALSE(minimax_h3_topology_is_valid({16, 1, 2, 8, 1, 2, 1}));
  EXPECT_FALSE(minimax_h3_topology_is_valid({1, 1, 1, 1, 1, 1, 1}));
  EXPECT_TRUE(minimax_h3_head_layout_is_valid());
}

TEST(QualifiedTpContract, FrameworkMapsRanksToTpAndUlyssesCoordinates) {
  RankGenerator generator(16);
  std::vector<int32_t> sizes{2, 8, 1, 1};
  std::vector<std::string> order{"tp", "sp", "cfg", "dp"};
  const auto groups = generator.get_ranks_mapping(sizes, order);
  ASSERT_EQ(groups.at("tp").size(), 8);
  ASSERT_EQ(groups.at("sp").size(), 2);
  for (int32_t rank = 0; rank < 16; ++rank) {
    EXPECT_EQ(groups.at("tp")[rank / 2][rank % 2], rank);
    EXPECT_EQ(groups.at("sp")[rank % 2][rank / 2], rank);
  }
}

TEST(QualifiedTpContract, PaddingAndInverseTrimPreserveLogicalHeads) {
  const torch::Tensor heads =
      torch::arange(2 * 28 * 128, torch::kFloat32).reshape({2, 28, 128});
  const torch::Tensor padded = minimax_h3_pad_heads_to_physical(heads);
  ASSERT_EQ(padded.sizes().vec(), std::vector<int64_t>({2, 32, 128}));
  EXPECT_TRUE(minimax_h3_dummy_heads_are_zero(padded));
  EXPECT_TRUE(torch::equal(minimax_h3_trim_heads_to_logical(padded), heads));
  const torch::Tensor grouped =
      torch::arange(56 * 3 * 128, torch::kFloat32).reshape({56 * 3 * 128, 1});
  const torch::Tensor shard = minimax_h3_slice_grouped_qkv_weight(grouped, 1);
  EXPECT_TRUE(torch::equal(shard.slice(0, 0, 28 * 3 * 128),
                           grouped.slice(0, 28 * 3 * 128)));
  EXPECT_EQ(torch::count_nonzero(shard.slice(0, 28 * 3 * 128)).item<int64_t>(),
            0);
}

TEST(QualifiedTpContract, MergedFfnSlicesBothHalvesForEveryRank) {
  const torch::Tensor weight =
      torch::arange(16, torch::kFloat32).reshape({8, 2});
  EXPECT_TRUE(torch::equal(
      minimax_h3_slice_gate_up_weight(weight, 0),
      torch::tensor({{0.F, 1.F}, {2.F, 3.F}, {8.F, 9.F}, {10.F, 11.F}})));
  EXPECT_TRUE(torch::equal(
      minimax_h3_slice_gate_up_weight(weight, 1),
      torch::tensor({{4.F, 5.F}, {6.F, 7.F}, {12.F, 13.F}, {14.F, 15.F}})));
}

TEST(QualifiedSchedulerDiagnostics, CaptureKeepsDefaultResultsAndClock) {
  MiniMaxH3Scheduler original(/*shift=*/12.0);
  MiniMaxH3Scheduler observed(/*shift=*/12.0);
  original.set_timesteps(/*num_inference_steps=*/49,
                         torch::Device(torch::kCPU));
  observed.set_timesteps(/*num_inference_steps=*/49,
                         torch::Device(torch::kCPU));
  torch::Tensor left =
      torch::arange(8, torch::dtype(torch::kFloat32)).reshape({1, 2, 4});
  torch::Tensor right = left.clone();
  const torch::Tensor velocity = torch::full_like(left, 0.25);
  for (int64_t step = 0; step < 49; ++step) {
    torch::Tensor x0;
    left = original.step(velocity, original.timesteps()[step], left);
    right = observed.step(velocity, observed.timesteps()[step], right, &x0);
    ASSERT_TRUE(x0.defined());
    ASSERT_EQ(x0.scalar_type(), torch::kFloat32);
    ASSERT_EQ(x0.sizes(), right.sizes());
    ASSERT_TRUE(torch::equal(left, right)) << "step=" << step;
    ASSERT_TRUE(torch::equal(original.timesteps(), observed.timesteps()));
  }
  // At sigma_next=0, the actual scheduler result is its observed x0.
  original.set_timesteps(/*num_inference_steps=*/1, torch::Device(torch::kCPU));
  torch::Tensor x0;
  const torch::Tensor last =
      original.step(velocity, original.timesteps()[0], left, &x0);
  EXPECT_TRUE(torch::equal(last, x0));
}

}  // namespace
}  // namespace xllm

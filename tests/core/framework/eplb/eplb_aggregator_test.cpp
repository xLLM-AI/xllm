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

#include "core/framework/eplb/eplb_aggregator.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

namespace xllm {
namespace {

// Baseline: two devices, one layer, two experts per device. Each device sees
// cumulative counters where device 0 owns experts [0, 1] and device 1 owns
// experts [2, 3]. After a step the deltas are (5, 3) on device 0 and (2, 7)
// on device 1, which must scatter into experts [0..3] respectively.
TEST(EplbAggregatorTest, SingleLayerTwoDevicesScattersDeltasIntoGlobalLoad) {
  constexpr int32_t kLayerNum = 1;
  constexpr int32_t kDeviceNum = 2;
  constexpr int32_t kDeviceExpertsNum = 2;
  constexpr int32_t kNumExperts = 4;
  EplbAggregator aggregator(kLayerNum, kDeviceNum, kDeviceExpertsNum);

  torch::Tensor global_load =
      torch::zeros({kLayerNum, kNumExperts}, torch::kInt64);
  torch::Tensor expert_ids = torch::tensor({{{0, 1}, {2, 3}}}, torch::kInt32);
  std::vector<torch::Tensor> per_device{
      // cumulative counters at t=1: (5, 3) and (2, 7)
      torch::tensor({{5, 8}}, torch::kInt64),
      torch::tensor({{2, 9}}, torch::kInt64),
  };

  aggregator.aggregate(global_load, expert_ids, per_device);

  torch::Tensor expected = torch::tensor({{5, 3, 2, 7}}, torch::kInt64);
  EXPECT_TRUE(global_load.equal(expected))
      << "got=" << global_load << " expected=" << expected;
}

TEST(EplbAggregatorTest, PreservesMeasuredPhysicalSlotLoad) {
  constexpr int32_t kLayerNum = 1;
  constexpr int32_t kDeviceNum = 2;
  constexpr int32_t kDeviceExpertsNum = 2;
  constexpr int32_t kNumExperts = 4;
  EplbAggregator aggregator(kLayerNum, kDeviceNum, kDeviceExpertsNum);

  torch::Tensor global_load =
      torch::zeros({kLayerNum, kNumExperts}, torch::kInt64);
  torch::Tensor physical_load =
      torch::zeros({kLayerNum, kDeviceNum, kDeviceExpertsNum}, torch::kInt64);
  torch::Tensor expert_ids = torch::tensor({{{0, 1}, {2, 3}}}, torch::kInt32);
  std::vector<torch::Tensor> per_device{
      torch::tensor({{5, 8}}, torch::kInt64),
      torch::tensor({{2, 9}}, torch::kInt64),
  };

  aggregator.aggregate(global_load, physical_load, expert_ids, per_device);

  torch::Tensor expected_physical =
      torch::tensor({{{5, 3}, {2, 7}}}, torch::kInt64);
  EXPECT_TRUE(physical_load.equal(expected_physical))
      << "got=" << physical_load << " expected=" << expected_physical;
}

// A second aggregate() call on the same aggregator must accumulate on top of
// the previous global load — the aggregator itself is stateless but the
// caller-owned load tensor persists across rebalance rounds and idempotently
// records per-round deltas.
TEST(EplbAggregatorTest, AccumulatesAcrossCalls) {
  constexpr int32_t kLayerNum = 1;
  constexpr int32_t kDeviceNum = 2;
  constexpr int32_t kDeviceExpertsNum = 1;
  constexpr int32_t kNumExperts = 2;
  EplbAggregator aggregator(kLayerNum, kDeviceNum, kDeviceExpertsNum);

  torch::Tensor global_load =
      torch::zeros({kLayerNum, kNumExperts}, torch::kInt64);
  torch::Tensor expert_ids = torch::tensor({{{0}, {1}}}, torch::kInt32);
  std::vector<torch::Tensor> per_device_a{
      torch::tensor({{7}}, torch::kInt64),
      torch::tensor({{3}}, torch::kInt64),
  };
  aggregator.aggregate(global_load, expert_ids, per_device_a);

  std::vector<torch::Tensor> per_device_b{
      // Fresh cumulative counters again: previous round's delta was (7,3),
      // this round's delta is (1,5). Global load must land at (8, 8).
      torch::tensor({{1}}, torch::kInt64),
      torch::tensor({{5}}, torch::kInt64),
  };
  aggregator.aggregate(global_load, expert_ids, per_device_b);

  torch::Tensor expected = torch::tensor({{8, 8}}, torch::kInt64);
  EXPECT_TRUE(global_load.equal(expected))
      << "got=" << global_load << " expected=" << expected;
}

// Post-migration: the slot map on device 1 has swapped to hold expert 0 in
// its own slot 0, so the aggregator must credit device 1's delta to expert
// 0, not expert 1. This locks the semantics that expert_ids drives scatter,
// not physical slot position.
TEST(EplbAggregatorTest, HonorsPostMigrationExpertIdMap) {
  constexpr int32_t kLayerNum = 1;
  constexpr int32_t kDeviceNum = 2;
  constexpr int32_t kDeviceExpertsNum = 1;
  constexpr int32_t kNumExperts = 2;
  EplbAggregator aggregator(kLayerNum, kDeviceNum, kDeviceExpertsNum);

  torch::Tensor global_load =
      torch::zeros({kLayerNum, kNumExperts}, torch::kInt64);
  // Both devices now hold expert 0 (redundant expert). Nothing owns expert 1.
  torch::Tensor expert_ids = torch::tensor({{{0}, {0}}}, torch::kInt32);
  std::vector<torch::Tensor> per_device{
      torch::tensor({{4}}, torch::kInt64),
      torch::tensor({{9}}, torch::kInt64),
  };
  aggregator.aggregate(global_load, expert_ids, per_device);

  torch::Tensor expected = torch::tensor({{13, 0}}, torch::kInt64);
  EXPECT_TRUE(global_load.equal(expected))
      << "got=" << global_load << " expected=" << expected;
}

// Multi-layer sanity: the aggregator must process each layer independently
// and never mix its counters. Layer 0 sees (1, 2), layer 1 sees (10, 20).
TEST(EplbAggregatorTest, ProcessesLayersIndependently) {
  constexpr int32_t kLayerNum = 2;
  constexpr int32_t kDeviceNum = 2;
  constexpr int32_t kDeviceExpertsNum = 1;
  constexpr int32_t kNumExperts = 2;
  EplbAggregator aggregator(kLayerNum, kDeviceNum, kDeviceExpertsNum);

  torch::Tensor global_load =
      torch::zeros({kLayerNum, kNumExperts}, torch::kInt64);
  torch::Tensor expert_ids =
      torch::tensor({{{0}, {1}}, {{0}, {1}}}, torch::kInt32);
  std::vector<torch::Tensor> per_device{
      // shape [layer_num, device_experts_num]
      torch::tensor({{1}, {10}}, torch::kInt64),
      torch::tensor({{2}, {20}}, torch::kInt64),
  };
  aggregator.aggregate(global_load, expert_ids, per_device);

  torch::Tensor expected = torch::tensor({{1, 2}, {10, 20}}, torch::kInt64);
  EXPECT_TRUE(global_load.equal(expected))
      << "got=" << global_load << " expected=" << expected;
}

}  // namespace
}  // namespace xllm

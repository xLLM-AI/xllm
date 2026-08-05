/* Copyright 2025-2026 The xLLM Authors.

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

#include "eplb_policy.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <limits>
#include <set>
#include <vector>

#include "core/platform/platform.h"

namespace xllm {

namespace {

// Common shape used across the policy tests below. 16 experts on 4 devices,
// 5 physical slots per device (i.e. 4 primaries + 1 redundant) = 20 total
// slots, matching the pre-refactor Build test.
constexpr int32_t kDeviceExpertsNum = 5;
constexpr int32_t kDeviceNum = 4;
constexpr int32_t kLayerNum = 1;
constexpr int32_t kNumExperts = 16;

torch::Tensor make_skewed_load() {
  std::vector<torch::Tensor> tensors;
  tensors.reserve(1);
  tensors.emplace_back(torch::arange(0, kNumExperts));
  auto expert_load = torch::stack(tensors, 0);
  expert_load[0] =
      torch::tensor({100, 100, 100, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 100});
  return expert_load;
}

// Validates a rebalanced distribution against the shape and slot-cardinality
// invariants both policies must satisfy. Fails the current gtest if any
// invariant breaks; returns silently otherwise.
void expect_distribution_valid(const torch::Tensor& distribution) {
  ASSERT_EQ(distribution.dim(), 3);
  ASSERT_EQ(distribution.size(0), kLayerNum);
  ASSERT_EQ(distribution.size(1), kDeviceNum);
  ASSERT_EQ(distribution.size(2), kDeviceExpertsNum);
  auto flat = distribution[0].contiguous().view({-1}).to(torch::kInt32);
  const int32_t* data = flat.data_ptr<int32_t>();
  for (int64_t s = 0; s < flat.numel(); ++s) {
    ASSERT_GE(data[s], 0) << "Slot " << s
                          << " unassigned in rebalanced distribution.";
    ASSERT_LT(data[s], kNumExperts)
        << "Slot " << s << " points to invalid expert id " << data[s];
  }
  // Each origin expert must appear at least once — otherwise routing to that
  // expert has no target and the plan is unusable.
  std::set<int32_t> seen;
  for (int64_t s = 0; s < flat.numel(); ++s) {
    seen.insert(data[s]);
  }
  ASSERT_EQ(static_cast<int32_t>(seen.size()), kNumExperts);
}

}  // namespace

TEST(EplbPolicyTest, Build_Greedy) {
  torch::Device device(Platform::type_torch(), 0);
  GreedyEplbPolicy policy(
      kDeviceExpertsNum, kDeviceNum, kLayerNum, EplbOptions{});
  auto expert_load = make_skewed_load();
  auto [rebalance_expert, enable_update_vec] =
      policy.rebalance_experts(expert_load);
  expect_distribution_valid(rebalance_expert);
  ASSERT_EQ(static_cast<int32_t>(enable_update_vec.size()), kLayerNum);
  EXPECT_TRUE(enable_update_vec[0]);
}

TEST(EplbPolicyTest, Build_Balanced) {
  torch::Device device(Platform::type_torch(), 0);
  BalancedEplbPolicy policy(
      kDeviceExpertsNum, kDeviceNum, kLayerNum, EplbOptions{});
  auto expert_load = make_skewed_load();
  auto [rebalance_expert, enable_update_vec] =
      policy.rebalance_experts(expert_load);
  expect_distribution_valid(rebalance_expert);
  ASSERT_EQ(static_cast<int32_t>(enable_update_vec.size()), kLayerNum);
  EXPECT_TRUE(enable_update_vec[0]);

  // Balanced policy MUST have equal-cardinality per device: exactly
  // kDeviceExpertsNum slots on every device, filled.
  auto flat = rebalance_expert[0].contiguous().to(torch::kInt32);
  for (int32_t d = 0; d < kDeviceNum; ++d) {
    auto row = flat[d];
    for (int32_t s = 0; s < kDeviceExpertsNum; ++s) {
      EXPECT_GE(row[s].item<int32_t>(), 0)
          << "Balanced policy left slot [" << d << "][" << s << "] unassigned.";
    }
  }
}

TEST(EplbPolicyTest, BalancedSpreadsReplicasAcrossDevices) {
  EplbOptions options;
  options.eplb_min_peak_load_improvement = 0.0;
  BalancedEplbPolicy policy(kDeviceExpertsNum, kDeviceNum, kLayerNum, options);
  torch::Tensor expert_load =
      torch::ones({kLayerNum, kNumExperts}, torch::kInt64);
  expert_load[0][0] = 10000;

  auto [distribution, update] = policy.rebalance_experts(expert_load);

  ASSERT_TRUE(update[0]);
  expect_distribution_valid(distribution);
  const torch::Tensor placement =
      distribution[0].to(torch::kCPU).to(torch::kInt32).contiguous();
  for (int32_t device = 0; device < kDeviceNum; ++device) {
    std::set<int32_t> local_experts;
    for (int32_t slot = 0; slot < kDeviceExpertsNum; ++slot) {
      local_experts.insert(placement[device][slot].item<int32_t>());
    }
    EXPECT_EQ(static_cast<int32_t>(local_experts.size()), kDeviceExpertsNum)
        << "Device " << device
        << " received multiple replicas of the same logical expert.";
  }
}

TEST(EplbPolicyTest, BalancedFindsUniformReplicaPlacementWhenGreedyCanDeadEnd) {
  EplbOptions options;
  options.redundant_experts_num = 1;
  options.eplb_min_peak_load_improvement = 0.0;
  BalancedEplbPolicy policy(
      /*device_experts_num=*/2, /*device_num=*/3, /*layer_num=*/1, options);
  const torch::Tensor uniform_load =
      torch::tensor({{10, 10, 10}}, torch::kInt64);

  auto [distribution, update] = policy.rebalance_experts(uniform_load);

  ASSERT_TRUE(update[0]);
  const torch::Tensor placement =
      distribution[0].to(torch::kCPU).to(torch::kInt32).contiguous();
  std::vector<int32_t> replica_counts(3, 0);
  for (int32_t device = 0; device < 3; ++device) {
    std::set<int32_t> local_experts;
    for (int32_t slot = 0; slot < 2; ++slot) {
      const int32_t expert = placement[device][slot].item<int32_t>();
      ASSERT_GE(expert, 0);
      ASSERT_LT(expert, 3);
      local_experts.insert(expert);
      ++replica_counts[static_cast<size_t>(expert)];
    }
    EXPECT_EQ(local_experts.size(), 2U);
  }
  EXPECT_EQ(replica_counts, (std::vector<int32_t>{2, 2, 2}));
}

TEST(EplbPolicyTest, BalancedReplicaSelectionDoesNotOverflowAtLargeLoads) {
  EplbOptions options;
  options.redundant_experts_num = 1;
  options.eplb_min_peak_load_improvement = 0.0;
  BalancedEplbPolicy policy(
      /*device_experts_num=*/2, /*device_num=*/3, /*layer_num=*/1, options);
  const int64_t half_max = std::numeric_limits<int64_t>::max() / 2;
  const torch::Tensor large_load =
      torch::tensor(std::vector<int64_t>{half_max + 100, half_max, 1},
                    torch::TensorOptions().dtype(torch::kInt64))
          .reshape({1, 3});

  auto [distribution, update] = policy.rebalance_experts(large_load);

  ASSERT_TRUE(update[0]);
  const torch::Tensor flat =
      distribution[0].to(torch::kCPU).to(torch::kInt64).reshape({-1});
  EXPECT_EQ(torch::sum(flat == 0).item<int64_t>(), 3);
  EXPECT_EQ(torch::sum(flat == 1).item<int64_t>(), 2);
  EXPECT_EQ(torch::sum(flat == 2).item<int64_t>(), 1);
}

TEST(EplbPolicyTest, ZeroLoadPublishesOnceForInitialPlacement) {
  torch::Tensor zero_load =
      torch::zeros({kLayerNum, kNumExperts}, torch::kInt64);

  GreedyEplbPolicy greedy(
      kDeviceExpertsNum, kDeviceNum, kLayerNum, EplbOptions{});
  auto [greedy_first_distribution, greedy_first_update] =
      greedy.rebalance_experts(zero_load);
  EXPECT_TRUE(greedy_first_update[0]);
  expect_distribution_valid(greedy_first_distribution);
  auto [greedy_second_distribution, greedy_second_update] =
      greedy.rebalance_experts(zero_load);
  EXPECT_FALSE(greedy_second_update[0]);
  EXPECT_TRUE(
      torch::equal(greedy_first_distribution, greedy_second_distribution));

  BalancedEplbPolicy balanced(
      kDeviceExpertsNum, kDeviceNum, kLayerNum, EplbOptions{});
  auto [balanced_first_distribution, balanced_first_update] =
      balanced.rebalance_experts(zero_load);
  EXPECT_TRUE(balanced_first_update[0]);
  expect_distribution_valid(balanced_first_distribution);
  auto [balanced_second_distribution, balanced_second_update] =
      balanced.rebalance_experts(zero_load);
  EXPECT_FALSE(balanced_second_update[0]);
  EXPECT_TRUE(
      torch::equal(balanced_first_distribution, balanced_second_distribution));
}

TEST(EplbPolicyTest, GreedyAbortRetriesUncommittedPlacement) {
  EplbOptions options;
  GreedyEplbPolicy policy(kDeviceExpertsNum, kDeviceNum, kLayerNum, options);
  torch::Tensor load = torch::arange(
      1, kNumExperts + 1, torch::TensorOptions().dtype(torch::kInt64));
  load = load.reshape({kLayerNum, kNumExperts});

  auto [first_distribution, first_update] = policy.rebalance_experts(load);
  ASSERT_TRUE(first_update[0]);
  policy.abort_layer(/*layer_id=*/0);

  auto [retry_distribution, retry_update] = policy.rebalance_experts(load);
  EXPECT_TRUE(retry_update[0]);
  EXPECT_TRUE(torch::equal(retry_distribution, first_distribution));
}

TEST(EplbPolicyTest, BalancedAbortRetriesUncommittedPlacement) {
  EplbOptions options;
  BalancedEplbPolicy policy(kDeviceExpertsNum, kDeviceNum, kLayerNum, options);
  torch::Tensor load = torch::arange(
      1, kNumExperts + 1, torch::TensorOptions().dtype(torch::kInt64));
  load = load.reshape({kLayerNum, kNumExperts});

  auto [first_distribution, first_update] = policy.rebalance_experts(load);
  ASSERT_TRUE(first_update[0]);
  policy.abort_layer(/*layer_id=*/0);

  auto [retry_distribution, retry_update] = policy.rebalance_experts(load);
  EXPECT_TRUE(retry_update[0]);
  EXPECT_TRUE(torch::equal(retry_distribution, first_distribution));
}

TEST(EplbPolicyTest, Factory_KindString) {
  EXPECT_EQ(eplb_policy_kind_from_string("greedy"), EplbPolicyKind::GREEDY);
  EXPECT_EQ(eplb_policy_kind_from_string("GREEDY"), EplbPolicyKind::GREEDY);
  EXPECT_EQ(eplb_policy_kind_from_string("balanced"), EplbPolicyKind::BALANCED);
  EXPECT_EQ(eplb_policy_kind_from_string("deepseek_flat"),
            EplbPolicyKind::BALANCED);
  EXPECT_EQ(eplb_policy_kind_from_string("DeepSeek_Flat"),
            EplbPolicyKind::BALANCED);
  EXPECT_EQ(eplb_policy_kind_from_string("flat"), EplbPolicyKind::BALANCED);
  // Unknown values fall back to GREEDY instead of throwing, so bad flag
  // strings do not knock the rebalance loop offline.
  EXPECT_EQ(eplb_policy_kind_from_string("bogus"), EplbPolicyKind::GREEDY);
  EXPECT_EQ(eplb_policy_kind_from_string(""), EplbPolicyKind::GREEDY);
}

TEST(EplbPolicyTest, Factory_MakeEplbPolicy) {
  const EplbOptions options;
  auto greedy = MakeEplbPolicy(EplbPolicyKind::GREEDY,
                               kDeviceExpertsNum,
                               kDeviceNum,
                               kLayerNum,
                               options);
  auto balanced = MakeEplbPolicy(EplbPolicyKind::BALANCED,
                                 kDeviceExpertsNum,
                                 kDeviceNum,
                                 kLayerNum,
                                 options);
  ASSERT_NE(greedy, nullptr);
  ASSERT_NE(balanced, nullptr);
  EXPECT_NE(dynamic_cast<GreedyEplbPolicy*>(greedy.get()), nullptr);
  EXPECT_NE(dynamic_cast<BalancedEplbPolicy*>(balanced.get()), nullptr);
}

TEST(EplbPolicyTest, LegacyFacadeStartsFromManagerIdentityPlacement) {
  EplbPolicy policy(kDeviceExpertsNum, kDeviceNum, kLayerNum);
  torch::Tensor uniform =
      torch::full({kLayerNum, kNumExperts}, 10, torch::kInt64);

  auto [distribution, update] = policy.rebalance_experts(uniform);

  EXPECT_FALSE(update[0]);
  EXPECT_EQ(distribution.size(0), kLayerNum);
  EXPECT_EQ(distribution.size(1), kDeviceNum);
  EXPECT_EQ(distribution.size(2), kDeviceExpertsNum);
}

TEST(EplbPolicyTest, Factory_LegacyNamesAliasToBalanced) {
  EXPECT_EQ(eplb_policy_kind_from_string("deepseek_hier"),
            EplbPolicyKind::BALANCED);
  EXPECT_EQ(eplb_policy_kind_from_string("DeepSeek_Hier"),
            EplbPolicyKind::BALANCED);
  EXPECT_EQ(eplb_policy_kind_from_string("hier"), EplbPolicyKind::BALANCED);
  EXPECT_EQ(eplb_policy_kind_from_string("hierarchical"),
            EplbPolicyKind::BALANCED);
}

TEST(EplbPolicyTest, SkipsUniformWorkloadRepackWithNoPeakLoadGain) {
  EplbOptions options;
  options.eplb_min_peak_load_improvement = 0.05;
  BalancedEplbPolicy policy(kDeviceExpertsNum, kDeviceNum, kLayerNum, options);

  torch::Tensor skewed = torch::tensor(
      {{50, 50, 50, 50, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5}}, torch::kInt64);
  auto [skewed_distribution, skewed_update] = policy.rebalance_experts(skewed);
  ASSERT_TRUE(skewed_update[0]);

  torch::Tensor uniform =
      torch::full({kLayerNum, kNumExperts}, 10, torch::kInt64);
  auto [uniform_distribution, uniform_update] =
      policy.rebalance_experts(uniform);

  EXPECT_FALSE(uniform_update[0])
      << "uniform expert load cannot reduce peak rank load by repacking";
  EXPECT_TRUE(torch::equal(uniform_distribution, skewed_distribution));
}

TEST(EplbPolicyTest, SeededInitialDistributionSkipsNoGainFirstRound) {
  EplbOptions options;
  options.eplb_min_peak_load_improvement = 0.05;
  BalancedEplbPolicy policy(kDeviceExpertsNum, kDeviceNum, kLayerNum, options);
  torch::Tensor initial_distribution = torch::tensor({{{0, 1, 2, 3, 3},
                                                       {4, 5, 6, 7, 7},
                                                       {8, 9, 10, 11, 11},
                                                       {12, 13, 14, 15, 15}}},
                                                     torch::kInt32);
  policy.initialize_distribution(initial_distribution);

  torch::Tensor uniform =
      torch::full({kLayerNum, kNumExperts}, 10, torch::kInt64);
  auto [distribution, update] = policy.rebalance_experts(uniform);

  EXPECT_FALSE(update[0]);
  EXPECT_TRUE(torch::equal(distribution, initial_distribution));
}

TEST(EplbPolicyTest, BalancedPreservesSlotsWhenRankMembershipUnchanged) {
  EplbOptions options;
  options.redundant_experts_num = 0;
  BalancedEplbPolicy policy(
      /*device_experts_num=*/2, /*device_num=*/2, /*layer_num=*/1, options);
  const torch::Tensor initial_distribution =
      torch::tensor({{{3, 0}, {2, 1}}}, torch::kInt32);
  policy.initialize_distribution(initial_distribution);

  const torch::Tensor logical_load =
      torch::tensor({{40, 30, 20, 10}}, torch::kInt64);
  const torch::Tensor physical_load =
      torch::tensor({{{100, 100}, {1, 1}}}, torch::kInt64);
  auto [distribution, update] =
      policy.rebalance_experts(logical_load, physical_load);

  EXPECT_FALSE(update[0]);
  EXPECT_TRUE(torch::equal(distribution, initial_distribution));
}

TEST(EplbPolicyTest, BalancedFillsOnlySlotsVacatedByRankMigrations) {
  EplbOptions options;
  options.redundant_experts_num = 1;
  BalancedEplbPolicy policy(
      /*device_experts_num=*/3, /*device_num=*/2, /*layer_num=*/1, options);
  const torch::Tensor initial_distribution =
      torch::tensor({{{0, 1, 3}, {1, 2, 0}}}, torch::kInt32);
  policy.initialize_distribution(initial_distribution);

  const torch::Tensor logical_load =
      torch::tensor({{100, 90, 80, 1}}, torch::kInt64);
  const torch::Tensor physical_load =
      torch::tensor({{{100, 100, 100}, {1, 1, 1}}}, torch::kInt64);
  auto [distribution, update] =
      policy.rebalance_experts(logical_load, physical_load);

  const torch::Tensor expected_distribution =
      torch::tensor({{{0, 1, 2}, {1, 3, 0}}}, torch::kInt32);
  EXPECT_TRUE(update[0]);
  EXPECT_TRUE(torch::equal(distribution, expected_distribution));
}

TEST(EplbPolicyTest, PublishesPlacementWithSufficientPeakLoadGain) {
  EplbOptions options;
  options.eplb_min_peak_load_improvement = 0.05;
  BalancedEplbPolicy policy(kDeviceExpertsNum, kDeviceNum, kLayerNum, options);

  torch::Tensor first = torch::tensor(
      {{100, 100, 100, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5}}, torch::kInt64);
  auto [first_distribution, first_update] = policy.rebalance_experts(first);
  ASSERT_TRUE(first_update[0]);

  torch::Tensor second = torch::tensor(
      {{5, 5, 5, 5, 5, 100, 100, 100, 5, 5, 5, 5, 5, 5, 5, 5}}, torch::kInt64);
  auto [second_distribution, second_update] = policy.rebalance_experts(second);

  EXPECT_TRUE(second_update[0]);
  EXPECT_FALSE(torch::equal(second_distribution, first_distribution));
}

TEST(EplbPolicyTest, BalancedUsesMeasuredPhysicalPeakForBenefitGate) {
  EplbOptions options;
  options.eplb_min_peak_load_improvement = 0.05;
  BalancedEplbPolicy policy(kDeviceExpertsNum, kDeviceNum, kLayerNum, options);

  torch::Tensor active_distribution = torch::tensor({{{0, 1, 2, 3, 3},
                                                      {4, 5, 6, 7, 7},
                                                      {8, 9, 10, 11, 11},
                                                      {12, 13, 14, 15, 15}}},
                                                    torch::kInt32);
  policy.initialize_distribution(active_distribution);

  // Logical history can span more than one placement window. The current
  // physical window is already balanced, so a candidate based only on logical
  // replica counts must not trigger a migration.
  torch::Tensor logical_load =
      torch::tensor({{100, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100}},
                    torch::kInt64);
  torch::Tensor physical_load = torch::tensor({{{20, 20, 20, 20, 20},
                                                {20, 20, 20, 20, 20},
                                                {20, 20, 20, 20, 20},
                                                {20, 20, 20, 20, 20}}},
                                              torch::kInt64);

  auto [distribution, update] =
      policy.rebalance_experts(logical_load, physical_load);

  EXPECT_FALSE(update[0]);
  EXPECT_TRUE(torch::equal(distribution, active_distribution));
}

TEST(EplbPolicyTest, RecomputesWhenOnlyMeasuredPhysicalPeakChanges) {
  EplbOptions options;
  options.eplb_min_peak_load_improvement = 0.05;
  BalancedEplbPolicy policy(kDeviceExpertsNum, kDeviceNum, kLayerNum, options);

  const torch::Tensor active_distribution =
      torch::tensor({{{0, 1, 2, 3, 3},
                      {4, 5, 6, 7, 7},
                      {8, 9, 10, 11, 11},
                      {12, 13, 14, 15, 15}}},
                    torch::kInt32);
  policy.initialize_distribution(active_distribution);

  const torch::Tensor logical_load =
      torch::tensor({{100, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100}},
                    torch::kInt64);
  const torch::Tensor balanced_physical_load = torch::full(
      {kLayerNum, kDeviceNum, kDeviceExpertsNum}, 20, torch::kInt64);
  auto [balanced_distribution, balanced_update] =
      policy.rebalance_experts(logical_load, balanced_physical_load);
  ASSERT_FALSE(balanced_update[0]);
  ASSERT_TRUE(torch::equal(balanced_distribution, active_distribution));

  const torch::Tensor skewed_physical_load =
      torch::tensor({{{100, 100, 100, 100, 100},
                      {1, 1, 1, 1, 1},
                      {1, 1, 1, 1, 1},
                      {1, 1, 1, 1, 1}}},
                    torch::kInt64);
  auto [rebalanced_distribution, rebalanced_update] =
      policy.rebalance_experts(logical_load, skewed_physical_load);

  EXPECT_TRUE(rebalanced_update[0])
      << "each interval must evaluate the current physical rank peak";
  EXPECT_FALSE(torch::equal(rebalanced_distribution, active_distribution));
}

TEST(EplbPolicyTest, BalancedBenefitGateIsInvariantToLogicalLoadScale) {
  EplbOptions options;
  options.eplb_min_peak_load_improvement = 0.05;
  BalancedEplbPolicy baseline_policy(
      kDeviceExpertsNum, kDeviceNum, kLayerNum, options);
  BalancedEplbPolicy scaled_policy(
      kDeviceExpertsNum, kDeviceNum, kLayerNum, options);
  const torch::Tensor active_distribution =
      torch::tensor({{{0, 1, 2, 3, 3},
                      {4, 5, 6, 7, 7},
                      {8, 9, 10, 11, 11},
                      {12, 13, 14, 15, 15}}},
                    torch::kInt32);
  baseline_policy.initialize_distribution(active_distribution);
  scaled_policy.initialize_distribution(active_distribution);

  const torch::Tensor logical_load = torch::tensor(
      {{100, 100, 100, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5}}, torch::kInt64);
  const torch::Tensor physical_load = torch::tensor({{{100, 100, 100, 5, 5},
                                                      {5, 5, 5, 5, 5},
                                                      {5, 5, 5, 5, 5},
                                                      {5, 5, 5, 5, 5}}},
                                                    torch::kInt64);

  auto [baseline_distribution, baseline_update] =
      baseline_policy.rebalance_experts(logical_load, physical_load);
  auto [scaled_distribution, scaled_update] =
      scaled_policy.rebalance_experts(logical_load * 10, physical_load);

  ASSERT_TRUE(baseline_update[0]);
  EXPECT_EQ(scaled_update, baseline_update);
  EXPECT_TRUE(torch::equal(scaled_distribution, baseline_distribution));
}

TEST(EplbPolicyTest, GreedyRequiresMinimumPeakLoadImprovement) {
  EplbOptions unguarded_options;
  unguarded_options.eplb_min_peak_load_improvement = 0.0;
  EplbOptions guarded_options;
  guarded_options.eplb_min_peak_load_improvement = 1.0;
  GreedyEplbPolicy unguarded_policy(
      kDeviceExpertsNum, kDeviceNum, kLayerNum, unguarded_options);
  GreedyEplbPolicy guarded_policy(
      kDeviceExpertsNum, kDeviceNum, kLayerNum, guarded_options);
  const torch::Tensor active_distribution =
      torch::tensor({{{0, 1, 2, 3, 3},
                      {4, 5, 6, 7, 7},
                      {8, 9, 10, 11, 11},
                      {12, 13, 14, 15, 15}}},
                    torch::kInt32);
  unguarded_policy.initialize_distribution(active_distribution);
  guarded_policy.initialize_distribution(active_distribution);

  const torch::Tensor logical_load =
      torch::tensor({{100, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100}},
                    torch::kInt64);
  const torch::Tensor physical_load = torch::tensor({{{100, 100, 100, 100, 100},
                                                      {1, 1, 1, 1, 1},
                                                      {1, 1, 1, 1, 1},
                                                      {1, 1, 1, 1, 1}}},
                                                    torch::kInt64);
  auto [unguarded_distribution, unguarded_update] =
      unguarded_policy.rebalance_experts(logical_load, physical_load);
  auto [guarded_distribution, guarded_update] =
      guarded_policy.rebalance_experts(logical_load, physical_load);

  ASSERT_TRUE(unguarded_update[0]);
  EXPECT_FALSE(torch::equal(unguarded_distribution, active_distribution));
  EXPECT_FALSE(guarded_update[0]);
  EXPECT_TRUE(torch::equal(guarded_distribution, active_distribution));
}

}  // namespace xllm

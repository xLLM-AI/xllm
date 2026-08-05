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

#include "core/framework/eplb/eplb_manager.h"

#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace xllm {
namespace {

using namespace std::chrono_literals;

class RecordingPolicy final : public IEplbPolicy {
 public:
  explicit RecordingPolicy(bool update_every_call = false)
      : update_every_call_(update_every_call) {}

  std::pair<torch::Tensor, std::vector<bool>> rebalance_experts(
      torch::Tensor expert_load,
      torch::Tensor physical_expert_load = torch::Tensor()) override {
    (void)physical_expert_load;
    std::lock_guard<std::mutex> lock(mutex_);
    recorded_loads_.push_back(expert_load.clone());
    recorded_physical_loads_.push_back(physical_expert_load.clone());
    ++call_count_;
    condition_.notify_all();
    const int64_t layer_count = expert_load.size(0);
    torch::Tensor distribution =
        torch::zeros({layer_count, 2, 1}, torch::kInt32);
    distribution.select(/*dim=*/1, /*index=*/0).fill_(1);
    if (call_count_ == 1 || update_every_call_) {
      return {distribution,
              std::vector<bool>(static_cast<size_t>(layer_count), true)};
    }
    return {distribution,
            std::vector<bool>(static_cast<size_t>(layer_count), false)};
  }

  void initialize_distribution(
      const torch::Tensor& current_distribution) override {
    std::lock_guard<std::mutex> lock(mutex_);
    initialized_distribution_ = current_distribution.clone();
  }

  void abort_layer(int32_t layer_id) override {
    std::lock_guard<std::mutex> lock(mutex_);
    aborted_layers_.emplace_back(layer_id);
    condition_.notify_all();
  }

  std::string name() const override { return "recording"; }

  bool wait_for_calls(int32_t expected_calls,
                      std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return condition_.wait_for(
        lock, timeout, [&] { return call_count_ >= expected_calls; });
  }

  bool wait_for_aborts(int32_t expected_aborts,
                       std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return condition_.wait_for(lock, timeout, [&] {
      return static_cast<int32_t>(aborted_layers_.size()) >= expected_aborts;
    });
  }

  torch::Tensor recorded_load(int32_t call_index) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return recorded_loads_.at(static_cast<size_t>(call_index)).clone();
  }

  torch::Tensor initialized_distribution() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return initialized_distribution_.clone();
  }

  torch::Tensor recorded_physical_load(int32_t call_index) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return recorded_physical_loads_.at(static_cast<size_t>(call_index)).clone();
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  int32_t call_count_ = 0;
  std::vector<torch::Tensor> recorded_loads_;
  std::vector<torch::Tensor> recorded_physical_loads_;
  std::vector<int32_t> aborted_layers_;
  torch::Tensor initialized_distribution_;
  bool update_every_call_ = false;
};

EplbOptions make_test_options() {
  EplbOptions options;
  options.redundant_experts_num = 0;
  options.eplb_update_interval = 0;
  options.eplb_prepare_timeout_seconds = 1;
  return options;
}

EplbInfo wait_for_prepare(EplbManager& manager) {
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (std::chrono::steady_clock::now() < deadline) {
    EplbInfo info = manager.get_eplb_info();
    if (info.prepare_layer_id != -1) {
      return info;
    }
    std::this_thread::sleep_for(1ms);
  }
  return EplbInfo{};
}

EplbInfo wait_for_update(EplbManager& manager) {
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (std::chrono::steady_clock::now() < deadline) {
    EplbInfo info = manager.get_eplb_info();
    if (info.update_layer_id != -1) {
      return info;
    }
    std::this_thread::sleep_for(1ms);
  }
  return EplbInfo{};
}

TEST(EplbManagerTest, SeedsPolicyWithActiveDistribution) {
  auto policy = std::make_unique<RecordingPolicy>();
  RecordingPolicy* policy_observer = policy.get();
  EplbManager manager(/*layer_num=*/1,
                      /*device_num=*/2,
                      /*experts_num=*/2,
                      make_test_options(),
                      std::move(policy));

  EXPECT_TRUE(torch::equal(policy_observer->initialized_distribution(),
                           torch::tensor({{{0}, {1}}}, torch::kInt32)));
}

TEST(EplbManagerTest, RebalanceTimerProcessesSamplesWithoutFollowupTraffic) {
  EplbOptions options = make_test_options();
  options.eplb_update_interval = 1;
  auto policy = std::make_unique<RecordingPolicy>();
  RecordingPolicy* policy_observer = policy.get();
  EplbManager manager(/*layer_num=*/1,
                      /*device_num=*/2,
                      /*experts_num=*/2,
                      options,
                      std::move(policy));

  std::this_thread::sleep_for(1200ms);
  manager.update_expert_load({torch::tensor({{10}}, torch::kInt64),
                              torch::tensor({{0}}, torch::kInt64)});

  EXPECT_FALSE(policy_observer->wait_for_calls(/*expected_calls=*/1, 100ms));
  EXPECT_TRUE(policy_observer->wait_for_calls(/*expected_calls=*/1, 2s));
}

TEST(EplbManagerTest, QueuedLoadUsesPlacementActiveAtSubmission) {
  auto policy = std::make_unique<RecordingPolicy>();
  RecordingPolicy* policy_observer = policy.get();
  EplbManager manager(/*layer_num=*/1,
                      /*device_num=*/2,
                      /*experts_num=*/2,
                      make_test_options(),
                      std::move(policy));

  manager.update_expert_load({torch::tensor({{10}}, torch::kInt64),
                              torch::tensor({{0}}, torch::kInt64)});
  ASSERT_TRUE(policy_observer->wait_for_calls(/*expected_calls=*/1, 2s));
  EXPECT_TRUE(
      torch::equal(policy_observer->recorded_physical_load(/*call_index=*/0),
                   torch::tensor({{{10}, {0}}}, torch::kInt64)));

  EplbInfo prepare = wait_for_prepare(manager);
  ASSERT_EQ(prepare.prepare_layer_id, 0);
  EXPECT_EQ(prepare.expert_ids, (std::vector<int32_t>{1, 0}));

  manager.update_expert_load({torch::tensor({{3}}, torch::kInt64),
                              torch::tensor({{7}}, torch::kInt64)});
  manager.set_prepared_tokens({prepare.prepare_token, prepare.prepare_token});
  EplbInfo update = wait_for_update(manager);
  ASSERT_EQ(update.update_layer_id, 0);
  ASSERT_GT(update.activation_token, 0);

  manager.update_expert_load({torch::tensor({{5}}, torch::kInt64),
                              torch::tensor({{11}}, torch::kInt64)},
                             /*completed_activation_token=*/-1);
  EXPECT_FALSE(policy_observer->wait_for_calls(/*expected_calls=*/2, 100ms))
      << "An older overlap output must not acknowledge the activation command.";

  manager.update_expert_load({torch::tensor({{1}}, torch::kInt64),
                              torch::tensor({{1}}, torch::kInt64)},
                             /*completed_activation_token=*/-1);
  EXPECT_FALSE(policy_observer->wait_for_calls(/*expected_calls=*/2, 100ms))
      << "The manager must keep waiting until the command's output arrives.";

  manager.update_expert_load({torch::tensor({{0}}, torch::kInt64),
                              torch::tensor({{0}}, torch::kInt64)},
                             update.activation_token);
  EXPECT_FALSE(policy_observer->wait_for_calls(/*expected_calls=*/2, 100ms))
      << "The activation-carrying sample still belongs to the old placement.";

  manager.update_expert_load({torch::tensor({{2}}, torch::kInt64),
                              torch::tensor({{4}}, torch::kInt64)},
                             /*completed_activation_token=*/-1);
  ASSERT_TRUE(policy_observer->wait_for_calls(/*expected_calls=*/2, 2s));
  EXPECT_TRUE(torch::equal(policy_observer->recorded_load(/*call_index=*/1),
                           torch::tensor({{4, 2}}, torch::kInt64)))
      << "A new generation must contain only load observed after activation.";
  EXPECT_TRUE(
      torch::equal(policy_observer->recorded_physical_load(/*call_index=*/1),
                   torch::tensor({{{2}, {4}}}, torch::kInt64)))
      << "The next policy round must use physical load observed under the new "
         "placement.";
}

TEST(EplbManagerTest, PrepareTimeoutSkipsStalledLayer) {
  auto policy = std::make_unique<RecordingPolicy>(/*update_every_call=*/true);
  RecordingPolicy* policy_observer = policy.get();
  EplbManager manager(/*layer_num=*/1,
                      /*device_num=*/2,
                      /*experts_num=*/2,
                      make_test_options(),
                      std::move(policy));

  manager.update_expert_load({torch::tensor({{10}}, torch::kInt64),
                              torch::tensor({{0}}, torch::kInt64)});
  ASSERT_TRUE(policy_observer->wait_for_calls(/*expected_calls=*/1, 2s));
  EplbInfo first_prepare = wait_for_prepare(manager);
  ASSERT_EQ(first_prepare.prepare_layer_id, 0);
  manager.set_prepared_tokens({-1, -1});

  const auto deadline = std::chrono::steady_clock::now() + 3s;
  while (!policy_observer->wait_for_calls(/*expected_calls=*/2, 50ms) &&
         std::chrono::steady_clock::now() < deadline) {
    manager.update_expert_load({torch::tensor({{0}}, torch::kInt64),
                                torch::tensor({{0}}, torch::kInt64)});
  }
  ASSERT_TRUE(policy_observer->wait_for_calls(/*expected_calls=*/2, 50ms));

  EplbInfo second_prepare = wait_for_prepare(manager);
  ASSERT_EQ(second_prepare.prepare_layer_id, 0);
  ASSERT_NE(second_prepare.prepare_token, first_prepare.prepare_token);
  manager.set_prepared_tokens(
      {first_prepare.prepare_token, first_prepare.prepare_token});
  std::this_thread::sleep_for(50ms);
  EXPECT_EQ(manager.get_eplb_info().update_layer_id, -1);

  manager.set_prepared_tokens(
      {second_prepare.prepare_token, second_prepare.prepare_token});
  EXPECT_EQ(wait_for_update(manager).update_layer_id, 0);
}

TEST(EplbManagerTest, PrepareTimeoutStartsAfterWorkerOutputIsObservable) {
  auto policy = std::make_unique<RecordingPolicy>();
  RecordingPolicy* policy_observer = policy.get();
  EplbManager manager(/*layer_num=*/1,
                      /*device_num=*/2,
                      /*experts_num=*/2,
                      make_test_options(),
                      std::move(policy));

  manager.update_expert_load({torch::tensor({{10}}, torch::kInt64),
                              torch::tensor({{0}}, torch::kInt64)});
  ASSERT_TRUE(policy_observer->wait_for_calls(/*expected_calls=*/1, 2s));
  EplbInfo prepare = wait_for_prepare(manager);
  ASSERT_EQ(prepare.prepare_layer_id, 0);

  std::this_thread::sleep_for(1200ms);
  manager.set_prepared_tokens({prepare.prepare_token, prepare.prepare_token});
  EXPECT_EQ(wait_for_update(manager).update_layer_id, 0);
}

TEST(EplbManagerTest, PrepareDispatchWithoutWorkerOutputEventuallyAborts) {
  auto policy = std::make_unique<RecordingPolicy>();
  RecordingPolicy* policy_observer = policy.get();
  EplbManager manager(/*layer_num=*/1,
                      /*device_num=*/2,
                      /*experts_num=*/2,
                      make_test_options(),
                      std::move(policy));

  manager.update_expert_load({torch::tensor({{10}}, torch::kInt64),
                              torch::tensor({{0}}, torch::kInt64)});
  ASSERT_TRUE(policy_observer->wait_for_calls(/*expected_calls=*/1, 2s));
  ASSERT_EQ(wait_for_prepare(manager).prepare_layer_id, 0);

  EXPECT_TRUE(policy_observer->wait_for_aborts(/*expected_aborts=*/1, 3s));
}

TEST(EplbManagerTest, PublishesNextPrepareWithPendingActivation) {
  auto policy = std::make_unique<RecordingPolicy>();
  RecordingPolicy* policy_observer = policy.get();
  EplbManager manager(/*layer_num=*/2,
                      /*device_num=*/2,
                      /*experts_num=*/2,
                      make_test_options(),
                      std::move(policy));

  manager.update_expert_load({torch::tensor({{10}, {10}}, torch::kInt64),
                              torch::tensor({{0}, {0}}, torch::kInt64)});
  ASSERT_TRUE(policy_observer->wait_for_calls(/*expected_calls=*/1, 2s));

  EplbInfo first_prepare = wait_for_prepare(manager);
  ASSERT_EQ(first_prepare.prepare_layer_id, 0);
  manager.set_prepared_tokens(
      {first_prepare.prepare_token, first_prepare.prepare_token});

  EplbInfo first_update = wait_for_update(manager);
  ASSERT_EQ(first_update.update_layer_id, 0);
  EXPECT_EQ(first_update.prepare_layer_id, 1);
  EXPECT_GT(first_update.prepare_token, 0);
}

TEST(EplbManagerTest, DefersPendingActivationUntilWeightUpdateIsAllowed) {
  auto policy = std::make_unique<RecordingPolicy>();
  RecordingPolicy* policy_observer = policy.get();
  EplbManager manager(/*layer_num=*/1,
                      /*device_num=*/2,
                      /*experts_num=*/2,
                      make_test_options(),
                      std::move(policy));

  manager.update_expert_load({torch::tensor({{10}}, torch::kInt64),
                              torch::tensor({{0}}, torch::kInt64)});
  ASSERT_TRUE(policy_observer->wait_for_calls(/*expected_calls=*/1, 2s));
  EplbInfo prepare = wait_for_prepare(manager);
  ASSERT_EQ(prepare.prepare_layer_id, 0);
  manager.set_prepared_tokens({prepare.prepare_token, prepare.prepare_token});
  std::this_thread::sleep_for(50ms);

  EplbInfo deferred = manager.get_eplb_info(/*allow_weight_update=*/false);
  EXPECT_EQ(deferred.update_layer_id, -1);
  EXPECT_EQ(deferred.prepare_layer_id, -1);
  EXPECT_EQ(wait_for_update(manager).update_layer_id, 0);
}

}  // namespace
}  // namespace xllm

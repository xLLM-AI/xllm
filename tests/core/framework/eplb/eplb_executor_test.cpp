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

#include "core/framework/eplb/eplb_executor.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace xllm {
namespace {

static_assert(std::is_same_v<decltype(EplbInfo::prepare_token), int64_t>);
static_assert(std::is_same_v<decltype(EplbInfo::activation_token), int64_t>);

class RecordingCausalLM final : public CausalLM {
 public:
  explicit RecordingCausalLM(const torch::Device& device)
      : device_(device), options_(torch::TensorOptions().device(device)) {}

  ModelOutput forward(const torch::Tensor&,
                      const torch::Tensor&,
                      std::vector<KVCache>&,
                      const ModelInputParams&) override {
    return ModelOutput();
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor&) override {
    return hidden_states;
  }

  void load_model(std::unique_ptr<ModelLoader>) override {}

  torch::Device device() const override { return device_; }

  void prepare_expert_weight(int32_t layer_id,
                             const std::vector<int32_t>&) override {
    std::lock_guard<std::mutex> lock(events_mutex_);
    events_.push_back("prepare:" + std::to_string(layer_id));
  }

  void start_expert_weight_transfer(int32_t layer_id) override {
    std::lock_guard<std::mutex> lock(events_mutex_);
    events_.push_back("start:" + std::to_string(layer_id));
  }

  void update_expert_weight(int32_t layer_id) override {
    std::lock_guard<std::mutex> lock(events_mutex_);
    events_.push_back("finish:" + std::to_string(layer_id));
  }

  const torch::TensorOptions& options() const override { return options_; }

  std::vector<std::string> events() const {
    std::lock_guard<std::mutex> lock(events_mutex_);
    return events_;
  }

 private:
  torch::Device device_;
  torch::TensorOptions options_;
  mutable std::mutex events_mutex_;
  std::vector<std::string> events_;
};

TEST(EplbExecutorTest, DefersActivationUntilForwardEnd) {
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  RecordingCausalLM model(device.unwrap());
  EplbExecutor executor(model, device.unwrap());
  EplbInfo info;
  info.update_layer_id = 7;

  executor.start_eplb_step(info);

  const std::vector<std::string> expected_before = {"start:7"};
  EXPECT_EQ(model.events(), expected_before);

  executor.finish_eplb_step();

  const std::vector<std::string> expected_after = {"start:7", "finish:7"};
  EXPECT_EQ(model.events(), expected_after);
}

TEST(EplbExecutorTest, DefersNextPrepareUntilActivationFinishes) {
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  RecordingCausalLM model(device.unwrap());
  EplbExecutor executor(model, device.unwrap());
  EplbInfo info;
  info.update_layer_id = 7;
  info.prepare_layer_id = 8;
  info.prepare_token = 123;
  info.expert_ids = {0, 1};

  executor.start_eplb_step(info);

  const auto prepare_deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
  while (std::chrono::steady_clock::now() < prepare_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_EQ(executor.consume_ready_prepare_token(), -1);
  EXPECT_EQ(model.events(), (std::vector<std::string>{"start:7"}));

  executor.finish_eplb_step();

  const auto finish_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  int64_t ready_prepare_token = -1;
  while (ready_prepare_token == -1 &&
         std::chrono::steady_clock::now() < finish_deadline) {
    ready_prepare_token = executor.consume_ready_prepare_token();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_EQ(ready_prepare_token, info.prepare_token);
  EXPECT_EQ(executor.consume_ready_prepare_token(), -1);
  EXPECT_EQ(model.events(),
            (std::vector<std::string>{"start:7", "finish:7", "prepare:8"}));
}

}  // namespace
}  // namespace xllm

/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "eplb_manager.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <torch/torch.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "common/global_flags.h"
#include "core/framework/config/eplb_config.h"

namespace xllm {

using namespace std::chrono_literals;

EplbManager::EplbManager(int32_t layer_num,
                         int32_t device_num,
                         int32_t experts_num)
    : layer_num_(layer_num),
      device_num_(device_num),
      experts_num_(experts_num),
      device_experts_num_(
          experts_num / device_num +
          ::xllm::EPLBConfig::get_instance().redundant_experts_num()) {
  // Initialize tensors with mutex protection
  {
    eplb_policy_ = std::make_unique<EplbPolicy>(
        device_experts_num_, device_num_, layer_num_);
    std::lock_guard<std::mutex> lock(state_.mtx);
    state_.expert_load =
        torch::zeros({layer_num_, experts_num_}, torch::kInt64);
    state_.prepared_layer_id.resize(device_num, -1);
    state_.expert_distribution = torch::zeros(
        {layer_num_, device_num_, device_experts_num_}, torch::kInt32);
    for (int32_t layer = 0; layer < layer_num_; ++layer) {
      for (int32_t device = 0; device < device_num_; ++device) {
        int32_t device_route_experts_num =
            device_experts_num_ -
            ::xllm::EPLBConfig::get_instance().redundant_experts_num();
        int32_t base = device * device_route_experts_num;
        for (int32_t expert = 0; expert < device_experts_num_; ++expert) {
          int32_t value = base + expert;
          if (expert >= device_route_experts_num) {
            value = base + device_route_experts_num - 1;
          }
          state_.expert_distribution[layer][device][expert] = value;
        }
      }
    }
  }

  // Start worker threads
  rebalance_thread_ = std::thread(&EplbManager::rebalance_experts_loop, this);
  manager_thread_ = std::thread(&EplbManager::eplb_manager_loop, this);
}

EplbManager::~EplbManager() {
  {
    std::lock_guard<std::mutex> lock(state_.mtx);
    state_.stop = true;
    state_.data_cv.notify_all();
    state_.state_cv.notify_all();
  }

  if (rebalance_thread_.joinable()) rebalance_thread_.join();
  if (manager_thread_.joinable()) manager_thread_.join();
}

void EplbManager::update_expert_load(
    const std::vector<torch::Tensor> expert_load) {
  std::lock_guard<std::mutex> lock(state_.mtx);
  state_.expert_load_queue.push(expert_load);
  state_.data_cv.notify_one();
}

void EplbManager::aggregate_multi_layer_expert_loads(
    torch::Tensor& expert_load,
    torch::Tensor& expert_ids_list,
    std::vector<torch::Tensor>& expert_loads_list) {
  auto options = torch::TensorOptions().dtype(torch::kInt32);

  for (int32_t device = 0; device < device_num_; ++device) {
    using namespace torch::indexing;
    torch::Tensor expert_load_data_right = expert_loads_list[device].slice(
        1, 1, expert_loads_list[device].size(1));
    torch::Tensor expert_load_data_left = expert_loads_list[device].slice(
        1, 0, expert_loads_list[device].size(1) - 1);
    torch::Tensor expert_load_data_sub =
        expert_load_data_right - expert_load_data_left;
    torch::Tensor first_col =
        expert_loads_list[device].select(1, 0).unsqueeze(1);

    expert_loads_list[device] =
        torch::cat({first_col, expert_load_data_sub}, 1);
  }

  std::vector<torch::Tensor> layer_ids, layer_loads;
  layer_ids.reserve(device_num_);
  layer_loads.reserve(device_num_);
  for (int32_t layer = 0; layer < layer_num_; ++layer) {
    layer_ids.clear();
    layer_loads.clear();
    for (int32_t device = 0; device < device_num_; ++device) {
      auto ids = expert_ids_list[layer][device];
      auto loads = expert_loads_list[device][layer];

      layer_ids.emplace_back(ids.flatten().to(torch::kInt64));
      layer_loads.emplace_back(loads.flatten().to(torch::kInt64));
    }
    torch::Tensor all_ids = torch::cat(layer_ids);
    torch::Tensor all_loads = torch::cat(layer_loads);
    expert_load[layer].scatter_add_(0, all_ids, all_loads);
  }
}

void EplbManager::rebalance_experts_loop() {
  int64_t latest_record_time = absl::ToUnixSeconds(absl::Now());
  while (true) {
    std::vector<std::vector<torch::Tensor>> expert_load_batch;
    {
      std::unique_lock<std::mutex> lock(state_.mtx);
      state_.data_cv.wait(lock, [&] {
        return state_.stop || !state_.expert_load_queue.empty();
      });

      if (state_.stop) return;

      while (!state_.expert_load_queue.empty()) {
        aggregate_multi_layer_expert_loads(state_.expert_load,
                                           state_.expert_distribution,
                                           state_.expert_load_queue.front());
        state_.expert_load_queue.pop();
        int64_t current_time = absl::ToUnixSeconds(absl::Now());
        if (current_time - latest_record_time >=
            ::xllm::EPLBConfig::get_instance().eplb_update_interval()) {
          latest_record_time = current_time;
          auto result = eplb_policy_->rebalance_experts(state_.expert_load);
          state_.expert_distribution = result.first;
          state_.enable_update_vec = result.second;
          state_.expert_load = torch::div(state_.expert_load, 2, "trunc");
          state_.to_be_prepared = find_next_true(state_.enable_update_vec, 0);
          state_.state_cv.notify_all();
        }
      }
    }
  }
}

size_t EplbManager::find_next_true(const std::vector<bool>& vec,
                                   size_t start_pos) {
  if (start_pos >= vec.size()) return static_cast<size_t>(-1);
  auto begin = vec.begin() + start_pos;
  auto it = std::find(begin, vec.end(), true);
  return (it != vec.end()) ? static_cast<size_t>(it - vec.begin())
                           : static_cast<size_t>(-1);
}

void EplbManager::eplb_manager_loop() {
  while (true) {
    {
      std::unique_lock<std::mutex> lock(state_.mtx);
      state_.state_cv.wait(
          lock, [&] { return state_.to_be_prepared != -1 || state_.stop; });

      if (state_.stop) {
        return;
      }
    }
    while (true) {
      {
        std::unique_lock<std::mutex> lock(state_.mtx);
        // Update preparation status
        if (state_.to_be_prepared >= 0) {
          bool all_prepared = true;
          for (auto& layer_id : state_.prepared_layer_id) {
            if (layer_id != state_.to_be_prepared) {
              all_prepared = false;
              break;
            }
          }
          if (all_prepared) {
            state_.ready_layer_id = state_.to_be_prepared;
            // state_.preparing_layer_id = state_.to_be_prepared;
            state_.to_be_prepared = find_next_true(state_.enable_update_vec,
                                                   ++state_.to_be_prepared);
            if (state_.to_be_prepared == -1) {
              state_.preparing_layer_id = -1;
            }
          }
        }
        if (state_.to_be_prepared < 0) {
          break;
        }
      }
    }
  }
}

EplbInfo EplbManager::get_eplb_info() {
  EplbInfo info;
  {
    std::lock_guard<std::mutex> lock(state_.mtx);
    info.update_layer_id = state_.ready_layer_id;
    if (state_.preparing_layer_id != state_.to_be_prepared &&
        state_.to_be_prepared != -1) {
      info.prepare_layer_id = state_.to_be_prepared;
      torch::Tensor distribution =
          state_.expert_distribution[state_.to_be_prepared].contiguous();
      info.expert_ids =
          std::vector<int>(distribution.data_ptr<int>(),
                           distribution.data_ptr<int>() + distribution.numel());
      state_.preparing_layer_id = state_.to_be_prepared;
    } else {
      info.prepare_layer_id = -1;
    }
    state_.ready_layer_id = -1;
  }
  return info;
}

void EplbManager::set_prepared_layer_ids(
    const std::vector<int32_t>& expert_layer_ids) {
  std::lock_guard<std::mutex> lock(state_.mtx);
  for (size_t i = 0;
       i < expert_layer_ids.size() && i < state_.prepared_layer_id.size();
       ++i) {
    if (expert_layer_ids[i] == state_.to_be_prepared) {
      state_.prepared_layer_id[i] = expert_layer_ids[i];
    }
  }
}

}  // namespace xllm

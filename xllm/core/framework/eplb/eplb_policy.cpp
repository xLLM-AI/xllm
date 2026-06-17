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

#include "eplb_policy.h"

#include <ATen/ATen.h>
#include <torch/nn/functional.h>
#include <torch/torch.h>

#include "common/global_flags.h"
#include "core/framework/config/eplb_config.h"

namespace xllm {

EplbPolicy::EplbPolicy(int32_t device_experts_num,
                       int32_t device_num,
                       int32_t layer_num)
    : device_experts_num_(device_experts_num),
      device_num_(device_num),
      layer_num_(layer_num) {
  old_expert_load_ = torch::zeros(
      {layer_num_,
       device_experts_num * device_num -
           device_num *
               ::xllm::EPLBConfig::get_instance().redundant_experts_num()},
      torch::kInt64);
  expert_distribution_ = torch::full(
      {layer_num_, device_num_, device_experts_num_}, -1, torch::kInt32);
}

std::pair<torch::Tensor, std::vector<bool>> EplbPolicy::rebalance_experts(
    torch::Tensor expert_load) {
  std::vector<bool> enable_update_vec(layer_num_, false);
  for (int64_t i = 0; i < layer_num_; ++i) {
    auto current_load = expert_load[i].to(torch::kFloat64);
    auto prev_load = old_expert_load_[i].to(torch::kFloat64);

    auto current_max_val = torch::max(current_load).item<double>() + 1e-6f;
    auto prev_max_val = torch::max(prev_load).item<double>() + 1e-6f;

    current_load = (current_load / current_max_val).unsqueeze(0);
    prev_load = (prev_load / prev_max_val).unsqueeze(0);

    auto cos_sim =
        torch::nn::functional::cosine_similarity(
            current_load,
            prev_load,
            torch::nn::functional::CosineSimilarityFuncOptions().dim(1))
            .item<double>();
    if (cos_sim < ::xllm::EPLBConfig::get_instance().eplb_update_threshold()) {
      enable_update_vec[i] = true;
      old_expert_load_[i] = expert_load[i];
    }
  }

  for (int64_t i = 0; i < layer_num_; ++i) {
    if (enable_update_vec[i]) {
      auto balanced = compute_balanced_pack(expert_load[i]);
      expert_distribution_.index_put_({i}, balanced);
    }
  }
  expert_distribution_ = expert_distribution_.contiguous();
  return {expert_distribution_, enable_update_vec};
}

torch::Tensor EplbPolicy::compute_balanced_pack(
    const torch::Tensor& expert_loads) {
  // Parameter Validation
  CHECK_EQ(expert_loads.dim(), 1) << "expert_loads must be 1D tensor";
  const int64_t num_experts = expert_loads.size(0);

  // Generate Redundant Experts
  auto [updated_weights, redundancy_map] = update_origin_weights(
      expert_loads,
      device_num_ * ::xllm::EPLBConfig::get_instance().redundant_experts_num());

  // Initialize Allocation Matrix
  auto options = torch::TensorOptions().dtype(torch::kInt64);
  auto device_assignments =
      torch::full({device_num_, device_experts_num_}, -1, options);
  auto device_loads = torch::zeros({device_num_}, torch::kInt64);

  // Assign Redundant Experts
  for (int64_t origin_id = 0; origin_id < num_experts; ++origin_id) {
    auto redundant_ids = redundancy_map[origin_id];
    for (int64_t i = 0; i < redundant_ids.size(0); ++i) {
      if (redundant_ids[i].item<int>() == -1) {
        break;
      }
      auto min_idx = torch::argmin(device_loads).item<int64_t>();
      auto available_pos = torch::nonzero(device_assignments[min_idx] == -1);
      if (available_pos.size(0) == 0) {
        LOG(FATAL) << "Device " << min_idx << " is full";
      }
      auto pos = available_pos.select(0, 0).item<int64_t>();

      device_assignments[min_idx][pos] = origin_id;
      device_loads[min_idx] += updated_weights[origin_id].item<int64_t>();
    }
  }

  // Assign Primary Experts
  auto sorted_indices = torch::argsort(-updated_weights);
  for (int64_t i = 0; i < sorted_indices.size(0); ++i) {
    auto expert_id = sorted_indices[i].item<int64_t>();
    auto weight = updated_weights[expert_id].item<int64_t>();

    auto candidate = (device_assignments == -1).sum(1) > 0;
    if (candidate.sum().item<int>() == 0) break;

    auto valid_devices_vec = torch::where(candidate);
    auto valid_devices = valid_devices_vec[0];

    auto min_idx = torch::argmin(device_loads.index({valid_devices}));
    auto target_device = valid_devices[min_idx].item<int64_t>();

    auto pos = torch::nonzero(device_assignments[target_device] == -1);
    if (pos.size(0) == 0) {
      LOG(FATAL) << "Target device " << target_device << " is full";
    }
    auto pos_idx = pos.select(0, 0).item<int64_t>();
    device_assignments[target_device][pos_idx] = expert_id;
    device_loads[target_device] += weight;
  }

  return device_assignments;
}

std::pair<torch::Tensor, torch::Tensor> EplbPolicy::update_origin_weights(
    torch::Tensor expert_loads,
    int32_t redundancy_experts) {
  //  Parameter Validation
  CHECK_EQ(expert_loads.dim(), 1) << "expert_loads must be 1D tensor";
  const int64_t num_experts = expert_loads.size(0);

  //  Initialize Data Structures
  auto redundancy_map =
      torch::full({num_experts, redundancy_experts}, -1, torch::kInt64);
  auto current_weights = expert_loads.clone();

  //  Dynamic Weight Adjustment
  for (int i = 0; i < redundancy_experts; ++i) {
    auto max_idx = torch::argmax(current_weights).item<int64_t>();
    auto redundancy_count =
        torch::sum(redundancy_map[max_idx] != -1).item<int>() + 1;

    // Update redundancy mapping
    redundancy_map[max_idx][redundancy_count - 1] = num_experts + i;

    // Adjust weights using dynamic formula
    auto new_weight =
        (current_weights[max_idx].item<int64_t>() * redundancy_count) /
        (redundancy_count + 1.0);
    current_weights[max_idx] = new_weight;
  }

  return {current_weights, redundancy_map};
}

}  // namespace xllm
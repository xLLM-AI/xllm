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

#include <glog/logging.h>

#include <utility>

namespace xllm {

EplbAggregator::EplbAggregator(int32_t layer_num,
                               int32_t device_num,
                               int32_t device_experts_num)
    : layer_num_(layer_num),
      device_num_(device_num),
      device_experts_num_(device_experts_num) {}

void EplbAggregator::aggregate(
    torch::Tensor& expert_load,
    torch::Tensor& expert_ids_list,
    std::vector<torch::Tensor>& expert_loads_list) const {
  torch::Tensor physical_expert_load = torch::zeros(
      {layer_num_, device_num_, device_experts_num_}, expert_load.options());
  aggregate(
      expert_load, physical_expert_load, expert_ids_list, expert_loads_list);
}

void EplbAggregator::aggregate(
    torch::Tensor& expert_load,
    torch::Tensor& physical_expert_load,
    torch::Tensor& expert_ids_list,
    std::vector<torch::Tensor>& expert_loads_list) const {
  CHECK_EQ(expert_load.dim(), 2);
  CHECK_EQ(expert_load.size(0), layer_num_);
  CHECK_EQ(physical_expert_load.dim(), 3);
  CHECK_EQ(physical_expert_load.size(0), layer_num_);
  CHECK_EQ(physical_expert_load.size(1), device_num_);
  CHECK_EQ(physical_expert_load.size(2), device_experts_num_);
  CHECK_EQ(expert_ids_list.dim(), 3);
  CHECK_EQ(expert_ids_list.size(0), layer_num_);
  CHECK_EQ(expert_ids_list.size(1), device_num_);
  CHECK_EQ(expert_ids_list.size(2), device_experts_num_);
  CHECK_EQ(expert_loads_list.size(), static_cast<size_t>(device_num_));
  for (int32_t device = 0; device < device_num_; ++device) {
    CHECK(expert_loads_list[device].defined())
        << "EPLB expert load tensor is missing for device " << device;
    CHECK_EQ(expert_loads_list[device].dim(), 2)
        << "EPLB expert load tensor must be 2D for device " << device;
    CHECK_EQ(expert_loads_list[device].size(0), layer_num_)
        << "EPLB expert load tensor layer size mismatch for device " << device;
    CHECK_EQ(expert_loads_list[device].size(1), device_experts_num_)
        << "EPLB expert load tensor expert size mismatch for device " << device;
    // Convert cumulative slot counters into deltas without allocating a
    // separate zero-prepend tensor. The first column is already its own delta.
    const torch::Tensor cumulative_load = expert_loads_list[device];
    torch::Tensor delta_load = cumulative_load.clone();
    const int64_t expert_count = cumulative_load.size(1);
    if (expert_count > 1) {
      delta_load.slice(/*dim=*/1, /*start=*/1, /*end=*/expert_count)
          .sub_(cumulative_load.slice(
              /*dim=*/1, /*start=*/0, /*end=*/expert_count - 1));
    }
    expert_loads_list[device] = std::move(delta_load);
    physical_expert_load.select(/*dim=*/1, device)
        .add_(expert_loads_list[device]);
  }

  std::vector<torch::Tensor> layer_ids, layer_loads;
  layer_ids.reserve(device_num_);
  layer_loads.reserve(device_num_);
  for (int32_t layer = 0; layer < layer_num_; ++layer) {
    layer_ids.clear();
    layer_loads.clear();
    for (int32_t device = 0; device < device_num_; ++device) {
      torch::Tensor ids = expert_ids_list[layer][device];
      torch::Tensor loads = expert_loads_list[device][layer];

      layer_ids.emplace_back(ids.flatten().to(torch::kInt64));
      layer_loads.emplace_back(loads.flatten().to(torch::kInt64));
    }
    torch::Tensor all_ids = torch::cat(layer_ids);
    torch::Tensor all_loads = torch::cat(layer_loads);
    expert_load[layer].scatter_add_(0, all_ids, all_loads);
  }
}

}  // namespace xllm

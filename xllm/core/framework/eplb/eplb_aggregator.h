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

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace xllm {

// Stateless aggregator for the per-device per-layer expert counters produced
// by the forward path. The counters are cumulative-along-experts (each step
// records "total tokens routed to each expert since worker start" so that a
// dropped update is idempotent). aggregate() converts them to per-step deltas
// and scatter-adds them into a global [layer_num, num_experts] load tensor.
//
// Extracted out of EplbManager so the transformation is unit-testable in
// isolation (no threads, no policy, no config) and so a future subclass of
// EplbManager can swap in a different aggregation without touching the
// three-thread control plane.
class EplbAggregator final {
 public:
  EplbAggregator(int32_t layer_num,
                 int32_t device_num,
                 int32_t device_experts_num);

  // Merge one batch of per-device cumulative counters into `expert_load`.
  //   expert_load       : [layer_num, num_experts] int64, mutated in place
  //   expert_ids_list   : [layer_num, device_num, device_experts_num] int32,
  //                       the current post-migration slot map. Not mutated;
  //                       we index into it per layer.
  //   expert_loads_list : one tensor per device, each shape
  //                       [layer_num, device_experts_num] int64,
  //                       replaced in place with the diff'd deltas.
  void aggregate(torch::Tensor& expert_load,
                 torch::Tensor& expert_ids_list,
                 std::vector<torch::Tensor>& expert_loads_list) const;

  // Merge the same counters while preserving the measured physical slot load
  // as [layer_num, device_num, device_experts_num]. The physical tensor is
  // used by the policy as the current-layout baseline; reducing it to logical
  // expert totals would lose the rank skew that EPLB is meant to fix.
  void aggregate(torch::Tensor& expert_load,
                 torch::Tensor& physical_expert_load,
                 torch::Tensor& expert_ids_list,
                 std::vector<torch::Tensor>& expert_loads_list) const;

 private:
  int32_t layer_num_;
  int32_t device_num_;
  int32_t device_experts_num_;
};

}  // namespace xllm

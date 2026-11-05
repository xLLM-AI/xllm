/* Copyright 2026 The xLLM Authors.

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

#pragma once

#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace xllm {

// Worker copy of the scheduler's recurrent-state slot assignment in the shared
// static arena. A state id is its group-0 slot; the other groups' slots arrive
// with each batch that uses the state. A state keeps its slots until it is
// released, so rows for states outside the current batch may be stale but are
// never read.
class LinearStateSlotTable final {
 public:
  LinearStateSlotTable(size_t num_slots,
                       size_t num_groups,
                       const torch::Device& device);

  size_t num_groups() const { return num_groups_; }

  // Each row is [state id, slot of group 0, ..., slot of group G-1].
  void update(const std::vector<int32_t>& rows);

  // Negative ids are sentinels and map to themselves.
  int64_t slot(int64_t state_id, size_t group) const;
  torch::Tensor slots(const torch::Tensor& state_ids, size_t group) const;

  // [num_groups, num_slots], for backends that translate ids themselves.
  const torch::Tensor& device_slots() const { return device_slots_; }

 private:
  size_t num_slots_;
  size_t num_groups_;
  // Both are [num_groups, num_slots].
  std::vector<int32_t> host_slots_;
  torch::Tensor device_slots_;
};

}  // namespace xllm

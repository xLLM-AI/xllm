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

#include "core/kv_cache/storage/linear_state_slot_table.h"

#include <glog/logging.h>

#include <limits>

namespace xllm {

LinearStateSlotTable::LinearStateSlotTable(size_t num_slots,
                                           size_t num_groups,
                                           const torch::Device& device)
    : num_slots_(num_slots), num_groups_(num_groups) {
  CHECK_GT(num_slots_, 0u);
  CHECK_GT(num_groups_, 0u);
  CHECK_LE(num_slots_,
           static_cast<size_t>(std::numeric_limits<int32_t>::max()));
  host_slots_.assign(num_groups_ * num_slots_, 0);
  last_update_rows_.resize(num_slots_);
  device_slots_ = torch::from_blob(host_slots_.data(),
                                   {static_cast<int64_t>(num_groups_),
                                    static_cast<int64_t>(num_slots_)},
                                   torch::kInt32)
                      .to(device,
                          torch::kInt32,
                          /*non_blocking=*/false,
                          /*copy=*/true);
}

void LinearStateSlotTable::update(const std::vector<int32_t>& rows) {
  if (rows.empty()) {
    return;
  }
  const size_t row_width = num_groups_ + 1;
  CHECK_EQ(rows.size() % row_width, 0u)
      << "Linear state slot rows must be [state id, group slots...]";
  const size_t num_rows = rows.size() / row_width;
  for (size_t row = 0; row < num_rows; ++row) {
    const int32_t* entry = rows.data() + row * row_width;
    const int32_t state_id = entry[0];
    CHECK_GE(state_id, 0);
    CHECK_LT(static_cast<size_t>(state_id), num_slots_);
    for (size_t group = 0; group < num_groups_; ++group) {
      CHECK_GE(entry[group + 1], 0);
    }
    last_update_rows_[state_id] = row;
  }

  std::vector<int64_t> state_ids;
  for (size_t row = 0; row < num_rows; ++row) {
    const int32_t* entry = rows.data() + row * row_width;
    const int32_t state_id = entry[0];
    if (last_update_rows_[state_id] != row) {
      continue;
    }
    bool changed = false;
    for (size_t group = 0; group < num_groups_; ++group) {
      int32_t& slot = host_slots_[group * num_slots_ + state_id];
      if (slot == entry[group + 1]) {
        continue;
      }
      slot = entry[group + 1];
      changed = true;
    }
    if (!changed) {
      continue;
    }
    if (state_ids.empty()) {
      state_ids.reserve(num_rows);
    }
    state_ids.emplace_back(state_id);
  }
  if (state_ids.empty()) {
    return;
  }

  const size_t num_changed_rows = state_ids.size();
  std::vector<int32_t> group_slots(num_groups_ * num_changed_rows);
  for (size_t row = 0; row < num_changed_rows; ++row) {
    const size_t state_id = static_cast<size_t>(state_ids[row]);
    for (size_t group = 0; group < num_groups_; ++group) {
      group_slots[group * num_changed_rows + row] =
          host_slots_[group * num_slots_ + state_id];
    }
  }
  const torch::Device device = device_slots_.device();
  const torch::Tensor index =
      torch::from_blob(state_ids.data(),
                       {static_cast<int64_t>(num_changed_rows)},
                       torch::kInt64)
          .to(device);
  const torch::Tensor source =
      torch::from_blob(group_slots.data(),
                       {static_cast<int64_t>(num_groups_),
                        static_cast<int64_t>(num_changed_rows)},
                       torch::kInt32)
          .to(device);
  device_slots_.index_copy_(1, index, source);
}

int64_t LinearStateSlotTable::slot(int64_t state_id, size_t group) const {
  if (state_id < 0) {
    return state_id;
  }
  CHECK_LT(group, num_groups_);
  CHECK_LT(static_cast<size_t>(state_id), num_slots_);
  return host_slots_[group * num_slots_ + static_cast<size_t>(state_id)];
}

torch::Tensor LinearStateSlotTable::slots(const torch::Tensor& state_ids,
                                          size_t group) const {
  if (!state_ids.defined()) {
    return state_ids;
  }
  CHECK_LT(group, num_groups_);
  const torch::Tensor group_slots =
      device_slots_.select(0, static_cast<int64_t>(group))
          .index_select(0,
                        state_ids.clamp_min(0).to(torch::kInt64).reshape({-1}))
          .reshape(state_ids.sizes());
  return torch::where(
      state_ids.lt(0), state_ids, group_slots.to(state_ids.scalar_type()));
}

}  // namespace xllm

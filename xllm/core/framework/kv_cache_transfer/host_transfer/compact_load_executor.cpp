/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "framework/kv_cache_transfer/host_transfer/compact_load_executor.h"

#include <glog/logging.h>

#include <algorithm>
#include <array>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <utility>
#include <vector>

#include "framework/kv_cache/kv_cache_utils.h"
#include "platform/batch_memcpy.h"
#include "platform/layer_synchronizer.h"
#include "platform/stream.h"

namespace xllm {
namespace {

std::vector<int64_t> get_staging_shape(const torch::Tensor& host_tensor,
                                       int64_t blocks_per_chunk,
                                       int64_t layer_count) {
  std::vector<int64_t> shape;
  shape.reserve(static_cast<size_t>(host_tensor.dim()));
  shape.emplace_back(blocks_per_chunk);
  shape.emplace_back(layer_count);
  for (int64_t dim = 2; dim < host_tensor.dim(); ++dim) {
    shape.emplace_back(host_tensor.size(dim));
  }
  return shape;
}

}  // namespace

CompactLoadExecutor::CompactLoadExecutor(const HostKVLayout& layout,
                                         const Device& device,
                                         uint32_t layer_copy_batches,
                                         BatchMemcpy& batch_memcpy,
                                         size_t target_bytes)
    : device_(device),
      target_bytes_(target_bytes),
      layers_per_event_(
          get_layers_per_event(layout.num_layers(), layer_copy_batches)),
      ranges_(build_layer_ranges(layout.num_layers(), layers_per_event_)),
      batch_memcpy_(batch_memcpy) {
  CHECK_GT(target_bytes_, static_cast<size_t>(0))
      << "compact H2D target bytes must be positive.";
  device_.set_device();
  init(layout);
}

CompactLoadExecutor::~CompactLoadExecutor() { drain(); }

bool CompactLoadExecutor::execute(
    const HostKVRequest& request,
    const std::shared_ptr<LayerSynchronizer>& synchronizer) {
  std::lock_guard<std::mutex> lock(mutex_);
  const GroupedHostKVMappings mappings_by_group = group_mappings(request);
  for (size_t range_index = 0; range_index < ranges_.size(); ++range_index) {
    for (const auto& [group_id, mappings] : mappings_by_group) {
      if (!submit_range(
              group_id, groups_.at(group_id), range_index, mappings)) {
        drain_or_die("H2D submission failed");
        return false;
      }
    }
    if (!synchronizer->record_stream(static_cast<int64_t>(range_index),
                                     copy_stream_.get())) {
      drain_or_die("layer-ready event recording failed");
      return false;
    }
  }
  return true;
}

void CompactLoadExecutor::drain() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (drained_) {
    return;
  }
  drain_or_die("shutdown");
  drained_ = true;
}

uint32_t CompactLoadExecutor::event_count() const {
  return static_cast<uint32_t>(ranges_.size());
}

uint32_t CompactLoadExecutor::layers_per_event() const {
  return layers_per_event_;
}

void CompactLoadExecutor::init(const HostKVLayout& layout) {
  int64_t max_blocks_per_chunk = 0;
  for (const HostKVGroupLayout& group : layout.groups()) {
    GroupState state = init_group(layout, group);
    max_blocks_per_chunk =
        std::max(max_blocks_per_chunk, state.blocks_per_chunk);
    groups_.emplace(group.group_id, std::move(state));
  }

  for (SlotState& slot : slots_) {
    for (const auto& [group_id, group] : groups_) {
      for (const auto& [role, role_state] : group.roles) {
        const std::vector<int64_t> shape =
            get_staging_shape(role_state.host_tensor,
                              group.blocks_per_chunk,
                              role_state.max_span_layers);
        slot.device_staging.emplace(
            GroupRoleKey{group_id, role},
            torch::empty(shape,
                         torch::TensorOptions()
                             .dtype(role_state.host_tensor.scalar_type())
                             .device(device_.unwrap())));
      }
    }
    create_host_page_aligned_tensor({max_blocks_per_chunk},
                                    torch::kInt64,
                                    &slot.host_indices,
                                    &slot.index_region);
    slot.device_indices = torch::empty(
        {max_blocks_per_chunk},
        torch::TensorOptions().dtype(torch::kInt64).device(device_.unwrap()));
  }
  copy_stream_ = device_.get_stream_from_pool();
  CHECK(copy_stream_ != nullptr) << "Compact H2D copy stream is unavailable.";
  log_init();
}

CompactLoadExecutor::GroupState CompactLoadExecutor::init_group(
    const HostKVLayout& layout,
    const HostKVGroupLayout& group) const {
  GroupState state;
  state.layers.reserve(group.layers.size());
  for (const HostKVLayerLayout& layer : group.layers) {
    state.layers.emplace_back(LayerState{
        layer.absolute_layer_id, layer.group_layer_slot, layer.device_roles});
  }
  for (KVCacheTensorRole::Value role : layout.active_roles(group.group_id)) {
    state.roles.emplace(role,
                        RoleState{group.host_roles.at(role),
                                  layout.block_bytes(group.group_id, role),
                                  0});
  }

  state.ranges.reserve(ranges_.size());
  for (const LayerRange& range : ranges_) {
    RangeState range_state = init_range(state, range);
    const int64_t span_layers = range_state.slot_end - range_state.slot_begin;
    for (KVCacheTensorRole::Value role : range_state.active_roles) {
      RoleState& role_state = state.roles.at(role);
      role_state.max_span_layers =
          std::max(role_state.max_span_layers, span_layers);
      const size_t span_bytes =
          static_cast<size_t>(span_layers) * role_state.block_bytes;
      state.max_span_bytes = std::max(state.max_span_bytes, span_bytes);
    }
    state.ranges.emplace_back(std::move(range_state));
  }
  CHECK_GT(state.max_span_bytes, static_cast<size_t>(0));
  const size_t block_count =
      std::max(target_bytes_ / state.max_span_bytes, static_cast<size_t>(1));
  CHECK_LE(block_count,
           static_cast<size_t>(std::numeric_limits<int64_t>::max()));
  state.blocks_per_chunk = static_cast<int64_t>(block_count);
  return state;
}

CompactLoadExecutor::RangeState CompactLoadExecutor::init_range(
    const GroupState& group,
    const LayerRange& range) const {
  RangeState state;
  const LayerState* first = nullptr;
  const LayerState* last = nullptr;
  for (const LayerState& layer : group.layers) {
    if (layer.absolute_layer_id < range.begin ||
        layer.absolute_layer_id >= range.end) {
      continue;
    }
    if (first == nullptr) {
      first = &layer;
    }
    last = &layer;
    for (const auto& [role, tensor] : layer.device_roles) {
      (void)tensor;
      if (std::find(state.active_roles.begin(),
                    state.active_roles.end(),
                    role) == state.active_roles.end()) {
        state.active_roles.emplace_back(role);
      }
    }
  }
  std::sort(state.active_roles.begin(), state.active_roles.end());
  if (first == nullptr) {
    return state;
  }
  state.slot_begin = first->group_layer_slot;
  state.slot_end = last->group_layer_slot + 1;
  int64_t expected_slot = state.slot_begin;
  for (const LayerState& layer : group.layers) {
    if (layer.absolute_layer_id < range.begin ||
        layer.absolute_layer_id >= range.end) {
      continue;
    }
    CHECK_EQ(layer.group_layer_slot, expected_slot)
        << "Compact H2D range requires contiguous group-layer slots.";
    ++expected_slot;
  }
  CHECK_EQ(expected_slot, state.slot_end)
      << "Compact H2D range requires a contiguous Host slice.";
  for (const auto& [role, role_state] : group.roles) {
    if (std::find(state.active_roles.begin(), state.active_roles.end(), role) !=
        state.active_roles.end()) {
      const torch::Tensor span = role_state.host_tensor[0].narrow(
          /*dim=*/0, state.slot_begin, state.slot_end - state.slot_begin);
      CHECK(span.is_contiguous())
          << "Compact H2D Host range slice must be contiguous.";
    }
  }
  return state;
}

bool CompactLoadExecutor::submit_range(
    int32_t group_id,
    const GroupState& group,
    size_t range_index,
    const std::vector<HostKVMapping>& mappings) {
  const RangeState& range = group.ranges[range_index];
  if (range.slot_begin == range.slot_end) {
    return true;
  }
  const size_t chunk_size = static_cast<size_t>(group.blocks_per_chunk);
  for (size_t offset = 0; offset < mappings.size(); offset += chunk_size) {
    const size_t count = std::min(chunk_size, mappings.size() - offset);
    SlotState& slot = next_slot();
    wait_slot(slot);
    fill_indices(slot, mappings, offset, count);
    if (!enqueue_h2d(slot, group_id, group, range, mappings, offset, count)) {
      return false;
    }
    scatter(
        slot, group_id, group, range_index, range, static_cast<int64_t>(count));
    slot.completion_event = copy_stream_->record_event();
    if (slot.completion_event == nullptr) {
      return false;
    }
  }
  return true;
}

CompactLoadExecutor::SlotState& CompactLoadExecutor::next_slot() {
  SlotState& slot = slots_[next_slot_index_];
  next_slot_index_ = (next_slot_index_ + 1) % slots_.size();
  return slot;
}

void CompactLoadExecutor::wait_slot(const SlotState& slot) const {
  if (slot.completion_event != nullptr &&
      !slot.completion_event->synchronize()) {
    LOG(FATAL) << "Failed to reuse Compact H2D staging slot.";
  }
}

void CompactLoadExecutor::fill_indices(
    SlotState& slot,
    const std::vector<HostKVMapping>& mappings,
    size_t offset,
    size_t count) const {
  int64_t* ids = slot.host_indices.data_ptr<int64_t>();
  for (size_t index = 0; index < count; ++index) {
    ids[index] = mappings[offset + index].device_block_id;
  }
}

bool CompactLoadExecutor::enqueue_h2d(
    SlotState& slot,
    int32_t group_id,
    const GroupState& group,
    const RangeState& range,
    const std::vector<HostKVMapping>& mappings,
    size_t offset,
    size_t count) {
  const int64_t span_layers = range.slot_end - range.slot_begin;
  std::vector<torch::Tensor> sources;
  std::vector<torch::Tensor> destinations;
  sources.reserve(count * range.active_roles.size() + 1);
  destinations.reserve(count * range.active_roles.size() + 1);
  for (size_t mapping_offset = 0; mapping_offset < count; ++mapping_offset) {
    const HostKVMapping& mapping = mappings[offset + mapping_offset];
    for (KVCacheTensorRole::Value role : range.active_roles) {
      // Sparse roles still use the contiguous Host window; scatter filters.
      const RoleState& role_state = group.roles.at(role);
      sources.emplace_back(role_state.host_tensor[mapping.host_block_id].narrow(
          /*dim=*/0, range.slot_begin, span_layers));
      torch::Tensor& staging =
          slot.device_staging.at(GroupRoleKey{group_id, role});
      destinations.emplace_back(staging[mapping_offset].narrow(
          /*dim=*/0, /*start=*/0, span_layers));
    }
  }
  sources.emplace_back(slot.host_indices.narrow(/*dim=*/0, /*start=*/0, count));
  destinations.emplace_back(
      slot.device_indices.narrow(/*dim=*/0, /*start=*/0, count));
  return batch_memcpy_.submit_h2d(sources, destinations, copy_stream_.get());
}

void CompactLoadExecutor::scatter(SlotState& slot,
                                  int32_t group_id,
                                  const GroupState& group,
                                  size_t range_index,
                                  const RangeState& range,
                                  int64_t count) const {
  const LayerRange& layer_range = ranges_[range_index];
  const c10::StreamGuard guard = copy_stream_->set_stream_guard();
  const torch::Tensor indices =
      slot.device_indices.narrow(/*dim=*/0, /*start=*/0, count);
  for (const LayerState& layer : group.layers) {
    if (layer.absolute_layer_id < layer_range.begin ||
        layer.absolute_layer_id >= layer_range.end) {
      continue;
    }
    const int64_t staging_layer = layer.group_layer_slot - range.slot_begin;
    for (const auto& [role, device_tensor] : layer.device_roles) {
      const torch::Tensor& staging =
          slot.device_staging.at(GroupRoleKey{group_id, role});
      device_tensor.index_copy_(
          /*dim=*/0,
          indices,
          staging.narrow(/*dim=*/0, /*start=*/0, count)
              .select(/*dim=*/1, staging_layer));
    }
  }
}

void CompactLoadExecutor::log_init() const {
  std::ostringstream summary;
  summary << "Compact H2D initialized: ranges=" << ranges_.size()
          << ", target_bytes=" << target_bytes_ << ", groups=[";
  bool first = true;
  for (const auto& [group_id, group] : groups_) {
    if (!first) {
      summary << ", ";
    }
    summary << "{group_id=" << group_id
            << ", max_span_bytes=" << group.max_span_bytes
            << ", blocks_per_chunk=" << group.blocks_per_chunk << ", roles=[";
    bool first_role = true;
    for (const auto& [role, role_state] : group.roles) {
      if (!first_role) {
        summary << ", ";
      }
      summary << "{role=" << static_cast<int32_t>(role) << ", spans=[";
      for (size_t index = 0; index < group.ranges.size(); ++index) {
        if (index > 0) {
          summary << ",";
        }
        const RangeState& range = group.ranges[index];
        const bool active = std::find(range.active_roles.begin(),
                                      range.active_roles.end(),
                                      role) != range.active_roles.end();
        summary << (active ? static_cast<size_t>(range.slot_end -
                                                 range.slot_begin) *
                                 role_state.block_bytes
                           : 0);
      }
      summary << "]}";
      first_role = false;
    }
    summary << "]}";
    first = false;
  }
  size_t staging_bytes = 0;
  for (const auto& [key, tensor] : slots_.front().device_staging) {
    (void)key;
    staging_bytes += tensor.nbytes();
  }
  summary << "], dual_slot_staging_bytes=" << staging_bytes * slots_.size();
  LOG(INFO) << summary.str();
}

void CompactLoadExecutor::drain_or_die(const char* reason) {
  try {
    for (SlotState& slot : slots_) {
      if (slot.completion_event != nullptr &&
          !slot.completion_event->synchronize()) {
        LOG(FATAL) << "Failed to drain Compact H2D slot: reason=" << reason;
      }
    }
    if (copy_stream_ != nullptr && copy_stream_->synchronize() != 0) {
      LOG(FATAL) << "Failed to drain Compact H2D stream: reason=" << reason;
    }
  } catch (const std::exception& error) {
    LOG(FATAL) << "Failed to drain Compact H2D stream: reason=" << reason
               << ", error=" << error.what();
  }
}

}  // namespace xllm

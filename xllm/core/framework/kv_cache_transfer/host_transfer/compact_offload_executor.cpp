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

#include "framework/kv_cache_transfer/host_transfer/compact_offload_executor.h"

#include <glog/logging.h>

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <utility>
#include <vector>

#include "framework/kv_cache_transfer/host_transfer/transfer_utils.h"
#include "platform/batch_memcpy.h"
#include "platform/device.h"
#include "platform/stream.h"
#if defined(USE_MLU)
#include "kernels/mlu/mlu_ops_api.h"
#endif

namespace xllm {
namespace {

// Match the contiguous KV cache tensor stride alignment.
constexpr size_t kCompactD2HSlotAlignment = 512;

bool mapping_less(const HostKVMapping& lhs, const HostKVMapping& rhs) {
  if (lhs.device_block_id != rhs.device_block_id) {
    return lhs.device_block_id < rhs.device_block_id;
  }
  return lhs.host_block_id < rhs.host_block_id;
}

size_t checked_multiply(size_t lhs, size_t rhs, const char* message) {
  if (lhs == 0 || rhs == 0) {
    return 0;
  }
  CHECK_LE(lhs, std::numeric_limits<size_t>::max() / rhs) << message;
  return lhs * rhs;
}

size_t checked_add(size_t lhs, size_t rhs, const char* message) {
  CHECK_LE(lhs, std::numeric_limits<size_t>::max() - rhs) << message;
  return lhs + rhs;
}

size_t aligned_slot_bytes(size_t bytes) {
  CHECK_LE(bytes,
           std::numeric_limits<size_t>::max() - (kCompactD2HSlotAlignment - 1))
      << "compact D2H aligned slot capacity overflows.";
  return ((bytes + kCompactD2HSlotAlignment - 1) / kCompactD2HSlotAlignment) *
         kCompactD2HSlotAlignment;
}

torch::Tensor byte_view(const torch::Tensor& tensor) {
  return tensor.view({-1}).view(torch::kUInt8);
}

}  // namespace

CompactOffloadExecutor::CompactOffloadExecutor(const HostKVLayout& layout,
                                               const Device& device,
                                               BatchMemcpy& batch_memcpy,
                                               size_t target_bytes)
    : device_(device),
      batch_memcpy_(batch_memcpy),
      target_bytes_(target_bytes) {
  CHECK_GT(target_bytes_, 0U) << "compact D2H target bytes must be positive.";
  device_.set_device();
  init(layout);
}

CompactOffloadExecutor::~CompactOffloadExecutor() { drain(); }

void CompactOffloadExecutor::execute(const HostKVRequest& request,
                                     const Stream& compute_stream) {
  std::lock_guard<std::mutex> lock(mutex_);
  GroupedHostKVMappings request_groups =
      group_mappings(request.target_mappings);
  for (const HostKVMapping& mapping : request.draft_mappings) {
    request_groups[mapping.group_id].emplace_back(mapping);
  }
  for (auto& [group_id, mappings] : request_groups) {
    (void)group_id;
    std::sort(mappings.begin(), mappings.end(), mapping_less);
  }

  pack_stream_->wait_stream(compute_stream);
  for (const auto& [group_id, mappings] : request_groups) {
    const GroupState& group = groups_.at(group_id);
    for (size_t mapping_offset = 0; mapping_offset < mappings.size();
         mapping_offset += static_cast<size_t>(group.blocks_per_tile)) {
      const size_t mapping_count =
          std::min(static_cast<size_t>(group.blocks_per_tile),
                   mappings.size() - mapping_offset);
      SlotState& slot = next_slot();
      reuse_slot(slot);
      submit(slot, group_id, group, mappings, mapping_offset, mapping_count);
    }
  }
  flush_slots();
}

void CompactOffloadExecutor::drain() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (drained_) {
    return;
  }
  flush_slots();
  drained_ = true;
}

void CompactOffloadExecutor::append_role(GroupState* group,
                                         const HostKVLayout& layout,
                                         const HostKVGroupLayout& layout_group,
                                         KVCacheTensorRole::Value role) {
  RoleState state;
  state.host = layout_group.host_roles.at(role);
  state.block_bytes = layout.block_bytes(layout_group.group_id, role);
  state.packed_offset = group->packed_bytes;
  state.padded_bytes = static_cast<size_t>(state.host[0].nbytes());

  const size_t role_end = checked_add(state.packed_offset,
                                      state.padded_bytes,
                                      "compact D2H role bytes overflow.");
  std::vector<std::pair<size_t, size_t>> ranges;
  ranges.reserve(layout_group.layers.size());
  state.sources.reserve(layout_group.layers.size());
  for (const HostKVLayerLayout& layer : layout_group.layers) {
    auto role_it = layer.device_roles.find(role);
    if (role_it == layer.device_roles.end()) {
      continue;
    }
    const torch::Tensor& source = role_it->second;
    const size_t layer_offset =
        checked_multiply(static_cast<size_t>(layer.group_layer_slot),
                         state.block_bytes,
                         "compact D2H layer offset overflows.");
    const size_t destination_offset =
        checked_add(state.packed_offset,
                    layer_offset,
                    "compact D2H destination offset overflows.");
    const size_t destination_end =
        checked_add(destination_offset,
                    state.block_bytes,
                    "compact D2H destination range overflows.");
    CHECK_LE(destination_end, role_end)
        << "compact D2H destination exceeds its padded role.";
    CHECK_EQ(destination_offset % source.element_size(), 0U)
        << "compact D2H destination offset is not dtype aligned.";
    CHECK_EQ(state.block_bytes % source.element_size(), 0U)
        << "compact D2H block bytes are not dtype aligned.";
    ranges.emplace_back(destination_offset, destination_end);
    state.sources.emplace_back(SourceState{source, destination_offset});
  }
  CHECK(!state.sources.empty())
      << "compact D2H role must contain active sources.";
  std::sort(ranges.begin(), ranges.end());
  for (size_t index = 1; index < ranges.size(); ++index) {
    CHECK_LE(ranges[index - 1].second, ranges[index].first)
        << "compact D2H destination ranges overlap.";
  }
  group->packed_bytes = role_end;
  group->roles.emplace_back(std::move(state));
}

CompactOffloadExecutor::GroupState CompactOffloadExecutor::make_group(
    const HostKVLayout& layout,
    const HostKVGroupLayout& layout_group) {
  GroupState group;
  const std::vector<KVCacheTensorRole::Value>& roles =
      layout.active_roles(layout_group.group_id);
  group.roles.reserve(roles.size());
  for (KVCacheTensorRole::Value role : roles) {
    append_role(&group, layout, layout_group, role);
  }
  CHECK(!group.roles.empty()) << "compact D2H group must have active roles.";
  CHECK_GT(group.packed_bytes, 0U)
      << "compact D2H group must have packed bytes.";
  for (const RoleState& role : group.roles) {
    for (const SourceState& source : role.sources) {
      CHECK_EQ(group.packed_bytes % source.tensor.element_size(), 0U)
          << "compact D2H mapping bytes are not dtype aligned.";
    }
  }
  return group;
}

void CompactOffloadExecutor::init(const HostKVLayout& layout) {
  size_t slot_bytes = 0;
  int64_t max_blocks_per_tile = 0;
  bool expanded_for_mapping = false;
  for (const HostKVGroupLayout& layout_group : layout.groups()) {
    GroupState group = make_group(layout, layout_group);
    expanded_for_mapping =
        expanded_for_mapping || group.packed_bytes > target_bytes_;
    const size_t tile_blocks =
        std::max(static_cast<size_t>(1), target_bytes_ / group.packed_bytes);
    CHECK_LE(tile_blocks,
             static_cast<size_t>(std::numeric_limits<int64_t>::max()))
        << "compact D2H tile block count overflows int64.";
    group.blocks_per_tile = static_cast<int64_t>(tile_blocks);
    const size_t group_slot_bytes =
        checked_multiply(tile_blocks,
                         group.packed_bytes,
                         "compact D2H slot capacity overflows.");
    slot_bytes = std::max(slot_bytes, group_slot_bytes);
    max_blocks_per_tile =
        std::max(max_blocks_per_tile, static_cast<int64_t>(tile_blocks));
    groups_.emplace(layout_group.group_id, std::move(group));
  }

  CHECK_GT(slot_bytes, 0U) << "compact D2H slot capacity must be positive.";
  slot_bytes = aligned_slot_bytes(slot_bytes);
  const torch::TensorOptions byte_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(device_.unwrap());
  const torch::TensorOptions id_options =
      torch::TensorOptions().dtype(torch::kInt64).device(device_.unwrap());
  for (SlotState& slot : slots_) {
    slot.packed =
        torch::zeros({static_cast<int64_t>(slot_bytes)}, byte_options);
    slot.host_block_ids = torch::empty({max_blocks_per_tile}, torch::kInt64);
    slot.device_block_ids = torch::empty({max_blocks_per_tile}, id_options);
  }
  pack_stream_ = device_.get_stream_from_pool();
  d2h_stream_ = device_.get_stream_from_pool();
  CHECK(pack_stream_ != nullptr) << "compact D2H pack stream must not be null.";
  CHECK(d2h_stream_ != nullptr) << "compact D2H D2H stream must not be null.";
  const size_t total_slot_bytes = checked_multiply(
      slots_.size(), slot_bytes, "compact D2H total slot capacity overflows.");
  log_init(slot_bytes, total_slot_bytes, expanded_for_mapping);
}

CompactOffloadExecutor::SlotState& CompactOffloadExecutor::next_slot() {
  SlotState& slot = slots_[next_slot_index_];
  next_slot_index_ = (next_slot_index_ + 1) % slots_.size();
  return slot;
}

void CompactOffloadExecutor::reuse_slot(SlotState& slot) {
  if (slot.d2h_complete == nullptr) {
    return;
  }
  CHECK(pack_stream_->wait_event(slot.d2h_complete))
      << "compact D2H pack stream cannot wait for D2H completion.";
  slot.d2h_complete.reset();
}

void CompactOffloadExecutor::pack_blocks(const GroupState& group,
                                         const torch::Tensor& block_ids,
                                         const torch::Tensor& packed) {
#if defined(USE_MLU)
  size_t source_count = 0;
  for (const RoleState& role : group.roles) {
    source_count = checked_add(source_count,
                               role.sources.size(),
                               "compact D2H source count overflows.");
  }
  std::vector<torch::Tensor> sources;
  std::vector<torch::Tensor> destinations;
  sources.reserve(source_count);
  destinations.reserve(source_count);
  for (const RoleState& role : group.roles) {
    for (const SourceState& source : role.sources) {
      sources.emplace_back(source.tensor);
      std::vector<int64_t> shape(source.tensor.sizes().begin(),
                                 source.tensor.sizes().end());
      shape[0] = block_ids.size(0);
      destinations.emplace_back(
          packed
              .narrow(1,
                      static_cast<int64_t>(source.destination_offset),
                      static_cast<int64_t>(role.block_bytes))
              .view(source.tensor.scalar_type())
              .view(shape));
    }
  }
  kernel::mlu::pack_cache_blocks(sources, block_ids, destinations);
#else
  static_cast<void>(group);
  static_cast<void>(block_ids);
  static_cast<void>(packed);
  LOG(FATAL) << "Compact D2H is only supported on MLU.";
#endif
}

void CompactOffloadExecutor::submit(SlotState& slot,
                                    int32_t group_id,
                                    const GroupState& group,
                                    const std::vector<HostKVMapping>& mappings,
                                    size_t mapping_offset,
                                    size_t mapping_count) {
  if (slot.last_group_id != group_id) {
    const c10::StreamGuard stream_guard = pack_stream_->set_stream_guard();
    slot.packed.zero_();
    slot.last_group_id = group_id;
  }

  const int64_t tile_count = static_cast<int64_t>(mapping_count);
  int64_t* const host_ids = slot.host_block_ids.data_ptr<int64_t>();
  for (size_t index = 0; index < mapping_count; ++index) {
    host_ids[index] = mappings[mapping_offset + index].device_block_id;
  }
  torch::Tensor device_ids = slot.device_block_ids.narrow(0, 0, tile_count);
  torch::Tensor packed =
      slot.packed
          .narrow(0,
                  0,
                  static_cast<int64_t>(checked_multiply(
                      mapping_count,
                      group.packed_bytes,
                      "compact D2H packed tile bytes overflow.")))
          .view({tile_count, static_cast<int64_t>(group.packed_bytes)});
  {
    const c10::StreamGuard stream_guard = pack_stream_->set_stream_guard();
    device_ids.copy_(slot.host_block_ids.narrow(0, 0, tile_count),
                     /*non_blocking=*/true);
    pack_blocks(group, device_ids, packed);
  }
  StreamEventPtr pack_complete = pack_stream_->record_event();
  CHECK(pack_complete != nullptr)
      << "compact D2H pack completion event failed.";

  CHECK(d2h_stream_->wait_event(pack_complete))
      << "compact D2H stream cannot wait for pack completion.";
  submit_d2h(slot, group, mappings, mapping_offset, mapping_count);
}

void CompactOffloadExecutor::submit_d2h(
    SlotState& slot,
    const GroupState& group,
    const std::vector<HostKVMapping>& mappings,
    size_t mapping_offset,
    size_t mapping_count) {
  const size_t descriptor_count =
      checked_multiply(mapping_count,
                       group.roles.size(),
                       "compact D2H descriptor count overflows.");
  std::vector<torch::Tensor> sources;
  std::vector<torch::Tensor> destinations;
  sources.reserve(descriptor_count);
  destinations.reserve(descriptor_count);
  for (size_t local_mapping = 0; local_mapping < mapping_count;
       ++local_mapping) {
    const HostKVMapping& mapping = mappings[mapping_offset + local_mapping];
    const size_t mapping_base =
        checked_multiply(local_mapping,
                         group.packed_bytes,
                         "compact D2H packed mapping offset overflows.");
    for (const RoleState& role : group.roles) {
      const size_t source_offset =
          checked_add(mapping_base,
                      role.packed_offset,
                      "compact D2H role offset overflows.");
      sources.emplace_back(
          slot.packed.narrow(0,
                             static_cast<int64_t>(source_offset),
                             static_cast<int64_t>(role.padded_bytes)));
      destinations.emplace_back(byte_view(role.host[mapping.host_block_id]));
    }
  }
  if (!batch_memcpy_.submit_d2h(sources, destinations, d2h_stream_.get())) {
    LOG(FATAL) << "Compact D2H descriptor submission failed.";
  }
  slot.d2h_complete = d2h_stream_->record_event();
  CHECK(slot.d2h_complete != nullptr) << "compact D2H completion event failed.";
}

void CompactOffloadExecutor::flush_slots() {
  for (SlotState& slot : slots_) {
    if (slot.d2h_complete == nullptr) {
      continue;
    }
    if (!slot.d2h_complete->synchronize()) {
      LOG(FATAL) << "Compact D2H completion event failed.";
    }
    slot.d2h_complete.reset();
  }
}

void CompactOffloadExecutor::log_init(size_t slot_bytes,
                                      size_t total_slot_bytes,
                                      bool expanded_for_mapping) const {
  std::ostringstream summary;
  summary << "Compact D2H initialized: target_bytes=" << target_bytes_
          << ", slot_bytes=" << slot_bytes << ", slots=" << slots_.size();
  LOG(INFO) << summary.str();
  if (expanded_for_mapping) {
    const size_t budgeted_bytes = checked_multiply(
        slots_.size(), target_bytes_, "compact D2H budget total overflows.");
    LOG(WARNING) << "Compact D2H mapping exceeds slot budget: budget_bytes="
                 << target_bytes_ << ", slot_bytes=" << slot_bytes
                 << ", total_slot_bytes=" << total_slot_bytes
                 << ", total_extra_hbm_bytes="
                 << total_slot_bytes - budgeted_bytes;
  }
}

}  // namespace xllm

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

#include "core/kv_cache/block/static_kv_cache_block_pool.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>

#include "core/kv_cache/layout/static_kv_cache_layout.h"

namespace xllm {
namespace {

constexpr size_t kUnavailableSlab = std::numeric_limits<size_t>::max();

size_t divide_up(size_t count, size_t divisor) {
  return count / divisor + static_cast<size_t>(count % divisor != 0);
}

}  // namespace

StaticKVCacheBlockPool::StaticKVCacheBlockPool(uint32_t num_slots,
                                               uint32_t linear_state_slots)
    : owners_(num_slots, -1),
      active_(num_slots, 0),
      slots_per_linear_state_(linear_state_slots) {
  CHECK_GT(linear_state_slots, 0u);
  CHECK_GT(num_slots, 1u);
  CHECK_LE(num_slots,
           static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
  CHECK_LE(static_cast<size_t>(num_slots),
           std::numeric_limits<size_t>::max() / linear_state_slots);
  linear_state_slots_.assign(
      static_cast<size_t>(num_slots) * slots_per_linear_state_, 0);
  free_slots_.reserve(num_slots - 1);
  // Pop smaller slot ids first.
  for (uint32_t slot = num_slots - 1; slot > 0; --slot) {
    free_slots_.emplace_back(static_cast<int32_t>(slot));
  }
}

StaticKVCacheBlockPool::StaticKVCacheBlockPool(
    uint32_t logical_capacity,
    const StaticKVCacheLayout& layout)
    : slots_per_linear_state_(0),
      slab_mode_(true),
      logical_capacity_(logical_capacity),
      slab_bytes_(layout.slab_bytes()),
      slice_bytes_(layout.slice_bytes()),
      bank_bytes_(layout.bank_bytes()),
      bank_count_(layout.bank_count()),
      full_layers_per_bank_(layout.full_layers_per_bank()),
      linear_layers_per_bank_(layout.linear_layers_per_bank()),
      linear_objects_per_state_(layout.linear_objects_per_state()),
      independent_linear_objects_(layout.draft_layer_count() > 0),
      full_read_padding_page_(
          static_cast<int32_t>(layout.full_read_padding_page())),
      object_capacities_{layout.objects_per_slab(KVCacheTensorRole::KEY),
                         layout.objects_per_slab(KVCacheTensorRole::CONV),
                         layout.objects_per_slab(KVCacheTensorRole::SSM)},
      num_reserved_slabs_(layout.num_reserved_slabs()),
      full_row_bytes_(layout.row_bytes(KVCacheTensorRole::KEY)),
      conv_row_bytes_(layout.row_bytes(KVCacheTensorRole::CONV)),
      ssm_row_bytes_(layout.row_bytes(KVCacheTensorRole::SSM)),
      checkpoint_stride_(layout.checkpoint_stride()),
      full_layers_(layout.full_layer_count()),
      linear_layers_(layout.linear_layer_count()),
      slabs_(layout.num_slabs()) {
  CHECK_GT(logical_capacity_, 1u);
  CHECK_LE(logical_capacity_,
           static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
  CHECK_GT(slabs_.size(), num_reserved_slabs_);
  CHECK_GT(full_layers_, 0u);
  CHECK_GT(linear_layers_, 0u);
  CHECK_GT(checkpoint_stride_, 0u);
  CHECK_LE(slabs_.size(),
           static_cast<size_t>(std::numeric_limits<uint32_t>::max()));
  // The layout already validates object geometry and arena byte arithmetic.
  CHECK_LE(full_layers_,
           std::numeric_limits<size_t>::max() / layout.full_pair_bytes());
  CHECK_LE(layout.ssm_bundle_bytes(),
           std::numeric_limits<size_t>::max() - conv_row_bytes_);
  CHECK_LE(linear_layers_,
           std::numeric_limits<size_t>::max() /
               (conv_row_bytes_ + layout.ssm_bundle_bytes()));
  CHECK_LE(linear_layers_, std::numeric_limits<size_t>::max() / 2);
  for (const SlabType type : {SlabType::KV, SlabType::CONV, SlabType::SSM}) {
    CHECK_LE(object_capacity(type),
             static_cast<size_t>(std::numeric_limits<uint32_t>::max()));
    const size_t view_bytes =
        type == SlabType::KV ? bank_bytes_ : layout.bytes_used();
    CHECK_LE((view_bytes - object_bytes(type)) / object_row_bytes(type),
             static_cast<size_t>(std::numeric_limits<int32_t>::max()));
  }
  free_slabs_.reserve(slabs_.size() - num_reserved_slabs_);
  for (size_t slab = slabs_.size(); slab > num_reserved_slabs_;) {
    --slab;
    free_slabs_.emplace_back(static_cast<uint32_t>(slab));
  }
  full_block_lease_epochs_.assign(logical_capacity_, 0);
  linear_state_lease_epochs_.assign(logical_capacity_, 0);
  for (const BlockType type : {BlockType::KV, BlockType::LINEAR}) {
    const size_t index = block_type_index(type);
    const size_t row_width =
        type == BlockType::KV ? full_layers_ : 2 * linear_layers_;
    CHECK_LE(logical_capacity_, std::numeric_limits<size_t>::max() / row_width);
    logical_states_[index].assign(logical_capacity_, 0);
    placement_hints_[index].assign(logical_capacity_, nullptr);
    physical_rows_[index].assign(logical_capacity_ * row_width, 0);
    free_ids_[index].reserve(logical_capacity_ - 1);
    for (uint32_t id = logical_capacity_ - 1; id > 0; --id) {
      free_ids_[index].emplace_back(static_cast<int32_t>(id));
    }
  }
}

size_t StaticKVCacheBlockPool::block_type_index(BlockType type) const {
  CHECK(type == BlockType::KV || type == BlockType::LINEAR);
  return type == BlockType::KV ? 0 : 1;
}

size_t StaticKVCacheBlockPool::object_bytes(SlabType type) const {
  switch (type) {
    case SlabType::KV:
      return 2 * full_layers_per_bank_ * full_row_bytes_;
    case SlabType::CONV:
      return conv_row_bytes_;
    case SlabType::SSM:
      return checkpoint_stride_ * ssm_row_bytes_;
    case SlabType::FREE:
      LOG(FATAL) << "A free slab has no object geometry";
      return 0;
  }
  LOG(FATAL) << "Unknown static cache slab type";
  return 0;
}

size_t StaticKVCacheBlockPool::object_row_bytes(SlabType type) const {
  return type == SlabType::SSM
             ? ssm_row_bytes_
             : (type == SlabType::KV ? full_row_bytes_ : conv_row_bytes_);
}

size_t StaticKVCacheBlockPool::object_capacity(SlabType type) const {
  return object_capacities_[static_cast<size_t>(type)];
}

uint32_t StaticKVCacheBlockPool::slab_for_row(int32_t row,
                                              SlabType type) const {
  return static_cast<uint32_t>(
      (static_cast<size_t>(row) * object_row_bytes(type) % bank_bytes_) /
      slice_bytes_);
}

int32_t StaticKVCacheBlockPool::object_row(uint32_t slab_id,
                                           uint32_t object,
                                           SlabType type,
                                           size_t bank) const {
  const size_t row_bytes = object_row_bytes(type);
  if (independent_linear_objects_ && type != SlabType::KV) {
    const size_t capacity = object_capacity(type) / bank_count_;
    bank += object / capacity;
    object = static_cast<uint32_t>(object % capacity);
  }
  const size_t bytes = bank * bank_bytes_ + slab_id * slice_bytes_;
  // Conv alignment depends on the absolute bank base, which need not be a
  // multiple of a Conv row. Keep the leading gap inside this bank's slice.
  const size_t row =
      divide_up(bytes, row_bytes) + object * (object_bytes(type) / row_bytes);
  CHECK_LE(row, static_cast<size_t>(std::numeric_limits<int32_t>::max()));
  return static_cast<int32_t>(row);
}

int32_t StaticKVCacheBlockPool::mirrored_row(int32_t row,
                                             SlabType type,
                                             size_t bank) const {
  const uint32_t slab_id = slab_for_row(row, type);
  const size_t first_row = static_cast<size_t>(object_row(slab_id, 0, type));
  const size_t rows_per_object = object_bytes(type) / object_row_bytes(type);
  const uint32_t object = static_cast<uint32_t>(
      (static_cast<size_t>(row) - first_row) / rows_per_object);
  return object_row(slab_id, object, type, bank);
}

void StaticKVCacheBlockPool::remove_available_slab(uint32_t slab_id) {
  Slab& slab = slabs_[slab_id];
  if (slab.available_index == kUnavailableSlab) {
    return;
  }
  std::vector<uint32_t>& available =
      available_slabs_[static_cast<size_t>(slab.type)];
  const uint32_t moved = available.back();
  available[slab.available_index] = moved;
  slabs_[moved].available_index = slab.available_index;
  available.pop_back();
  slab.available_index = kUnavailableSlab;
}

int32_t StaticKVCacheBlockPool::allocate_object(SlabType type,
                                                RequestPlacement* placement) {
  const size_t index = static_cast<size_t>(type);
  if (available_slabs_[index].empty()) {
    if (free_slabs_.empty()) {
      return -1;
    }
    const uint32_t slab_id = free_slabs_.back();
    free_slabs_.pop_back();
    Slab& slab = slabs_[slab_id];
    CHECK_EQ(slab.live_objects, 0u);
    slab.type = type;
    ++slab.generation;
    const size_t capacity = object_capacity(type);
    slab.free_objects.clear();
    slab.free_objects.reserve(capacity);
    for (size_t object = capacity; object > 0; --object) {
      slab.free_objects.emplace_back(static_cast<uint32_t>(object - 1));
    }
    available_object_counts_[index] += capacity;
    slab.available_index = available_slabs_[index].size();
    available_slabs_[index].emplace_back(slab_id);
  }
  uint32_t slab_id = available_slabs_[index].back();
  if (placement != nullptr) {
    const SlabAffinity& affinity = placement->recent_slabs[index];
    const Slab& preferred = slabs_[affinity.slab_id];
    if (preferred.type == type && preferred.generation == affinity.generation &&
        !preferred.free_objects.empty()) {
      slab_id = affinity.slab_id;
    } else if (slab_id == affinity.slab_id &&
               available_slabs_[index].size() > 1) {
      // Do not immediately return to a slab whose generation invalidated this
      // request's affinity. The global choice should make progress on another
      // available slab when one exists; the stale slab remains eligible after
      // the alternatives are exhausted.
      slab_id = available_slabs_[index][available_slabs_[index].size() - 2];
    }
    placement->recent_slabs[index] = {slab_id, slabs_[slab_id].generation};
  }
  Slab& slab = slabs_[slab_id];
  const uint32_t object = slab.free_objects.back();
  slab.free_objects.pop_back();
  --available_object_counts_[index];
  ++slab.live_objects;
  if (slab.free_objects.empty()) {
    remove_available_slab(slab_id);
  }
  return object_row(slab_id, object, type);
}

void StaticKVCacheBlockPool::release_object(int32_t row, SlabType type) {
  const uint32_t slab_id = slab_for_row(row, type);
  CHECK_GE(slab_id, num_reserved_slabs_);
  Slab& slab = slabs_[slab_id];
  CHECK(slab.type == type);
  CHECK_GT(slab.live_objects, 0u);
  const size_t bank =
      independent_linear_objects_ && type != SlabType::KV
          ? static_cast<size_t>(row) * object_row_bytes(type) / bank_bytes_
          : 0;
  const size_t first_row =
      static_cast<size_t>(object_row(slab_id, 0, type, bank));
  const size_t row_offset = static_cast<size_t>(row) - first_row;
  const size_t rows_per_object = object_bytes(type) / object_row_bytes(type);
  CHECK_EQ(row_offset % rows_per_object, 0u);
  const bool was_full = slab.free_objects.empty();
  const size_t bank_objects =
      independent_linear_objects_ && type != SlabType::KV
          ? bank * (object_capacity(type) / bank_count_)
          : 0;
  slab.free_objects.emplace_back(
      static_cast<uint32_t>(bank_objects + row_offset / rows_per_object));
  ++available_object_counts_[static_cast<size_t>(type)];
  --slab.live_objects;
  if (slab.live_objects == 0) {
    remove_available_slab(slab_id);
    available_object_counts_[static_cast<size_t>(type)] -=
        slab.free_objects.size();
    slab.type = SlabType::FREE;
    slab.free_objects.clear();
    free_slabs_.emplace_back(slab_id);
    return;
  }
  if (was_full) {
    std::vector<uint32_t>& available =
        available_slabs_[static_cast<size_t>(type)];
    slab.available_index = available.size();
    available.emplace_back(slab_id);
  }
}

std::vector<int32_t> StaticKVCacheBlockPool::allocate_slab_blocks(
    size_t num_blocks,
    BlockType type,
    const void* placement_hint) {
  const size_t index = block_type_index(type);
  if (num_blocks > free_ids_[index].size()) {
    return {};
  }
  const size_t width =
      type == BlockType::KV ? 1 : 2 * linear_objects_per_state_;
  RequestPlacement placement;
  if (placement_hint != nullptr) {
    const auto existing = request_placements_.find(placement_hint);
    if (existing != request_placements_.end()) {
      placement = existing->second;
    }
  }
  std::vector<int32_t> rows;
  rows.reserve(num_blocks * width);
  for (size_t object = 0; object < num_blocks * width; ++object) {
    const SlabType slab_type =
        type == BlockType::KV
            ? SlabType::KV
            : (object % 2 == 0 ? SlabType::CONV : SlabType::SSM);
    const int32_t row = allocate_object(
        slab_type, placement_hint == nullptr ? nullptr : &placement);
    if (row < 0) {
      for (size_t allocated = 0; allocated < rows.size(); ++allocated) {
        const SlabType allocated_type =
            type == BlockType::KV
                ? SlabType::KV
                : (allocated % 2 == 0 ? SlabType::CONV : SlabType::SSM);
        release_object(rows[allocated], allocated_type);
      }
      return {};
    }
    rows.emplace_back(row);
  }
  const size_t stored_width =
      type == BlockType::KV ? full_layers_ : 2 * linear_layers_;
  std::vector<int32_t> blocks;
  blocks.reserve(num_blocks);
  for (size_t block = 0; block < num_blocks; ++block) {
    const int32_t id = free_ids_[index].back();
    free_ids_[index].pop_back();
    logical_states_[index][id] = 2;
    placement_hints_[index][id] = placement_hint;
    auto& epochs = type == BlockType::KV ? full_block_lease_epochs_
                                         : linear_state_lease_epochs_;
    CHECK_LT(epochs[id], std::numeric_limits<uint64_t>::max());
    ++epochs[id];
    blocks.emplace_back(id);
    int32_t* destination = physical_rows_[index].data() + id * stored_width;
    if (type == BlockType::KV) {
      std::fill_n(destination, full_layers_, rows[block]);
      continue;
    }
    if (independent_linear_objects_) {
      std::copy_n(rows.data() + block * width, width, destination);
      continue;
    }
    for (size_t layer = 0; layer < linear_layers_; ++layer) {
      const size_t bank = layer / linear_layers_per_bank_;
      const size_t source =
          block * width + 2 * (layer % linear_layers_per_bank_);
      destination[2 * layer] = mirrored_row(rows[source], SlabType::CONV, bank);
      destination[2 * layer + 1] =
          mirrored_row(rows[source + 1], SlabType::SSM, bank);
    }
  }
  const size_t bytes = num_blocks * slots_per_block(type);
  if (placement_hint != nullptr && num_blocks > 0) {
    placement.owned_blocks += num_blocks;
    request_placements_.insert_or_assign(placement_hint, placement);
  }
  owned_bytes_ += bytes;
  num_used_slots_ += bytes;
  return blocks;
}

size_t StaticKVCacheBlockPool::slots_per_block(BlockType type) const {
  if (slab_mode_) {
    block_type_index(type);
    return type == BlockType::KV
               ? bank_count_ * object_bytes(SlabType::KV)
               : linear_layers_ *
                     (conv_row_bytes_ + object_bytes(SlabType::SSM));
  }
  switch (type) {
    case BlockType::KV:
      return 1;
    case BlockType::LINEAR:
      return slots_per_linear_state_;
    default:
      LOG(FATAL) << "Unsupported shared static cache block type: "
                 << static_cast<int32_t>(type);
      return 0;
  }
}

std::vector<int32_t> StaticKVCacheBlockPool::allocate(
    size_t num_blocks,
    BlockType type,
    const void* placement_hint) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (slab_mode_) {
    return allocate_slab_blocks(num_blocks, type, placement_hint);
  }
  const size_t cost = slots_per_block(type);
  if (num_blocks > free_slots_.size() / cost) {
    return {};
  }
  std::vector<int32_t> blocks;
  blocks.reserve(num_blocks);
  for (size_t block = 0; block < num_blocks; ++block) {
    const int32_t block_id = free_slots_.back();
    for (size_t group = 0; group < cost; ++group) {
      const int32_t slot = free_slots_.back();
      free_slots_.pop_back();
      owners_[slot] = static_cast<int32_t>(type);
      if (type == BlockType::LINEAR) {
        linear_state_slots_[static_cast<size_t>(block_id) * cost + group] =
            slot;
      }
    }
    active_[block_id] = 1;
    blocks.emplace_back(block_id);
  }
  num_used_slots_ += num_blocks * cost;
  return blocks;
}

void StaticKVCacheBlockPool::release_slot(int32_t slot) {
  owners_[slot] = -1;
  free_slots_.emplace_back(slot);
}

void StaticKVCacheBlockPool::release(int32_t block_id, BlockType type) {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK_GT(block_id, 0);
  if (slab_mode_) {
    CHECK_LT(static_cast<uint32_t>(block_id), logical_capacity_);
    const size_t index = block_type_index(type);
    CHECK_NE(logical_states_[index][block_id], 0u);
    const size_t cost = slots_per_block(type);
    num_used_slots_ -= logical_states_[index][block_id] == 2 ? cost : 0;
    owned_bytes_ -= cost;
    const size_t width =
        type == BlockType::KV ? full_layers_ : 2 * linear_layers_;
    const size_t begin = static_cast<size_t>(block_id) * width;
    const size_t objects =
        type == BlockType::KV ? 1 : 2 * linear_objects_per_state_;
    for (size_t object = 0; object < objects; ++object) {
      const SlabType slab_type =
          type == BlockType::KV
              ? SlabType::KV
              : (object % 2 == 0 ? SlabType::CONV : SlabType::SSM);
      release_object(physical_rows_[index][begin + object], slab_type);
    }
    std::fill_n(physical_rows_[index].begin() + begin, width, 0);
    logical_states_[index][block_id] = 0;
    const void* placement_hint = placement_hints_[index][block_id];
    if (placement_hint != nullptr) {
      auto placement = request_placements_.find(placement_hint);
      CHECK(placement != request_placements_.end());
      CHECK_GT(placement->second.owned_blocks, 0u);
      if (--placement->second.owned_blocks == 0) {
        request_placements_.erase(placement);
      }
      placement_hints_[index][block_id] = nullptr;
    }
    free_ids_[index].emplace_back(block_id);
    return;
  }
  CHECK_LT(static_cast<size_t>(block_id), owners_.size());
  CHECK_EQ(owners_[block_id], static_cast<int32_t>(type));
  const size_t cost = slots_per_block(type);
  num_used_slots_ -= active_[block_id] * cost;
  active_[block_id] = 0;
  if (type != BlockType::LINEAR) {
    release_slot(block_id);
    return;
  }
  const size_t row = static_cast<size_t>(block_id) * cost;
  for (size_t group = 0; group < cost; ++group) {
    const int32_t slot = linear_state_slots_[row + group];
    CHECK_EQ(owners_[slot], static_cast<int32_t>(type));
    release_slot(slot);
    linear_state_slots_[row + group] = 0;
  }
}

void StaticKVCacheBlockPool::mark_active(int32_t block_id, BlockType type) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (slab_mode_) {
    CHECK_GT(block_id, 0);
    CHECK_LT(static_cast<uint32_t>(block_id), logical_capacity_);
    const size_t index = block_type_index(type);
    CHECK_NE(logical_states_[index][block_id], 0u);
    num_used_slots_ +=
        logical_states_[index][block_id] == 1 ? slots_per_block(type) : 0;
    logical_states_[index][block_id] = 2;
    return;
  }
  CHECK_EQ(owners_[block_id], static_cast<int32_t>(type));
  num_used_slots_ += active_[block_id] == 0 ? slots_per_block(type) : 0;
  active_[block_id] = 1;
}

void StaticKVCacheBlockPool::mark_inactive(int32_t block_id, BlockType type) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (slab_mode_) {
    CHECK_GT(block_id, 0);
    CHECK_LT(static_cast<uint32_t>(block_id), logical_capacity_);
    const size_t index = block_type_index(type);
    CHECK_NE(logical_states_[index][block_id], 0u);
    num_used_slots_ -=
        logical_states_[index][block_id] == 2 ? slots_per_block(type) : 0;
    logical_states_[index][block_id] = 1;
    return;
  }
  CHECK_EQ(owners_[block_id], static_cast<int32_t>(type));
  num_used_slots_ -= active_[block_id] * slots_per_block(type);
  active_[block_id] = 0;
}

void StaticKVCacheBlockPool::append_linear_state_slots(
    int32_t state_id,
    std::vector<int32_t>* slots) const {
  CHECK(slots != nullptr);
  if (slab_mode_) {
    append_linear_state_rows(state_id, slots);
    return;
  }
  if (state_id < 0) {
    slots->insert(slots->end(), slots_per_linear_state_, state_id);
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK_LT(static_cast<size_t>(state_id), owners_.size());
  const auto row = linear_state_slots_.begin() +
                   static_cast<size_t>(state_id) * slots_per_linear_state_;
  CHECK_EQ(*row, state_id) << "Slot " << state_id
                           << " does not identify a recurrent state";
  slots->insert(slots->end(), row, row + slots_per_linear_state_);
}

void StaticKVCacheBlockPool::append_rows(
    int32_t block_id,
    BlockType type,
    std::vector<int32_t>* rows,
    std::vector<uint64_t>* lease_epochs) const {
  CHECK(rows != nullptr);
  CHECK(slab_mode_);
  const size_t width =
      type == BlockType::KV ? full_layers_ : 2 * linear_layers_;
  if (block_id < 0) {
    rows->insert(rows->end(), width, block_id);
    if (lease_epochs != nullptr) {
      lease_epochs->emplace_back(0);
    }
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK_LT(static_cast<uint32_t>(block_id), logical_capacity_);
  const size_t index = block_type_index(type);
  CHECK(block_id == 0 || logical_states_[index][block_id] != 0);
  const auto begin = physical_rows_[index].begin() + block_id * width;
  rows->insert(rows->end(), begin, begin + width);
  if (lease_epochs != nullptr) {
    lease_epochs->emplace_back(full_block_lease_epochs_[block_id]);
  }
}

void StaticKVCacheBlockPool::append_full_block_rows(
    int32_t block_id,
    std::vector<int32_t>* rows,
    std::vector<uint64_t>* lease_epochs) const {
  append_rows(block_id, BlockType::KV, rows, lease_epochs);
}

void StaticKVCacheBlockPool::append_linear_state_rows(
    int32_t state_id,
    std::vector<int32_t>* rows) const {
  append_rows(state_id, BlockType::LINEAR, rows);
}

void StaticKVCacheBlockPool::append_changed_rows(
    BlockType type,
    Slice<Block> blocks,
    const std::vector<uint64_t>& known_epochs,
    std::vector<int32_t>* rows,
    std::vector<uint64_t>* epochs) const {
  CHECK(slab_mode_);
  CHECK(rows != nullptr);
  CHECK(epochs != nullptr);
  CHECK_EQ(known_epochs.size(), logical_capacity_);
  const size_t index = block_type_index(type);
  const size_t width =
      type == BlockType::KV ? full_layers_ : 2 * linear_layers_;
  std::lock_guard<std::mutex> lock(mutex_);
  const auto& allocation_epochs = type == BlockType::KV
                                      ? full_block_lease_epochs_
                                      : linear_state_lease_epochs_;
  for (const Block& block : blocks) {
    if (!block.is_valid()) {
      continue;
    }
    const int32_t id = block.id();
    CHECK_GE(id, 0);
    CHECK_LT(static_cast<uint32_t>(id), logical_capacity_);
    CHECK(id == 0 || logical_states_[index][id] != 0);
    if (known_epochs[id] == allocation_epochs[id]) {
      continue;
    }
    rows->emplace_back(id);
    const auto begin = physical_rows_[index].begin() + id * width;
    rows->insert(rows->end(), begin, begin + width);
    epochs->emplace_back(allocation_epochs[id]);
  }
}

int32_t StaticKVCacheBlockPool::final_full_page(int32_t logical_id) const {
  CHECK(slab_mode_);
  if (logical_id <= 0) {
    return logical_id;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK_LT(static_cast<uint32_t>(logical_id), logical_capacity_);
  CHECK_NE(logical_states_[0][logical_id], 0u);
  return physical_rows_[0][static_cast<size_t>(logical_id) * full_layers_];
}

void StaticKVCacheBlockPool::append_final_full_pages(
    Slice<Block> blocks,
    std::vector<int32_t>* pages) const {
  CHECK(slab_mode_);
  CHECK(pages != nullptr);
  std::lock_guard<std::mutex> lock(mutex_);
  pages->reserve(pages->size() + blocks.size());
  for (const Block& block : blocks) {
    const int32_t id = block.id();
    if (id <= 0) {
      pages->emplace_back(id);
      continue;
    }
    CHECK_LT(static_cast<uint32_t>(id), logical_capacity_);
    CHECK_NE(logical_states_[0][id], 0u);
    pages->emplace_back(
        physical_rows_[0][static_cast<size_t>(id) * full_layers_]);
  }
}

int32_t StaticKVCacheBlockPool::full_read_padding_page() const {
  CHECK(slab_mode_);
  return full_read_padding_page_;
}

size_t StaticKVCacheBlockPool::num_slots() const {
  return slab_mode_ ? logical_capacity_ : owners_.size();
}

size_t StaticKVCacheBlockPool::num_total_slots() const {
  return slab_mode_ ? (slabs_.size() - num_reserved_slabs_) * slab_bytes_
                    : owners_.size() - 1;
}

size_t StaticKVCacheBlockPool::num_free_slots() const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (slab_mode_) {
    return num_total_slots() - owned_bytes_;
  }
  return free_slots_.size();
}

size_t StaticKVCacheBlockPool::num_used_slots() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return num_used_slots_;
}

size_t StaticKVCacheBlockPool::available_objects(SlabType type) const {
  return available_object_counts_[static_cast<size_t>(type)];
}

size_t StaticKVCacheBlockPool::reclaimable_slab_count(
    BlockType type,
    const std::vector<int32_t>& block_ids) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!slab_mode_) {
    return 0;
  }
  const size_t index = block_type_index(type);
  const size_t width =
      type == BlockType::KV ? full_layers_ : 2 * linear_layers_;
  std::unordered_map<uint32_t, size_t> candidate_objects;
  for (const int32_t id : block_ids) {
    if (id <= 0 || static_cast<uint32_t>(id) >= logical_capacity_ ||
        logical_states_[index][id] != 1) {
      continue;
    }
    const size_t objects =
        type == BlockType::KV ? 1 : 2 * linear_objects_per_state_;
    for (size_t object = 0; object < objects; ++object) {
      const SlabType slab_type =
          type == BlockType::KV
              ? SlabType::KV
              : (object % 2 == 0 ? SlabType::CONV : SlabType::SSM);
      const int32_t row = physical_rows_[index][id * width + object];
      const uint32_t slab_id = slab_for_row(row, slab_type);
      ++candidate_objects[slab_id];
    }
  }
  size_t reclaimable = 0;
  for (const auto& [slab_id, objects] : candidate_objects) {
    reclaimable += objects == slabs_[slab_id].live_objects;
  }
  return reclaimable;
}

size_t StaticKVCacheBlockPool::num_free_blocks(BlockType type) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!slab_mode_) {
    return free_slots_.size() / slots_per_block(type);
  }
  const size_t index = block_type_index(type);
  if (type == BlockType::KV) {
    const size_t objects = available_objects(SlabType::KV) +
                           free_slabs_.size() * object_capacity(SlabType::KV);
    return std::min(free_ids_[index].size(), objects);
  }
  const size_t conv_available = available_objects(SlabType::CONV);
  const size_t ssm_available = available_objects(SlabType::SSM);
  size_t lower = 0;
  size_t upper = free_ids_[index].size();
  while (lower < upper) {
    const size_t count = lower + (upper - lower + 1) / 2;
    const size_t objects = count * linear_objects_per_state_;
    const size_t conv_needed =
        objects > conv_available ? objects - conv_available : 0;
    const size_t ssm_needed =
        objects > ssm_available ? objects - ssm_available : 0;
    const size_t slabs_needed =
        divide_up(conv_needed, object_capacity(SlabType::CONV)) +
        divide_up(ssm_needed, object_capacity(SlabType::SSM));
    if (slabs_needed <= free_slabs_.size()) {
      lower = count;
    } else {
      upper = count - 1;
    }
  }
  return lower;
}

}  // namespace xllm

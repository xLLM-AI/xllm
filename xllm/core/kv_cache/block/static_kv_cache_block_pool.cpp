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

#include <limits>

namespace xllm {

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

size_t StaticKVCacheBlockPool::slots_per_block(BlockType type) const {
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

std::vector<int32_t> StaticKVCacheBlockPool::allocate(size_t num_blocks,
                                                      BlockType type) {
  std::lock_guard<std::mutex> lock(mutex_);
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
  CHECK_EQ(owners_[block_id], static_cast<int32_t>(type));
  num_used_slots_ += active_[block_id] == 0 ? slots_per_block(type) : 0;
  active_[block_id] = 1;
}

void StaticKVCacheBlockPool::mark_inactive(int32_t block_id, BlockType type) {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK_EQ(owners_[block_id], static_cast<int32_t>(type));
  num_used_slots_ -= active_[block_id] * slots_per_block(type);
  active_[block_id] = 0;
}

void StaticKVCacheBlockPool::append_linear_state_slots(
    int32_t state_id,
    std::vector<int32_t>* slots) const {
  CHECK(slots != nullptr);
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

size_t StaticKVCacheBlockPool::num_free_slots() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return free_slots_.size();
}

size_t StaticKVCacheBlockPool::num_used_slots() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return num_used_slots_;
}

}  // namespace xllm

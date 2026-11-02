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
#include <iterator>
#include <limits>

namespace xllm {

StaticKVCacheBlockPool::StaticKVCacheBlockPool(uint32_t num_units,
                                               uint32_t full_block_units,
                                               uint32_t linear_state_units)
    : owners_(num_units, -1),
      active_(num_units, 0),
      full_block_units_(full_block_units),
      linear_state_units_(linear_state_units),
      padding_units_(std::max(full_block_units, linear_state_units)) {
  CHECK_GT(full_block_units, 0u);
  CHECK_GT(linear_state_units, 0u);
  CHECK_GT(num_units, padding_units_);
  CHECK_LE(num_units,
           static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
  num_free_units_ = num_units - padding_units_;
  free_extents_.emplace(static_cast<int32_t>(padding_units_), num_free_units_);
}

size_t StaticKVCacheBlockPool::block_units(BlockType type) const {
  switch (type) {
    case BlockType::KV:
      return full_block_units_;
    case BlockType::LINEAR:
      return linear_state_units_;
    default:
      LOG(FATAL) << "Unsupported shared static cache block type: "
                 << static_cast<int32_t>(type);
      return 0;
  }
}

std::vector<int32_t> StaticKVCacheBlockPool::allocate(size_t num_blocks,
                                                      BlockType type) {
  std::lock_guard<std::mutex> lock(mutex_);
  const size_t units = block_units(type);
  size_t available_blocks = 0;
  for (const auto& [start, length] : free_extents_) {
    available_blocks += length / units;
    if (available_blocks >= num_blocks) {
      break;
    }
  }
  if (num_blocks > available_blocks) {
    return {};
  }
  std::vector<int32_t> blocks;
  blocks.reserve(num_blocks);
  auto it = free_extents_.begin();
  while (blocks.size() < num_blocks) {
    if (it->second < units) {
      ++it;
      continue;
    }
    const int32_t start = it->first;
    const size_t count =
        std::min(it->second / units, num_blocks - blocks.size());
    const size_t consumed_units = count * units;
    const size_t remaining_units = it->second - consumed_units;
    it = free_extents_.erase(it);
    if (remaining_units > 0) {
      free_extents_.emplace(start + static_cast<int32_t>(consumed_units),
                            remaining_units);
    }
    for (size_t index = 0; index < count; ++index) {
      const int32_t block_id = start + static_cast<int32_t>(index * units);
      owners_[block_id] = static_cast<int32_t>(type);
      active_[block_id] = 1;
      blocks.emplace_back(block_id);
    }
  }
  num_free_units_ -= num_blocks * units;
  num_used_blocks_ += num_blocks * units;
  return blocks;
}

void StaticKVCacheBlockPool::release(int32_t block_id, BlockType type) {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK_GT(block_id, 0);
  CHECK_LT(static_cast<size_t>(block_id), owners_.size());
  CHECK_EQ(owners_[block_id], static_cast<int32_t>(type));
  const size_t units = block_units(type);
  num_used_blocks_ -= active_[block_id] * units;
  active_[block_id] = 0;
  owners_[block_id] = -1;
  int32_t start = block_id;
  size_t length = units;
  auto next = free_extents_.lower_bound(start);
  if (next != free_extents_.begin()) {
    const auto previous = std::prev(next);
    if (previous->first + static_cast<int32_t>(previous->second) == start) {
      start = previous->first;
      length += previous->second;
      free_extents_.erase(previous);
    }
  }
  if (next != free_extents_.end() &&
      start + static_cast<int32_t>(length) == next->first) {
    length += next->second;
    free_extents_.erase(next);
  }
  free_extents_.emplace(start, length);
  num_free_units_ += units;
}

void StaticKVCacheBlockPool::mark_active(int32_t block_id, BlockType type) {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK_EQ(owners_[block_id], static_cast<int32_t>(type));
  num_used_blocks_ += active_[block_id] == 0 ? block_units(type) : 0;
  active_[block_id] = 1;
}

void StaticKVCacheBlockPool::mark_inactive(int32_t block_id, BlockType type) {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK_EQ(owners_[block_id], static_cast<int32_t>(type));
  num_used_blocks_ -= active_[block_id] * block_units(type);
  active_[block_id] = 0;
}

size_t StaticKVCacheBlockPool::num_free_blocks() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return num_free_units_;
}

size_t StaticKVCacheBlockPool::num_used_blocks() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return num_used_blocks_;
}

}  // namespace xllm

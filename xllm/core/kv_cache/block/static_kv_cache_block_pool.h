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

#include <cstdint>
#include <map>
#include <mutex>
#include <vector>

#include "core/kv_cache/block/block.h"

namespace xllm {

// Owns the physical unit space shared by Full KV blocks and linear states.
// Each block id starts a contiguous extent whose cost is group-specific.
// Device storage indexes the same units; final Block release returns an extent.
class StaticKVCacheBlockPool final {
 public:
  explicit StaticKVCacheBlockPool(uint32_t num_units,
                                  uint32_t full_block_units = 1,
                                  uint32_t linear_state_units = 1);

  std::vector<int32_t> allocate(size_t num_blocks, BlockType type);
  void release(int32_t block_id, BlockType type);
  void mark_active(int32_t block_id, BlockType type);
  void mark_inactive(int32_t block_id, BlockType type);

  // Capacity and occupancy are physical units, rather than logical blocks.
  size_t num_total_blocks() const { return owners_.size() - padding_units_; }
  size_t num_free_blocks() const;
  size_t num_used_blocks() const;
  size_t num_units() const { return owners_.size(); }
  size_t block_units(BlockType type) const;

 private:
  mutable std::mutex mutex_;
  std::map<int32_t, size_t> free_extents_;
  std::vector<int32_t> owners_;
  std::vector<uint8_t> active_;
  size_t full_block_units_;
  size_t linear_state_units_;
  size_t padding_units_;
  size_t num_free_units_ = 0;
  size_t num_used_blocks_ = 0;
};

}  // namespace xllm

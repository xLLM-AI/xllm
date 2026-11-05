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
#include <mutex>
#include <vector>

#include "core/kv_cache/block/block.h"

namespace xllm {

// Owns the fixed-size slots shared by Full KV blocks and recurrent states.
// A Full block costs one slot and is identified by it. A recurrent state costs
// one slot per Linear group; its id is the group-0 slot, and the other slots
// are recorded for execution. Slot 0 is padding for every group. Any free slot
// can serve either type, so free capacity is never fragmented.
class StaticKVCacheBlockPool final {
 public:
  StaticKVCacheBlockPool(uint32_t num_slots, uint32_t linear_state_slots);

  std::vector<int32_t> allocate(size_t num_blocks, BlockType type);
  void release(int32_t block_id, BlockType type);
  void mark_active(int32_t block_id, BlockType type);
  void mark_inactive(int32_t block_id, BlockType type);

  // Physical slots of a recurrent state in Linear group order. Negative ids
  // stay sentinels in every group.
  void append_linear_state_slots(int32_t state_id,
                                 std::vector<int32_t>* slots) const;

  size_t num_slots() const { return owners_.size(); }
  size_t num_total_slots() const { return owners_.size() - 1; }
  size_t num_free_slots() const;
  size_t num_used_slots() const;
  size_t slots_per_block(BlockType type) const;

 private:
  void release_slot(int32_t slot);

  mutable std::mutex mutex_;
  std::vector<int32_t> free_slots_;
  // Block type owning each slot, or -1 when free.
  std::vector<int32_t> owners_;
  // Indexed by block id; accounts the whole block once.
  std::vector<uint8_t> active_;
  // Row per recurrent state id: [num_slots, linear_state_slots].
  std::vector<int32_t> linear_state_slots_;
  size_t slots_per_linear_state_;
  size_t num_used_slots_ = 0;
};

}  // namespace xllm

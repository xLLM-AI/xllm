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

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "core/kv_cache/block/block.h"

namespace xllm {

class StaticKVCacheLayout;

// Owns stable logical IDs backed by a shared physical slab pool. Each slab
// packs Full K/V pairs, Conv rows, or contiguous SSM checkpoint bundles; an
// empty slab can change type. Leading slabs isolate writable graph scratch
// and immutable Full read padding from allocatable objects.
// The legacy constructor retains coarse slots for existing callers.
class StaticKVCacheBlockPool final {
 public:
  StaticKVCacheBlockPool(uint32_t num_slots, uint32_t linear_state_slots);
  StaticKVCacheBlockPool(uint32_t logical_capacity,
                         const StaticKVCacheLayout& layout);

  // The optional non-owning request identity affects placement only.
  std::vector<int32_t> allocate(size_t num_blocks,
                                BlockType type,
                                const void* placement_hint = nullptr);
  void release(int32_t block_id, BlockType type);
  void mark_active(int32_t block_id, BlockType type);
  void mark_inactive(int32_t block_id, BlockType type);

  // Physical slots of a recurrent state in Linear group order. Negative ids
  // stay sentinels in every group.
  void append_linear_state_slots(int32_t state_id,
                                 std::vector<int32_t>* slots) const;
  // Full-layer order; KEY and VALUE use the same row through adjacent views.
  void append_full_block_rows(
      int32_t block_id,
      std::vector<int32_t>* rows,
      std::vector<uint64_t>* lease_epochs = nullptr) const;
  // Linear-layer order: Conv row, then the first SSM checkpoint row.
  void append_linear_state_rows(int32_t state_id,
                                std::vector<int32_t>* rows) const;

  size_t num_slots() const;
  // In slab mode these resource-accounting methods return payload bytes.
  size_t num_total_slots() const;
  size_t num_free_slots() const;
  size_t num_used_slots() const;
  size_t slots_per_block(BlockType type) const;
  size_t num_free_blocks(BlockType type) const;
  // Scores distinct cache-only candidates without changing ownership.
  size_t reclaimable_slab_count(BlockType type,
                                const std::vector<int32_t>& block_ids) const;
  bool uses_slabs() const { return slab_mode_; }

 private:
  friend class StaticKVCacheBlockPoolTestPeer;

  enum class SlabType : uint8_t { KV = 0, CONV = 1, SSM = 2, FREE = 3 };

  struct SlabAffinity {
    uint32_t slab_id = 0;
    uint64_t generation = 0;
  };

  struct RequestPlacement {
    std::array<SlabAffinity, 3> recent_slabs;
    size_t owned_blocks = 0;
  };

  struct Slab {
    SlabType type = SlabType::FREE;
    size_t live_objects = 0;
    size_t available_index = 0;
    uint64_t generation = 0;
    std::vector<uint32_t> free_objects;
  };

  size_t block_type_index(BlockType type) const;
  size_t object_bytes(SlabType type) const;
  size_t object_row_bytes(SlabType type) const;
  size_t available_objects(SlabType type) const;
  int32_t allocate_object(SlabType type, RequestPlacement* placement);
  void release_object(int32_t row, SlabType type);
  void remove_available_slab(uint32_t slab_id);
  std::vector<int32_t> allocate_slab_blocks(size_t num_blocks,
                                            BlockType type,
                                            const void* placement_hint);
  void append_rows(int32_t block_id,
                   BlockType type,
                   std::vector<int32_t>* rows,
                   std::vector<uint64_t>* lease_epochs = nullptr) const;

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

  bool slab_mode_ = false;
  uint32_t logical_capacity_ = 0;
  size_t slab_bytes_ = 0;
  size_t num_reserved_slabs_ = 0;
  size_t full_row_bytes_ = 0;
  size_t conv_row_bytes_ = 0;
  size_t ssm_row_bytes_ = 0;
  size_t checkpoint_stride_ = 0;
  size_t full_layers_ = 0;
  size_t linear_layers_ = 0;
  size_t owned_bytes_ = 0;
  std::vector<Slab> slabs_;
  std::vector<uint32_t> free_slabs_;
  std::array<std::vector<uint32_t>, 3> available_slabs_;
  std::array<size_t, 3> available_object_counts_{};
  std::array<std::vector<int32_t>, 2> free_ids_;
  // 0 is free, 1 is owned by prefix cache, and 2 is active.
  std::array<std::vector<uint8_t>, 2> logical_states_;
  std::array<std::vector<int32_t>, 2> physical_rows_;
  // Full logical ID allocation epochs survive physical row and type reuse.
  std::vector<uint64_t> full_block_lease_epochs_;
  std::array<std::vector<const void*>, 2> placement_hints_;
  std::unordered_map<const void*, RequestPlacement> request_placements_;
};

}  // namespace xllm

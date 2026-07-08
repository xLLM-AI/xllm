/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <cstdint>
#include <list>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/macros.h"
#include "framework/block/block.h"
#include "framework/block/single_block_manager.h"
#include "util/hash_util.h"

namespace xllm {

class Sequence;
struct LinearStateCacheOp;

// Aggregated batch state produced by LinearStatePrefixCache::resolve_cache_ops
// and consumed by LinearStatePrefixCache::commit_reservations after this
// step's forward finishes. Saves are handled entirely by promotion (no extra
// device copy), mirroring how KV-cache blocks are inserted into the prefix
// cache without a copy.
class LinearStateCheckpointReservations final {
 public:
  LinearStateCheckpointReservations() = default;
  MOVE_ONLY(LinearStateCheckpointReservations);

  // A live slot the scheduler decided to "promote" into a committed
  // checkpoint: the sequence keeps a fresh replacement live slot and the
  // existing live slot is grafted directly into the prefix cache after this
  // step's forward writes its end-of-step contents.
  class Promotion final {
   public:
    Promotion(const XXH3Key& hash,
              Sequence* sequence,
              int32_t live_slot_id,
              Block replacement_slot);
    MOVE_ONLY(Promotion);

    const XXH3Key& hash() const { return hash_; }
    Sequence* sequence() const { return sequence_; }
    int32_t live_slot_id() const { return live_slot_id_; }
    Block take_replacement_slot() { return std::move(replacement_slot_); }

   private:
    XXH3Key hash_;
    Sequence* sequence_ = nullptr;
    int32_t live_slot_id_ = -1;
    Block replacement_slot_;
  };

  int32_t dp_rank = -1;
  // Pins checkpoint slots matched for restore until the worker has copied them
  // into sequence-owned live slots.
  std::vector<Block> restore_pins;
  std::vector<Promotion> promotions;
};

// Scheduler-side prefix cache for Qwen3.5 GDN linear-state checkpoints,
// mirroring the KV-cache architecture (a free-list slot pool + an LRU prefix
// index sharing one id space). All physical slots [0, num_slots) live here:
//   - a slot held by a running sequence is "live" (its Block ref_count >= 1);
//   - a slot held by a reservation is a pending checkpoint and is invisible to
//     prefix-cache matching;
//   - a slot pinned by a prefix-cache entry is a committed checkpoint
//     (ref_count == 1, held by this cache), evictable via LRU;
//   - everything else is free.
//
// Slot 0 is reserved as the padding slot (kPaddingLinearStateId) by the
// underlying SingleBlockManager and is never handed out.
//
// Unlike KV blocks, GDN recurrent state is overwritten in place and cannot be
// shared, so a live slot always has exactly one owner and restore/save still
// require a device copy on the worker. This cache only owns the scheduler-side
// bookkeeping: which slots are live, which hold a checkpoint, and the LRU
// order used to reclaim checkpoints under pressure. Eviction is purely local
// here -- there is no cross-process notification, because the worker is a
// thin executor that copies between the (src, dst) slots this cache dictates.
class LinearStatePrefixCache final {
 public:
  explicit LinearStatePrefixCache(int32_t num_slots);

  LinearStatePrefixCache(const LinearStatePrefixCache&) = delete;
  LinearStatePrefixCache& operator=(const LinearStatePrefixCache&) = delete;

  // Allocate a live slot for a running sequence. Reclaims the least-recently
  // used checkpoint when no slot is free. Returns an invalid Block only when
  // every slot is already live or pinned (no checkpoint left to reclaim).
  // Mirrors BlockManagerImpl::allocate(1) in the KV path.
  Block allocate_live_slot();

  // Match the checkpoint slot for a prefix hash, refreshing its LRU position.
  // The returned Block pins the checkpoint slot until the caller drops it.
  Block match(const XXH3Key& prefix_hash);

  // Insert a checkpoint that already contains saved recurrent state. Used when
  // a sequence-owned live slot is promoted directly into the committed cache.
  int32_t insert_checkpoint(const XXH3Key& prefix_hash, Block checkpoint_block);

  // True if a checkpoint exists for the given hash (no LRU refresh).
  bool contains(const XXH3Key& prefix_hash) const;

  // Resolve this batch's restore/save copy plan. Saves are handled by
  // promotion (no worker-side copy); restores are matched against the
  // committed cache and produce restore_src_slot_id values for the worker.
  // The returned reservation carries no dp_rank; that pool-level routing tag
  // is stamped by BlockManagerPool, which owns the per-dp-rank cache lookup.
  LinearStateCheckpointReservations resolve_cache_ops(
      std::vector<LinearStateCacheOp>* cache_ops,
      const std::vector<Sequence*>& sequences = {});

  // Apply this batch's promotions into the committed cache after this step's
  // forward finishes (so the frozen slot's contents are stable).
  void commit_reservations(LinearStateCheckpointReservations&& reservations);

 private:
  struct CacheEntry {
    Block block;  // pins the slot so the free list cannot hand it out
    std::list<XXH3Key>::iterator lru_it;
  };

  // Move an existing checkpoint to the most-recently-used end of the LRU.
  void touch(CacheEntry& entry);

  SingleBlockManager slots_;

  std::unordered_map<XXH3Key,
                     CacheEntry,
                     FixedStringKeyHash,
                     FixedStringKeyEqual>
      cached_slots_;
  std::list<XXH3Key> lru_;  // front = least recently used
};

}  // namespace xllm

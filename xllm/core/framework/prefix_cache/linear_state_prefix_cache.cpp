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

#include "linear_state_prefix_cache.h"

#include <glog/logging.h>

#include <algorithm>
#include <iterator>
#include <unordered_map>
#include <utility>

#include "framework/model/model_input_params.h"
#include "framework/request/sequence.h"

namespace xllm {

namespace {

// All saves now go through promotion: the sequence keeps a fresh live slot and
// the just-finished slot is grafted into the prefix cache directly. This works
// at every prefill checkpoint boundary because the slot's contents are frozen
// from the previous step and not yet overwritten by the current one.
bool can_promote_linear_state_checkpoint(const Sequence* sequence,
                                         const LinearStateCacheOp& cache_op) {
  if (sequence == nullptr || !sequence->has_linear_state_slot()) {
    return false;
  }
  // The sole caller (resolve_cache_ops) has already screened out zero-hash
  // saves before reaching here, so the save_prefix_hash is known non-zero.
  if (cache_op.linear_state_id < 0 ||
      cache_op.linear_state_id != sequence->get_linear_state_slot_id()) {
    return false;
  }
  return true;
}

}  // namespace

LinearStateCheckpointReservations::Promotion::Promotion(const XXH3Key& hash,
                                                        Sequence* sequence,
                                                        int32_t live_slot_id,
                                                        Block replacement_slot)
    : hash_(hash),
      sequence_(sequence),
      live_slot_id_(live_slot_id),
      replacement_slot_(std::move(replacement_slot)) {}

LinearStatePrefixCache::LinearStatePrefixCache(int32_t num_slots)
    : slots_(static_cast<uint32_t>(num_slots),
             /*resource_name=*/"linear state slot",
             /*exhaustion_message=*/"") {
  CHECK_GT(num_slots, 0) << "linear state prefix cache needs at least one slot";
}

Block LinearStatePrefixCache::allocate_live_slot() {
  if (slots_.num_free_blocks() == 0) {
    // Walk the LRU front-to-back; skip checkpoints currently pinned by a
    // restore in this batch (ref_count > 1), evict the first non-pinned one.
    // Erasing the entry drops its Block handle, which returns the slot to the
    // underlying free list synchronously.
    auto victim_it = lru_.begin();
    while (victim_it != lru_.end()) {
      auto cached_it = cached_slots_.find(*victim_it);
      CHECK(cached_it != cached_slots_.end())
          << "LRU and linear-state prefix cache out of sync";
      if (cached_it->second.block.is_shared()) {
        ++victim_it;
        continue;
      }
      lru_.erase(victim_it);
      cached_slots_.erase(cached_it);
      break;
    }
    if (slots_.num_free_blocks() == 0) {
      // Every slot is live or held by a pending reservation; nothing can be
      // reclaimed from the committed prefix cache.
      return Block();
    }
  }
  auto blocks = slots_.allocate(1);
  CHECK_EQ(blocks.size(), 1u) << "linear state slot allocation failed";
  return std::move(blocks[0]);
}

Block LinearStatePrefixCache::match(const XXH3Key& prefix_hash) {
  auto it = cached_slots_.find(prefix_hash);
  if (it == cached_slots_.end()) {
    return Block();
  }
  touch(it->second);
  return it->second.block;
}

int32_t LinearStatePrefixCache::insert_checkpoint(const XXH3Key& prefix_hash,
                                                  Block checkpoint_block) {
  if (!checkpoint_block.is_valid()) {
    return -1;
  }

  auto it = cached_slots_.find(prefix_hash);
  if (it != cached_slots_.end()) {
    touch(it->second);
    return it->second.block.id();
  }

  const int32_t slot_id = checkpoint_block.id();
  lru_.push_back(prefix_hash);
  cached_slots_.emplace(
      prefix_hash,
      CacheEntry{std::move(checkpoint_block), std::prev(lru_.end())});
  return slot_id;
}

bool LinearStatePrefixCache::contains(const XXH3Key& prefix_hash) const {
  return cached_slots_.find(prefix_hash) != cached_slots_.end();
}

LinearStateCheckpointReservations LinearStatePrefixCache::resolve_cache_ops(
    std::vector<LinearStateCacheOp>* cache_ops,
    const std::vector<Sequence*>& sequences) {
  LinearStateCheckpointReservations checkpoint_reservations;
  if (cache_ops == nullptr || cache_ops->empty()) {
    return checkpoint_reservations;
  }

  const bool has_aligned_sequences = sequences.size() == cache_ops->size();
  for (LinearStateCacheOp& cache_op : *cache_ops) {
    if (is_zero_prefix_hash(cache_op.restore_prefix_hash)) {
      continue;
    }
    Block restore_pin = match(XXH3Key(cache_op.restore_prefix_hash.data()));
    if (restore_pin.is_valid()) {
      cache_op.restore_src_slot_id = restore_pin.id();
      checkpoint_reservations.restore_pins.emplace_back(std::move(restore_pin));
    }
  }

  // Saves are always handled by promotion: the sequence keeps a fresh live
  // slot and the old slot is grafted into the prefix cache. This mirrors the
  // KV-cache shape where the just-written block becomes a cache entry without
  // any extra device copy. The actual slot swap is deferred until
  // commit_reservations() (after this step's forward), because the model in
  // this step still writes the old slot and we need its end-of-step contents
  // to become the frozen checkpoint.
  std::unordered_map<XXH3Key, int32_t, FixedStringKeyHash, FixedStringKeyEqual>
      promoted_hashes;
  for (size_t i = 0; i < cache_ops->size(); ++i) {
    LinearStateCacheOp& cache_op = (*cache_ops)[i];
    if (is_zero_prefix_hash(cache_op.save_prefix_hash)) {
      continue;
    }
    const XXH3Key save_hash(cache_op.save_prefix_hash.data());
    auto promoted_it = promoted_hashes.find(save_hash);
    if (promoted_it != promoted_hashes.end()) {
      // Another sequence in this batch already plans to freeze the same
      // prefix; record the slot id but do not promote a second time.
      cache_op.save_dst_slot_id = promoted_it->second;
      continue;
    }
    if (contains(save_hash)) {
      // Already in the committed cache; promotion would be redundant.
      continue;
    }

    Sequence* sequence = has_aligned_sequences ? sequences[i] : nullptr;
    if (!can_promote_linear_state_checkpoint(sequence, cache_op)) {
      continue;
    }
    Block replacement_slot = allocate_live_slot();
    if (!replacement_slot.is_valid()) {
      // No free slot for the sequence's next live slot; skip this save. The
      // cache stays sparse rather than evicting under pressure.
      continue;
    }
    const int32_t live_slot_id = cache_op.linear_state_id;
    cache_op.save_dst_slot_id = live_slot_id;
    promoted_hashes.emplace(save_hash, live_slot_id);
    checkpoint_reservations.promotions.emplace_back(
        save_hash, sequence, live_slot_id, std::move(replacement_slot));
  }
  return checkpoint_reservations;
}

void LinearStatePrefixCache::commit_reservations(
    LinearStateCheckpointReservations&& checkpoint_reservations) {
  for (LinearStateCheckpointReservations::Promotion& promotion :
       checkpoint_reservations.promotions) {
    Sequence* sequence = promotion.sequence();
    DCHECK(sequence != nullptr && sequence->has_linear_state_slot() &&
           sequence->get_linear_state_slot_id() == promotion.live_slot_id());
    if (sequence == nullptr || !sequence->has_linear_state_slot() ||
        sequence->get_linear_state_slot_id() != promotion.live_slot_id()) {
      continue;
    }
    Block old_live_slot = sequence->reset_linear_state_slot();
    insert_checkpoint(promotion.hash(), std::move(old_live_slot));
    sequence->set_linear_state_slot(promotion.take_replacement_slot());
    sequence->reset_linear_state_initialized();
  }
}

void LinearStatePrefixCache::touch(CacheEntry& entry) {
  // Move the node to the MRU end in place; splice keeps lru_it valid and
  // avoids a list-node realloc.
  lru_.splice(lru_.end(), lru_, entry.lru_it);
}

}  // namespace xllm

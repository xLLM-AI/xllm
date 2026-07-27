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

#include "linear_state_block_manager.h"

#include <glog/logging.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "framework/request/sequence.h"

namespace xllm {

namespace {

BlockManager::Options make_linear_state_options(uint32_t num_slots,
                                                int32_t chunk_stride) {
  BlockManager::Options options;
  options.num_blocks(num_slots);
  options.block_size(chunk_stride);
  options.enable_prefix_cache(true);
  options.enable_disagg_pd(false);
  options.block_type(BlockType::LINEAR);
  return options;
}

}  // namespace

LinearStateBlockManager::LinearStateBlockManager(uint32_t num_slots,
                                                 int32_t chunk_stride)
    : BlockManagerImpl(make_linear_state_options(num_slots, chunk_stride)) {
  CHECK_GT(num_slots, 1u)
      << "linear-state leaf needs at least one usable slot (plus padding)";
  CHECK_GT(chunk_stride, 0)
      << "linear-state leaf needs a positive chunk stride";
}

std::optional<std::vector<Block>>
LinearStateBlockManager::allocate_for_sequence(Sequence* seq,
                                               size_t /*num_tokens*/) {
  if (seq == nullptr) {
    return std::nullopt;
  }
  // Step 1: execute a save deferred from the previous step. The batch builder
  // recorded the boundary hash last step (set_pending_linear_save) and the
  // intervening forward wrote that boundary's recurrent state into the live
  // slot, so the slot contents now match the recorded hash. When a save is due
  // and a free slot is available we rotate the slot with the SAME accounting
  // verbs KV blocks use: checkpoint the warm slot into the index under the
  // RECORDED hash, mount it as this sequence's class-B restore source,
  // deallocate the old slot through the standard manager path, and return the
  // fresh slot for the composite to add_blocks -- no bespoke in-place swap. Any
  // reason to skip (no pending save, dup hash, no live slot to promote, no free
  // slot) leaves the current slot untouched: a best-effort checkpoint must
  // never surface as an allocation failure, or the composite would roll back
  // the whole sequence.
  const std::optional<XXH3Key> pending_hash = seq->take_pending_linear_save();
  const bool should_apply_save = pending_hash.has_value() &&
                                 seq->has_linear_state_slot() &&
                                 !prefix_cache_->contains(*pending_hash);
  Block new_live_slot = should_apply_save ? allocate() : Block();
  if (new_live_slot.is_valid()) {
    // Pin the warm slot into the checkpoint index (refcount+1) and mount a
    // second alias as the class-B restore source the next step's builder
    // consumes (copy old->new). copy_block leaves the slot in the sequence too.
    Block checkpoint_slot = seq->copy_block(BlockType::LINEAR);
    seq->set_linear_restore_src_block(Block(checkpoint_slot));
    insert_with_recorded_hash(std::move(checkpoint_slot), *pending_hash);
    // Retire the old slot through the standard deallocate path: its refcount is
    // still 3 (index + restore src + sequence), above the <=2u threshold, so
    // used-block accounting is untouched here -- the index owner keeps it live
    // until LRU eviction, exactly like a shared KV block. Then drop only the
    // sequence's own alias by clearing the LINEAR vector; NOT erase_blocks,
    // which would also reset the restore source just mounted.
    deallocate(seq->kv_state().blocks(BlockType::LINEAR));
    seq->kv_state().mutable_blocks(BlockType::LINEAR)->clear();
    // Hand the fresh slot to the composite; its add_blocks commit installs it
    // as the sequence's sole LINEAR slot, the same path a KV allocation takes.
    std::vector<Block> rotated;
    rotated.emplace_back(std::move(new_live_slot));
    return rotated;
  }
  // Step 2: no rotation happened. A continued chunk already holds its live slot
  // -- return no new blocks, the composite keeps the slot it already has. Only
  // a fresh sequence (no slot yet) falls through to acquire its first slot,
  // returned for the composite to append.
  if (seq->has_linear_state_slot()) {
    return std::vector<Block>{};
  }
  Block slot = allocate();
  if (!slot.is_valid()) {
    LOG(ERROR) << "Failed to acquire linear state slot! free="
               << num_free_blocks() << ", used=" << num_used_blocks()
               << ", total=" << num_total_blocks();
    return std::nullopt;
  }
  std::vector<Block> blocks;
  blocks.emplace_back(std::move(slot));
  return blocks;
}

Block LinearStateBlockManager::allocate() {
  std::vector<Block> blocks = BlockManagerImpl::allocate(1);
  if (blocks.empty()) {
    return Block();
  }
  return std::move(blocks[0]);
}

std::vector<Block> LinearStateBlockManager::allocate_shared(
    const Slice<int32_t>& token_ids,
    const Slice<Block>& /*existed_shared_blocks*/,
    const MMData& /*mm_data*/,
    const Slice<XXH3Key>& block_hashes) {
  // Probe + mount through the base match virtual: prefix_cache_ is the
  // LinearStatePrefixCache, whose gap-tolerant walk over the chunk-strided
  // hash domain (chunk stride captured at construction from the scheduler
  // config; block_size_ of this cache) returns a vector of shape
  // `[hit_or_inv, ..., last_hit]` per chunk. |block_hashes| carries the
  // sequence's precomputed chained per-chunk linear-state hashes
  // (Sequence::update_linear_state_hashes) so the probe consumes them
  // instead of recomputing; the composite picks the deepest valid slot out
  // as the class-A restore source. Linear thus travels the same
  // allocate_shared verb as KV, and beneath it the same match verb -- no
  // downcast, plain virtual dispatch. The index is KV-blind; the composite
  // owns the cross-leaf min + chunk alignment + deepest-pick.
  //
  // Reserve one fresh token by slicing off the last input token before the
  // probe. The forward must compute at least one token, so a checkpoint that
  // sits exactly at token_ids.size() would leave zero tokens to run; slicing
  // to size-1 makes match() naturally stop at the deepest chunk STRICTLY
  // below the final token. This is LINEAR-specific -- it cannot be delegated
  // to KV's own size-1 (the add_shared_blocks last-block pop): that pop lands
  // on an arbitrary KV block boundary, whereas linear checkpoints exist only
  // on sparse chunk boundaries. If reuse were clamped by KV's boundary the
  // restore would look for a checkpoint that need not be there; on a miss the
  // sequence cold-starts on a non-empty KV prefix and its recurrent state is
  // corrupt.
  if (token_ids.size() == 0) {
    return {};
  }
  return prefix_cache_->match(token_ids.slice(0, token_ids.size() - 1),
                              /*existed_shared_blocks=*/{},
                              MMData(),
                              block_hashes);
}

void LinearStateBlockManager::cache(const Slice<int32_t>& /*token_ids*/,
                                    std::vector<Block>& /*blocks*/,
                                    size_t /*existed_shared_blocks_num*/,
                                    const MMData& /*mm_data*/,
                                    const Slice<XXH3Key>& /*block_hashes*/) {
  NOT_IMPLEMENTED();
}

void LinearStateBlockManager::insert_with_recorded_hash(Block&& slot,
                                                        const XXH3Key& hash) {
  // Key the checkpoint under the RECORDED boundary hash, never a hash
  // recomputed from the sequence's current tokens: the pending save was
  // recorded at a shallower boundary than where the sequence now stands, so a
  // recompute would key the deeper current boundary and the checkpoint could
  // never be matched. set_hash_value stamps the recorded key, then the
  // inherited by-block cache() primitive inserts without recomputing.
  slot.set_hash_value(hash.data);
  std::vector<Block> checkpoint;
  checkpoint.emplace_back(std::move(slot));
  cache(checkpoint);
}

}  // namespace xllm

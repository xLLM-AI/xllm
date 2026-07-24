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
#include <optional>
#include <vector>

#include "block_manager_impl.h"
#include "util/hash_util.h"

namespace xllm {

class Sequence;

// CompositeBlockManager leaf for Qwen3.5 GDN linear-state checkpoints.
// Registered under BlockType::LINEAR, enable_prefix_cache=true.
//
// Inherits BlockManagerImpl for the slot id pool and prefix cache. The
// inherited prefix_cache_ serves as the checkpoint index (LRU + eviction that
// skips is_shared() slots). Hash domain is INDEPENDENT of the KV leaf — keys
// are chained per prefill chunk (Sequence::update_linear_state_hashes), not
// from KV block hashes.
//
// The id pool is unified: slot ids [1, num_slots) serve LIVE slots (held by a
// sequence under composite_blocks_[LINEAR]) AND committed CHECKPOINT slots
// (owned by prefix_cache_) interchangeably under reference counting. Slot 0 is
// the inherited padding block.
class LinearStateBlockManager final : public BlockManagerImpl {
 public:
  // |chunk_stride| is the linear-state checkpoint stride in tokens (one prefill
  // chunk), used as block_size for both the slot pool and the prefix-cache hash
  // domain. Defaults to -1 (probe disabled) for block-level unit tests that
  // drive the slot pool directly and never exercise the token-based match().
  LinearStateBlockManager(uint32_t num_slots, int32_t chunk_stride = -1);
  ~LinearStateBlockManager() override = default;

  // ---- BlockManagerImpl overrides ----

  // Per-step entry for this leaf. Owns the full slot lifecycle:
  //   1. First executes any save deferred from the previous step -- the
  //      sequence's live slot now holds the just-computed boundary state, so it
  //      is promoted into the checkpoint index and the sequence is rotated onto
  //      a fresh live slot.
  //   2. Then, if the sequence still lacks a live slot, acquires one.
  // A save rotates the slot with the same accounting verbs KV uses: it
  // deallocates the old slot and RETURNS the fresh one for the composite to
  // add_blocks, rather than mutating the sequence in place. Step 2 likewise
  // returns any newly acquired slot; a continued chunk that already holds its
  // slot returns nothing.
  std::optional<std::vector<Block>> allocate_for_sequence(
      Sequence* seq,
      size_t num_tokens) override;

  using BlockManagerImpl::allocate;
  Block allocate() override;

  // Probe (do NOT admit) this leaf's checkpoint index for the furthest
  // recoverable prefix. This IS the base allocate_shared verb -- KV and LINEAR
  // travel the same virtual, no downcast -- but its semantics diverge from
  // KV's: KV marks the returned blocks used and the composite adds them to the
  // sequence, whereas the checkpoint returned here is a restore SOURCE only. It
  // never enters the sequence's block vector and is never counted against
  // admission. |token_ids| is the sequence's full token span; |block_hashes|
  // carries the sequence's precomputed chained per-chunk linear-state hashes
  // (own hash domain, chunk-strided -- not the KV by-block chain), forwarded to
  // the probe; |existed_shared_blocks|/|mm_data| are unused. Reach in tokens
  // is derived from `returned.size() * block_size()` (block_size() is the
  // chunk stride), symmetric with KV/SWA. The leaf is KV-blind: the composite
  // owns the cross-leaf min + chunk alignment + KV-block clamp.
  //
  // The recurrent state is cumulatively compressed, so at most ONE checkpoint
  // survives -- the single deepest hit, whose state subsumes every earlier
  // one. The returned vector has the shape `[inv, inv, ..., deepest_valid]`
  // at chunk-stride granularity (deepest slot fetched via find(), so refcount
  // is +1 and it is LRU-promoted, pinned until restored). The caller
  // (composite) picks the deepest slot out as the class-A restore source and
  // never mounts LINEAR blocks into the sequence's live LINEAR vector (that
  // slot is the live, in-place-updated one). Reach in tokens is
  // `returned.size() * block_size()` where block_size() is the chunk stride;
  // an empty vector means nothing hit.
  std::vector<Block> allocate_shared(
      const Slice<int32_t>& token_ids,
      const Slice<Block>& existed_shared_blocks = {},
      const MMData& mm_data = MMData(),
      const Slice<XXH3Key>& block_hashes = {}) override;
  void cache(const Slice<int32_t>& token_ids,
             std::vector<Block>& blocks,
             size_t existed_shared_blocks_num = 0,
             const MMData& mm_data = MMData(),
             const Slice<XXH3Key>& block_hashes = {}) override;
  // The pre-hashed cache(const std::vector<Block>&) primitive is reused
  // verbatim for checkpoint insertion, so keep the base version instead of a
  // bespoke one.
  using BlockManagerImpl::cache;

 private:
  // Grants the test peer access to the inherited prefix_cache_ so unit tests
  // can assert by-hash checkpoint presence/lookup directly, mirroring how KV
  // unit tests assert against PrefixCache::find / contains. Production never
  // probes by hash here -- the composite mounts restore sources through
  // allocate_shared -- so no by-hash probe method is exposed on this leaf.
  friend class BlockManagerPoolTestPeer;

  // Insert |slot| into the checkpoint index under the RECORDED boundary |hash|
  // (stamped via set_hash_value), reusing the inherited by-block cache()
  // primitive so the key is never recomputed from the sequence's current
  // tokens. Recomputing here would key the deeper current boundary, not the
  // saved one, and the checkpoint would never be matched.
  void insert_with_recorded_hash(Block&& slot, const XXH3Key& hash);
};

}  // namespace xllm

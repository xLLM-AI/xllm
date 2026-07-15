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
#include <vector>

#include "core/framework/multimodal/mm_data.h"
#include "framework/block/block.h"
#include "prefix_cache.h"
#include "util/hash_util.h"
#include "util/slice.h"

namespace xllm {

// Checkpoint index for Qwen3.5 GDN linear-state slots. Same LRU / eviction /
// insert machinery as the base cache; the only addition is the admission-time
// prefix probe. Restore still goes through the base find(XXH3Key) point-lookup,
// so this subclass owns just the "how long a prefix can I recover" question.
// The chained per-chunk hashes it probes with are precomputed once on the
// sequence (Sequence::update_linear_state_hashes) and passed into match() via
// |block_hashes|, symmetric with how KV passes seq->block_hashes(). A prefix
// cache indexes on a single block-boundary stride: for KV that is the KV block
// size, for this index it is one prefill chunk in tokens. That stride IS the
// base block_size_ -- the factory feeds the chunk stride into the single
// block_size slot -- so the size-1 boundary rule below reads it directly and
// stays self-contained and unit-testable.
class LinearStatePrefixCache final : public PrefixCache {
 public:
  LinearStatePrefixCache(uint32_t chunk_stride, BlockHasherType hasher_type)
      : PrefixCache(chunk_stride, hasher_type) {}

  // Linear-state override of the KV match. |token_ids| is hashed into chained
  // per-chunk keys (chunk = one prefill chunk in tokens, captured at
  // construction from the scheduler config) for every whole chunk STRICTLY
  // BELOW the final token, and the index is probed for the furthest recoverable
  // prefix. Each hit extends the reach to (chunk_index + 1) * chunk_stride
  // tokens. Gaps are allowed -- a later hit extends past an intervening miss.
  // |matched_tokens| is written with that reach (a multiple of chunk_stride; 0
  // when nothing hits).
  //
  // The final token is reserved (probe limit = size - 1) so the forward always
  // has at least one fresh token to compute. This size-1 is NOT redundant with
  // KV's own size-1 (the add_shared_blocks last-block pop): that pop lands on
  // an arbitrary KV *block* boundary, whereas linear checkpoints exist only on
  // sparse *chunk* boundaries. If reuse were clamped by KV's boundary the
  // restore would look for a checkpoint that need not be there; on a miss the
  // sequence cold-starts on a non-empty KV prefix and its recurrent state is
  // corrupt (per-token KV attention tolerates arbitrary-boundary cold starts,
  // cumulatively-compressed recurrent state does not). Only this index knows
  // where the checkpoints are, so the snap-to-deepest-checkpoint-below-final
  // must happen here. The override stays KV-blind otherwise: it counts only in
  // its own token / chunk domain and never sees the KV match length -- bounding
  // reuse to what KV also matched is the composite's job.
  //
  // The probe uses contains() and does NOT touch LRU recency. Because recurrent
  // state is cumulatively compressed, only the single deepest hit is mounted:
  // its checkpoint subsumes every earlier one. On a hit the returned vector
  // holds that one block, fetched via find() (refcount+1, LRU-promoted) so it
  // stays pinned until restored; empty when nothing hits.
  //
  // |existed_shared_blocks|/|mm_data| are unused. |block_hashes| carries the
  // sequence's precomputed chained per-chunk linear-state hashes (own hash
  // domain, chunk-strided -- not the KV by-block chain); the probe consumes
  // them directly instead of recomputing. |matched_tokens|, when non-null,
  // receives the recovered token length.
  std::vector<Block> match(const Slice<int32_t>& token_ids,
                           const Slice<Block>& existed_shared_blocks = {},
                           const MMData& mm_data = MMData(),
                           const Slice<XXH3Key>& block_hashes = {},
                           size_t* matched_tokens = nullptr) override;
};

}  // namespace xllm

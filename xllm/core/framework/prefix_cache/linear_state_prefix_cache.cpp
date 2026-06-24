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

namespace xllm {

std::vector<Block> LinearStatePrefixCache::match(
    const Slice<int32_t>& token_ids,
    const Slice<Block>& /*existed_shared_blocks*/,
    const MMData& /*mm_data*/,
    const Slice<XXH3Key>& block_hashes,
    size_t* matched_tokens) {
  if (matched_tokens != nullptr) {
    *matched_tokens = 0;
  }
  // The chunk stride (one prefill chunk in tokens) is the linear hash domain's
  // step. It is this cache's single block-boundary size (the inherited
  // block_size_), fed in at construction from the scheduler config; the engine
  // enforces a positive multiple of the KV block size when linear prefix cache
  // is on. A misconfigured run leaves the stride at its -1 default, which
  // arrives here as a saturated unsigned value that drives probe_chunk_count to
  // 0 below (nothing probed); guard the zero case explicitly so the probe
  // arithmetic never divides by zero.
  if (block_size_ == 0) {
    return {};
  }
  const size_t chunk_stride = block_size_;
  // The chained per-chunk hashes are precomputed once on the sequence
  // (update_linear_state_hashes) and passed in via |block_hashes|; they cover
  // every whole chunk of the prefix, including one that may end exactly at the
  // final token. Probe only chunks STRICTLY below the final token: the forward
  // must compute at least one fresh token, so a checkpoint sitting exactly at
  // token_ids.size() is unusable (restoring it would leave zero tokens to run).
  // This is pure token-domain arithmetic -- still KV-blind -- but it cannot be
  // delegated to KV's own size-1 (the add_shared_blocks pop): that pop lands on
  // an arbitrary KV *block* boundary, which need not be a linear *checkpoint*
  // boundary. When the two diverge the restore misses and the sequence silently
  // cold-starts on a non-empty KV prefix -> corrupt recurrent state. Only this
  // index knows where the checkpoints are, so snapping to the deepest one below
  // the final token must happen here; the caller (composite) then takes the min
  // with the KV match.
  const size_t probe_chunk_count =
      token_ids.size() == 0 ? 0 : (token_ids.size() - 1) / chunk_stride;
  const size_t probe_limit = std::min(probe_chunk_count, block_hashes.size());
  size_t deepest_hit_chunk = probe_limit;  // sentinel: nothing hit
  for (size_t chunk_idx = 0; chunk_idx < probe_limit; ++chunk_idx) {
    // Probe with contains() (no LRU touch): a probed-but-unused prefix must not
    // pollute recency. Recurrent state is cumulatively compressed, so a later
    // hit's checkpoint subsumes earlier ones -- keep only the deepest. Gaps are
    // allowed: do NOT break on a miss, a later hit extends past it.
    if (contains(block_hashes[chunk_idx])) {
      deepest_hit_chunk = chunk_idx;
    }
  }
  if (deepest_hit_chunk >= probe_limit) {
    return {};
  }
  // One-block insight: mount only the single deepest checkpoint. Its
  // cumulatively-compressed recurrent state covers the whole recoverable
  // prefix. find() pins it (refcount+1) and promotes its LRU recency now that
  // it is about to be restored, so the earlier probe stays side-effect free.
  Block deepest = find(block_hashes[deepest_hit_chunk]);
  if (!deepest.is_valid()) {
    return {};
  }
  if (matched_tokens != nullptr) {
    *matched_tokens = (deepest_hit_chunk + 1) * chunk_stride;
  }
  std::vector<Block> mounted;
  mounted.reserve(1);
  mounted.emplace_back(std::move(deepest));
  return mounted;
}

}  // namespace xllm

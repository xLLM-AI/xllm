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

#include "sliding_window_block_manager.h"

#include <algorithm>

#include "framework/prefix_cache/prefix_cache.h"

namespace xllm {

SlidingWindowBlockManager::SlidingWindowBlockManager(const Options& options)
    : BlockManagerImpl(options) {
  CHECK_GT(options_.swa_blocks_per_seq(), 0u)
      << "swa_blocks_per_seq must be positive";
  if (options_.enable_prefix_cache()) {
    // SWA prefix cache uses Sequence::block_hashes_ (TEXT chain). VLM MM
    // hasher is not compatible; fail loud instead of silently corrupting hits.
    CHECK(options_.hasher_type() == BlockHasherType::TEXT)
        << "SWA prefix cache does not yet support VLM (MM hasher). "
           "Disable prefix cache for VLM DSV4 or wait for VLM support.";
  }
}

void SlidingWindowBlockManager::release_out_of_window(Sequence* seq) {
  if (seq == nullptr) {
    return;
  }
  KVCacheState& kv_state = seq->kv_state();
  std::vector<Block>& swa_blocks = *kv_state.mutable_blocks(block_type());
  const size_t block_size = options_.block_size();
  if (block_size == 0 || swa_blocks.empty()) {
    return;
  }
  const size_t cached_tokens = kv_state.kv_cache_tokens_num();
  const size_t num_spec_tokens =
      static_cast<size_t>(options_.num_speculative_tokens());
  const size_t sliding_window_tokens =
      std::max<size_t>(options_.sliding_window_size(), 1);
  if (cached_tokens < (sliding_window_tokens + num_spec_tokens)) {
    return;
  }
  const size_t skipped_tokens =
      cached_tokens - sliding_window_tokens - num_spec_tokens + 1;
  const size_t skipped_blocks = skipped_tokens / block_size;
  const size_t release_blocks = std::min(skipped_blocks, swa_blocks.size());
  if (release_blocks == 0) {
    return;
  }
  // Move slid-out blocks out (leaving invalid placeholders so positional
  // indexing stays stable). Cache alias still pins the physical block, so
  // deallocate walks the ref<=2u branch and only clears usage bookkeeping.
  std::vector<Block> blocks_to_release;
  blocks_to_release.reserve(release_blocks);
  for (size_t j = 0; j < release_blocks; ++j) {
    if (swa_blocks[j].is_valid()) {
      blocks_to_release.emplace_back(std::move(swa_blocks[j]));
    }
  }
  if (!blocks_to_release.empty()) {
    deallocate(blocks_to_release);
  }
}

std::vector<Block> SlidingWindowBlockManager::allocate_shared(
    const Slice<int32_t>& token_ids,
    const Slice<Block>& /*existed_shared_blocks*/,
    const MMData& mm_data,
    const Slice<XXH3Key>& block_hashes) {
  if (!options_.enable_prefix_cache() || options_.block_size() == 0 ||
      prefix_cache_ == nullptr) {
    return {};
  }
  AUTO_COUNTER(prefix_cache_latency_seconds_match);
  std::vector<Block> result = prefix_cache_->match(token_ids,
                                                   /*existed_shared_blocks=*/{},
                                                   mm_data,
                                                   block_hashes);
  if (result.empty()) {
    return {};
  }

  // Bookkeeping: mark_used only for valid positions. mark_used is idempotent
  // per block id, so blocks shared across sequences are only counted once.
  size_t added = 0;
  for (const auto& b : result) {
    if (b.is_valid() && mark_used(&usage_accounted_ids_, b.id())) {
      ++added;
    }
  }
  num_used_blocks_.fetch_add(added, std::memory_order_relaxed);

  const size_t reach_tokens =
      result.size() * static_cast<size_t>(options_.block_size());
  COUNTER_ADD(prefix_cache_match_length_total, reach_tokens);
  return result;
}

}  // namespace xllm

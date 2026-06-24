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

namespace xllm {

namespace {

// SWA never serves prefix cache; force it off so the base does not build one.
BlockManager::Options without_prefix_cache(BlockManager::Options options) {
  options.enable_prefix_cache(false);
  return options;
}

}  // namespace

SlidingWindowBlockManager::SlidingWindowBlockManager(const Options& options)
    : BlockManagerImpl(without_prefix_cache(options)) {
  CHECK_GT(options_.swa_blocks_per_seq(), 0u)
      << "swa_blocks_per_seq must be positive";
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
  // Leading logical blocks that have fully slid out of the window can be freed.
  // Runs AFTER the composite has committed this round's growth, and only on a
  // successful allocation, so a failed allocate_sequence never releases the
  // sequence's existing SWA blocks.
  const size_t cached_tokens = kv_state.kv_cache_tokens_num();
  const size_t sliding_window_tokens =
      std::max<size_t>(options_.sliding_window_size(), 1);
  if (cached_tokens < sliding_window_tokens) {
    return;
  }
  const size_t skipped_tokens = cached_tokens - sliding_window_tokens + 1;
  const size_t skipped_blocks = skipped_tokens / block_size;
  const size_t release_blocks = std::min(skipped_blocks, swa_blocks.size());
  if (release_blocks == 0) {
    return;
  }
  // Move slid-out blocks out (leaving invalid placeholders so logical-position
  // indexing stays stable) and deallocate them through the base free list. The
  // is_valid() guard makes re-entry a safe no-op.
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
    const Slice<int32_t>& /*token_ids*/,
    const Slice<Block>& /*existed_shared_blocks*/,
    const MMData& /*mm_data*/,
    const Slice<XXH3Key>& /*block_hashes*/,
    size_t* /*matched_tokens*/) {
  NOT_IMPLEMENTED();
  return {};
}

void SlidingWindowBlockManager::cache(const Slice<int32_t>& /*token_ids*/,
                                      std::vector<Block>& /*blocks*/,
                                      size_t /*existed_shared_blocks_num*/,
                                      const MMData& /*mm_data*/,
                                      const Slice<XXH3Key>& /*block_hashes*/) {
  NOT_IMPLEMENTED();
}

void SlidingWindowBlockManager::cache(const std::vector<Block>& /*blocks*/) {
  NOT_IMPLEMENTED();
}

}  // namespace xllm

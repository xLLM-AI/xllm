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

std::optional<std::vector<Block>>
SlidingWindowBlockManager::allocate_for_sequence(Sequence* seq,
                                                 size_t num_tokens) {
  if (seq == nullptr) {
    return std::nullopt;
  }
  return allocate_for_sequence(seq, seq->kv_state(), num_tokens);
}

std::optional<std::vector<Block>>
SlidingWindowBlockManager::allocate_for_sequence(Sequence* seq,
                                                 KVCacheState& kv_state,
                                                 size_t num_tokens) {
  if (seq == nullptr) {
    return std::nullopt;
  }
  const size_t block_size = options_.block_size();
  if (block_size == 0) {
    return std::vector<Block>{};
  }

  const size_t held = kv_state.num_blocks(block_type());
  const size_t needed = (num_tokens + block_size - 1) / block_size;
  if (needed <= held) {
    return std::vector<Block>{};
  }

  const bool device_state = &kv_state == &seq->kv_state();
  const size_t completed_tokens = device_state ? seq->kv_cache_tokens_num()
                                               : kv_state.kv_cache_tokens_num();
  const size_t skipped =
      std::min(num_slid_out_blocks(completed_tokens), needed);
  const size_t first_physical_position = std::max(held, skipped);
  const size_t physical_count = needed - first_physical_position;

  // Releasing the completed prefix mutates the sequence in place. Prove that
  // its uniquely owned blocks plus the current free list can satisfy the new
  // tail before making that mutation. The SWA pool is sized from the global
  // burst budget, so it must not depend on evicting unrelated cached blocks.
  const Slice<Block> current_blocks = kv_state.blocks(block_type());
  const size_t release_positions = std::min(skipped, held);
  size_t releasable_blocks = 0;
  for (size_t position = 0; position < release_positions; ++position) {
    const Block& block = current_blocks[position];
    if (block.is_valid() && block.ref_count() <= 2u) {
      ++releasable_blocks;
    }
  }
  const size_t available_without_eviction =
      num_free_blocks() + releasable_blocks;
  if (physical_count > available_without_eviction) {
    const size_t eviction_deficit = physical_count - available_without_eviction;
    if (prefix_cache_ == nullptr ||
        !prefix_cache_->can_evict(eviction_deficit)) {
      return std::nullopt;
    }
  }

  release_out_of_window(seq, kv_state, completed_tokens);
  std::vector<Block> physical_blocks = allocate(physical_count);
  CHECK_EQ(physical_blocks.size(), physical_count)
      << "SWA capacity preflight succeeded but allocation failed";

  std::vector<Block> blocks(needed - held);
  for (size_t position = first_physical_position; position < needed;
       ++position) {
    blocks[position - held] =
        std::move(physical_blocks[position - first_physical_position]);
  }
  return blocks;
}

size_t SlidingWindowBlockManager::num_slid_out_blocks(
    size_t cached_tokens) const {
  const size_t block_size = options_.block_size();
  if (block_size == 0) {
    return 0;
  }
  const size_t num_spec_tokens =
      static_cast<size_t>(options_.num_speculative_tokens());
  const size_t sliding_window_tokens =
      std::max<size_t>(options_.sliding_window_size(), 1);
  if (cached_tokens < sliding_window_tokens + num_spec_tokens) {
    return 0;
  }
  const size_t skipped_tokens =
      cached_tokens - sliding_window_tokens - num_spec_tokens + 1;
  return skipped_tokens / block_size;
}

void SlidingWindowBlockManager::release_out_of_window(Sequence* seq) {
  if (seq == nullptr) {
    return;
  }
  release_out_of_window(seq, seq->kv_state(), seq->kv_cache_tokens_num());
}

void SlidingWindowBlockManager::release_out_of_window(Sequence* seq,
                                                      KVCacheState& kv_state) {
  release_out_of_window(seq, kv_state, kv_state.kv_cache_tokens_num());
}

void SlidingWindowBlockManager::release_out_of_window(Sequence* seq,
                                                      KVCacheState& kv_state,
                                                      size_t cached_tokens) {
  if (seq == nullptr) {
    return;
  }
  std::vector<Block>& swa_blocks = *kv_state.mutable_blocks(block_type());
  const size_t block_size = options_.block_size();
  if (block_size == 0 || swa_blocks.empty()) {
    return;
  }
  const size_t skipped_blocks = num_slid_out_blocks(cached_tokens);
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

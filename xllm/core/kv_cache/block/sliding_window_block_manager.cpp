/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/kv_cache/block/sliding_window_block_manager.h"

#include <algorithm>
#include <iterator>

#include "core/framework/request/sequence.h"
#include "core/kv_cache/prefix/prefix_cache.h"

namespace xllm {

SlidingWindowBlockManager::SlidingWindowBlockManager(const Options& options)
    : BlockManagerImpl(options) {
  CHECK_GT(options_.swa_blocks_per_seq(), 0u)
      << "swa_blocks_per_seq must be positive";
  if (options_.enable_prefix_cache()) {
    // SWA prefix cache uses Sequence::block_hashes_ (TEXT chain). VLM MM
    // hasher is not compatible; fail loud instead of silently corrupting hits.
    CHECK(options_.hasher_type() == BlockHasherType::TEXT ||
          options_.hasher_type() == BlockHasherType::MTP_TEXT)
        << "SWA prefix cache does not yet support VLM (MM hasher). "
           "Disable prefix cache for VLM DSV4 or wait for VLM support.";
  }
}

std::optional<std::vector<Block>>
SlidingWindowBlockManager::allocate_for_sequence(Sequence* seq,
                                                 KVCacheState& kv_state,
                                                 size_t num_tokens) {
  if (seq == nullptr) {
    return std::nullopt;
  }
  const size_t restore_tokens = seq->kv_cache_tokens_num();
  if (restore_tokens > kv_state.kv_cache_tokens_num()) {
    CHECK_LE(restore_tokens, num_tokens);
    const size_t block_size = options_.block_size();
    const size_t held = kv_state.num_blocks(block_type());
    const size_t logical_blocks = (num_tokens + block_size - 1) / block_size;
    if (logical_blocks <= held) {
      return std::vector<Block>{};
    }

    const size_t sliding_window_tokens =
        std::max<size_t>(options_.sliding_window_size(), 1);
    const size_t restore_window_start =
        restore_tokens - std::min(restore_tokens, sliding_window_tokens);
    const size_t first_live_block = restore_window_start / block_size;
    const size_t live_begin = std::max(held, first_live_block);
    const size_t active_blocks = logical_blocks - live_begin;
    std::vector<Block> live_blocks = allocate(active_blocks);
    if (live_blocks.size() != active_blocks) {
      const size_t reclaimable =
          num_reclaimable_out_of_window_blocks(kv_state, restore_tokens);
      if (num_free_blocks() + reclaimable < active_blocks) {
        return std::nullopt;
      }
      release_out_of_window(seq, kv_state, restore_tokens);
      live_blocks = allocate(active_blocks);
      if (live_blocks.size() != active_blocks) {
        return std::nullopt;
      }
    }

    std::vector<Block> sparse_blocks(live_begin - held);
    sparse_blocks.insert(sparse_blocks.end(),
                         std::make_move_iterator(live_blocks.begin()),
                         std::make_move_iterator(live_blocks.end()));
    return sparse_blocks;
  }
  if (!options_.instance_is_decode() || kv_state.num_blocks(block_type()) > 0) {
    std::optional<std::vector<Block>> blocks =
        BlockManagerImpl::allocate_for_sequence(seq, kv_state, num_tokens);
    if (blocks.has_value()) {
      return blocks;
    }

    // A block completed by the previous forward may now be outside the active
    // window. Release slid-out sequence references, then retry: uncached blocks
    // return directly to the free list, while cached checkpoint-window blocks
    // can be evicted and reused. Only mutate the sequence when those
    // reclaimable blocks make the retry large enough to succeed; otherwise a
    // failed composite round must preserve the existing SWA state.
    const size_t block_size = options_.block_size();
    const size_t held = kv_state.num_blocks(block_type());
    const size_t num_blocks_needed = (num_tokens + block_size - 1) / block_size;
    CHECK_GT(num_blocks_needed, held);
    const size_t num_additional = num_blocks_needed - held;
    const size_t reclaimable =
        num_reclaimable_out_of_window_blocks(kv_state, restore_tokens);
    if (num_free_blocks() + reclaimable < num_additional) {
      return std::nullopt;
    }
    release_out_of_window(seq, kv_state, restore_tokens);
    return BlockManagerImpl::allocate_for_sequence(seq, kv_state, num_tokens);
  }

  const size_t block_size = options_.block_size();
  const size_t logical_blocks = (num_tokens + block_size - 1) / block_size;
  const size_t sliding_window_tokens =
      std::max<size_t>(options_.sliding_window_size(), 1);
  const size_t window_start =
      num_tokens - std::min(num_tokens, sliding_window_tokens);
  const size_t inactive_blocks = window_start / block_size;
  const size_t active_blocks = logical_blocks - inactive_blocks;
  std::vector<Block> live_blocks = allocate(active_blocks);
  if (live_blocks.size() != active_blocks) {
    return std::nullopt;
  }

  std::vector<Block> sparse_blocks(inactive_blocks);
  sparse_blocks.insert(sparse_blocks.end(),
                       std::make_move_iterator(live_blocks.begin()),
                       std::make_move_iterator(live_blocks.end()));
  return sparse_blocks;
}

bool SlidingWindowBlockManager::allocate_for_prefetch(Sequence* seq,
                                                      size_t num_tokens) {
  if (seq == nullptr) {
    return false;
  }
  const size_t block_size = options_.block_size();
  const size_t target_blocks = num_tokens / block_size;
  constexpr size_t c128_span_blocks = 128;
  const size_t window_span =
      std::min<size_t>(options_.swa_blocks_per_seq(), c128_span_blocks);

  KVCacheState& host_state = seq->host_kv_state();
  const size_t cached_cursor =
      std::min(host_state.num_cached_blocks(BlockType::SWA), target_blocks);
  const size_t shared_blocks = host_state.shared_blocks_num(BlockType::SWA);
  std::vector<Block> old = host_state.take_blocks(BlockType::SWA);
  old.resize(target_blocks);

  std::vector<Block> blocks(target_blocks);
  for (size_t index = 0; index < old.size(); ++index) {
    if (index % c128_span_blocks >= c128_span_blocks - window_span) {
      blocks[index] = std::move(old[index]);
    }
  }
  deallocate(old);
  old.clear();

  seq->update_block_hashes(static_cast<uint32_t>(block_size),
                           options_.hasher_type());
  const Slice<XXH3Key> hashes = seq->block_hashes();
  CHECK_GE(hashes.size(), target_blocks);
  std::vector<size_t> missing;
  for (size_t index = cached_cursor; index < target_blocks; ++index) {
    if (index % c128_span_blocks >= c128_span_blocks - window_span &&
        !blocks[index].is_valid()) {
      missing.emplace_back(index);
    }
  }

  const bool complete = allocate_missing_blocks(missing, hashes, blocks);
  if (!blocks.empty()) {
    host_state.replace_composite_blocks(BlockType::SWA,
                                        std::move(blocks),
                                        std::min(shared_blocks, target_blocks),
                                        target_blocks);
  }
  return complete;
}

void SlidingWindowBlockManager::trim_prefetch_blocks(Sequence& seq,
                                                     size_t max_hit_tokens) {
  KVCacheState& host_state = seq.host_kv_state();
  std::vector<Block> blocks = host_state.take_blocks(BlockType::SWA);
  const size_t keep = std::min(max_hit_tokens / block_size(), blocks.size());
  size_t window_end = 0;
  for (size_t index = keep; index > 0; --index) {
    if (blocks[index - 1].is_valid()) {
      window_end = index;
      break;
    }
  }
  const size_t window_begin =
      window_end - std::min<size_t>(window_end, options_.swa_blocks_per_seq());
  std::vector<Block> window(
      std::make_move_iterator(blocks.begin() + window_begin),
      std::make_move_iterator(blocks.begin() + window_end));
  deallocate(blocks);
  blocks.clear();
  if (window.empty()) {
    return;
  }

  seq.update_block_hashes(static_cast<uint32_t>(block_size()),
                          options_.hasher_type());
  const Slice<XXH3Key> hashes = seq.block_hashes();
  CHECK_GE(hashes.size(), window_end);
  for (size_t i = 0; i < window.size(); ++i) {
    window[i].set_hash_value(hashes[window_begin + i].data);
  }
  BlockManagerImpl::cache(window);

  blocks.resize(window_end);
  std::move(window.begin(), window.end(), blocks.begin() + window_begin);
  host_state.replace_composite_blocks(
      BlockType::SWA, std::move(blocks), window_end, window_end);
}

void SlidingWindowBlockManager::release_out_of_window(Sequence* seq,
                                                      KVCacheState& kv_state) {
  if (seq == nullptr) {
    return;
  }
  release_out_of_window(seq, kv_state, seq->kv_cache_tokens_num());
}

void SlidingWindowBlockManager::release_out_of_window(Sequence* seq,
                                                      KVCacheState& kv_state,
                                                      size_t cached_tokens) {
  if (seq == nullptr) {
    return;
  }
  std::vector<Block>& swa_blocks = *kv_state.mutable_blocks(block_type());
  const size_t release_blocks =
      num_out_of_window_blocks(kv_state, cached_tokens);
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

size_t SlidingWindowBlockManager::num_out_of_window_blocks(
    const KVCacheState& kv_state,
    size_t cached_tokens) const {
  const size_t block_size = options_.block_size();
  const size_t held = kv_state.num_blocks(block_type());
  if (held == 0) {
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
  const size_t skipped_blocks = skipped_tokens / block_size;
  return std::min(skipped_blocks, held);
}

size_t SlidingWindowBlockManager::num_reclaimable_out_of_window_blocks(
    const KVCacheState& kv_state,
    size_t cached_tokens) const {
  const Slice<Block> swa_blocks = kv_state.blocks(block_type());
  const size_t release_blocks =
      num_out_of_window_blocks(kv_state, cached_tokens);
  const uint32_t max_reclaimable_ref_count =
      options_.enable_prefix_cache() ? 2u : 1u;
  size_t reclaimable = 0;
  for (size_t i = 0; i < release_blocks; ++i) {
    if (swa_blocks[i].is_valid() &&
        swa_blocks[i].ref_count() <= max_reclaimable_ref_count) {
      ++reclaimable;
    }
  }
  return reclaimable;
}

std::vector<Block> SlidingWindowBlockManager::allocate_shared(
    const Slice<int32_t>& token_ids,
    const Slice<Block>& /*existed_shared_blocks*/,
    const MMData& mm_data,
    const Slice<XXH3Key>& block_hashes) {
  if (!options_.enable_prefix_cache()) {
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
    if (b.is_valid() && mark_used(usage_accounted_ids_, b.id())) {
      ++added;
    }
  }
  num_used_blocks_.fetch_add(added, std::memory_order_relaxed);

  return result;
}

}  // namespace xllm

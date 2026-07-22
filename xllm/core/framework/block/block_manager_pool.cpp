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

#include "block_manager_pool.h"

#include <algorithm>
#include <limits>

#include "block_manager_impl.h"
#include "common/global_flags.h"
#include "composite_block_manager.h"
#include "concurrent_block_manager_impl.h"
#include "core/framework/config/kv_cache_config.h"
#include "framework/model/model_input_params.h"
#include "framework/xtensor/page_allocator.h"
#include "framework/xtensor/phy_page_pool.h"
#include "framework/xtensor/xtensor_block_manager_impl.h"
#include "linear_state_block_manager.h"

namespace xllm {

BlockManagerPool::BlockManagerPool(const Options& options, int32_t dp_size)
    : options_(options) {
  CHECK(dp_size > 0) << "dp_size must be greater than 0";
  block_managers_.reserve(dp_size);
  const uint32_t default_max_single_block_sequences =
      options_.max_seqs_per_batch();

  BlockManager::Options block_options;
  block_options.num_blocks(options_.num_blocks())
      .block_size(options_.block_size())
      .enable_prefix_cache(options_.enable_prefix_cache())
      .enable_disagg_pd(options_.enable_disagg_pd())
      .enable_kvcache_store(options_.enable_kvcache_store())
      .enable_host_offload(options_.enable_host_offload())
      .sliding_window_size(options_.sliding_window_size())
      .swa_blocks_per_seq(options_.swa_blocks_per_seq())
      .max_tokens_per_batch(options_.max_tokens_per_batch())
      .manager_types(options_.manager_types())
      .compress_ratios(options_.compress_ratios())
      .max_seqs_per_batch(options_.max_seqs_per_batch())
      .hasher_type(options_.hasher_type())
      .enable_xtensor(options_.enable_xtensor())
      .num_layers(options_.num_layers())
      .slot_size(options_.slot_size())
      .model_id(options_.model_id())
      .enable_linear_state(options_.enable_linear_state())
      .linear_state_num_slots(options_.linear_state_num_slots())
      .linear_chunk_stride(options_.linear_chunk_stride())
      .num_speculative_tokens(options_.num_speculative_tokens());

  uint32_t num_single_blocks = std::max<uint32_t>(
      options_.num_single_blocks(), default_max_single_block_sequences + 2);
  CHECK_GT(num_single_blocks, 0u) << "num_single_blocks must be positive";

  for (int32_t i = 0; i < dp_size; ++i) {
    // The pool always holds a CompositeBlockManager. Its KV leaf is a flat
    // BlockManagerImpl, or an XTensorBlockManagerImpl when enable_xtensor (the
    // builder picks); SWA / C4 / C128 come from manager_types; the per-sequence
    // SINGLE resource leaf is appended here under the SINGLE key. Every leaf is
    // routed by its BlockType, so xtensor and Single are ordinary leaves.
    auto leaves = build_composite_leaves(block_options, /*dp_rank=*/i);
    // SINGLE leaf needs the same concurrency wrapper as the other leaves when
    // sequence-level entry points run off the scheduler thread (disagg PD /
    // kvcache store prefill threadpools call try_allocate concurrently, and the
    // host-offload D2H callback frees blocks off-thread).
    std::unique_ptr<BlockManager> single_leaf =
        std::make_unique<SingleBlockManager>(
            /*num_blocks=*/num_single_blocks,
            /*resource_name=*/"single block",
            /*exhaustion_message=*/"No more single-block ids available");
    if (options_.enable_disagg_pd() || options_.enable_kvcache_store() ||
        options_.enable_host_offload()) {
      single_leaf =
          std::make_unique<ConcurrentBlockManagerImpl>(std::move(single_leaf));
    }
    leaves.emplace(
        BlockType::SINGLE,
        CompositeBlockManager::LeafEntry{std::move(single_leaf),
                                         /*participates_in_admission=*/false,
                                         /*supports_prefix_cache=*/false});
    block_managers_.emplace_back(
        std::make_unique<CompositeBlockManager>(std::move(leaves)));
  }
  swap_block_transfer_infos_.clear();
  swap_block_transfer_infos_.resize(block_managers_.size());
}

int32_t BlockManagerPool::get_manager_with_max_free_blocks() const {
  if (block_managers_.empty()) {
    return 0;
  }

  size_t max_index = 0;
  size_t max_free = block_managers_[0]->num_free_blocks();

  for (size_t i = 1; i < block_managers_.size(); ++i) {
    const size_t current_free = block_managers_[i]->num_free_blocks();
    if (current_free > max_free) {
      max_free = current_free;
      max_index = i;
    }
  }
  return max_index;
}

int32_t BlockManagerPool::get_dp_rank(Sequence* sequence) const {
  int32_t dp_rank;
  if (sequence->dp_rank() >= 0) {
    dp_rank = sequence->dp_rank();
  } else {
    dp_rank = get_manager_with_max_free_blocks();
    sequence->set_dp_rank(dp_rank);
  }
  return dp_rank;
}

void BlockManagerPool::deallocate(Request* request) {
  DCHECK(request != nullptr);
  for (auto& sequence : request->sequences()) {
    deallocate(sequence.get());
  }
}

void BlockManagerPool::deallocate(std::vector<Sequence*>& sequences) {
  for (auto* sequence : sequences) {
    deallocate(sequence);
  }
}

void BlockManagerPool::deallocate(Sequence* sequence) {
  DCHECK(sequence != nullptr);
  int32_t dp_rank = get_dp_rank(sequence);
  // The composite fans deallocate (with final cache) out across all leaves,
  // including the SINGLE resource leaf, the LINEAR leaf, and the (flat or
  // xtensor) KV leaf.
  auto* composite =
      static_cast<CompositeBlockManager*>(block_managers_[dp_rank].get());
  composite->deallocate_for_sequence(sequence);
  sequence->reset();
}

std::vector<std::vector<BlockTransferInfo>>*
BlockManagerPool::get_swap_block_transfer_infos() {
  return &swap_block_transfer_infos_;
}

bool BlockManagerPool::allocate(Sequence* sequence) {
  DCHECK(sequence != nullptr);
  return allocate(sequence, sequence->num_tokens());
}

bool BlockManagerPool::allocate(std::vector<Sequence*>& sequences) {
  for (auto* sequence : sequences) {
    DCHECK(sequence != nullptr);
    if (!allocate(sequence, sequence->num_tokens())) {
      // should we gurantee the atomicity of the allocation? all or nothing?
      return false;
    }
  }
  return true;
}

bool BlockManagerPool::allocate(Sequence* sequence, size_t num_tokens) {
  AUTO_COUNTER(allocate_blocks_latency_seconds);
  DCHECK(sequence != nullptr);
  int32_t dp_rank = get_dp_rank(sequence);
  // "Started empty" = the sequence holds no cache-bearing blocks of ANY type
  // (KV / SWA / C4 / C128), not just KV. DSV4 sequences never hold KV, so a
  // KV-only check would treat every DSV4 grow as a fresh allocation and, on
  // failure, wrongly deallocate + reset the already-held SWA/C4/C128 blocks.
  const bool started_empty = !sequence->kv_state().has_any_blocks();

  // The leaves (KV / SWA / C4 / C128 / SINGLE / LINEAR) each apply their own
  // strategy; the pool only orchestrates prefix-share-then-beam-then-grow,
  // which beam (KV copy-on-write) must sit between.
  auto* composite =
      static_cast<CompositeBlockManager*>(block_managers_[dp_rank].get());
  if (started_empty) {
    composite->allocate_shared_for_sequence(sequence);
  }
  // Beam swap decision uses the KV block count after the shared attach.
  const size_t kv_blocks = sequence->kv_state().num_blocks(BlockType::KV);
  const size_t block_size = options_.block_size();
  const size_t kv_blocks_needed = (num_tokens + block_size - 1) / block_size;
  const bool kv_already_satisfied = kv_blocks_needed <= kv_blocks;
  if (!process_beam_search(sequence, /*need_swap*/ kv_already_satisfied)) {
    return false;
  }
  // Always run the composite growth pass: even when KV is already satisfied, a
  // sequence (e.g. a fork/clone of a beam parent) may still be missing its
  // per-sequence SINGLE / LINEAR block. The composite skips no-op leaves
  // internally.
  if (!composite->allocate_sequence(sequence, num_tokens)) {
    if (started_empty) {
      composite->deallocate_for_sequence(sequence);
      sequence->reset();
    }
    return false;
  }
  return true;
}

bool BlockManagerPool::allocate(Sequence* sequence,
                                size_t num_tokens,
                                size_t needed_copy_in_blocks_num) {
  LOG(FATAL)
      << "allocate(Sequence* sequence, size_t num_tokens, size_t "
         "needed_copy_in_blocks_num) is not implemented in BlockManagerPool.";
  return false;
}

std::vector<Block> BlockManagerPool::allocate(size_t num_tokens,
                                              int32_t& dp_rank) {
  dp_rank = get_manager_with_max_free_blocks();
  const size_t block_size = options_.block_size();
  const size_t num_blocks_needed = (num_tokens + block_size - 1) / block_size;
  // block_managers_[dp_rank] is always a CompositeBlockManager, whose
  // type-ambiguous allocate(size_t) is NOT_IMPLEMENTED. Route to the KV leaf
  // explicitly (this raw-block path is the flat-KV-only helper used to seed /
  // evict the prefix cache; DSV4 has no KV leaf and must not call it).
  auto* composite =
      static_cast<CompositeBlockManager*>(block_managers_[dp_rank].get());
  return composite->allocate_blocks(BlockType::KV, num_blocks_needed);
}

bool BlockManagerPool::try_allocate(Sequence* sequence) {
  int32_t dp_rank = get_dp_rank(sequence);

  // Composite path: prefix-share + grow via the leaves, then advance the token
  // counter (try_allocate admits the full prompt up front).
  auto* composite =
      static_cast<CompositeBlockManager*>(block_managers_[dp_rank].get());
  composite->allocate_shared_for_sequence(sequence);
  if (!composite->allocate_sequence(sequence, sequence->num_tokens())) {
    composite->deallocate_for_sequence(sequence);
    sequence->reset();
    return false;
  }
  // add_shared_blocks (called inside allocate_shared_for_sequence) may have
  // already advanced kv_cache_tokens_num_ to num_shared_tokens on a prefix hit.
  // Only increment the remaining delta so the total reaches tokens().size().
  const size_t already_counted = sequence->kv_state().kv_cache_tokens_num();
  const size_t total_tokens = sequence->tokens().size();
  if (total_tokens > already_counted) {
    sequence->kv_state().incr_kv_cache_tokens_num(total_tokens -
                                                  already_counted);
  }
  return true;
}

bool BlockManagerPool::process_beam_search(Sequence* sequence, bool need_swap) {
  if (!sequence->check_beam_search()) {
    return true;
  }

  auto src_blocks = sequence->kv_state().src_blocks();
  if (src_blocks.size() == 0) {
    return true;
  }

  // when sequence need to swap the last block and no new block appended,
  // allocate a new block for this sequence
  if (need_swap && sequence->kv_state().need_swap()) {
    int32_t dp_rank = get_dp_rank(sequence);
    // Beam copy-on-write needs exactly one KV block; route through the
    // composite's typed allocation so it reaches the (possibly concurrency-
    // wrapped) KV leaf.
    auto* composite =
        static_cast<CompositeBlockManager*>(block_managers_[dp_rank].get());
    std::vector<Block> new_blocks =
        composite->allocate_blocks(BlockType::KV, 1);
    if (new_blocks.size() == 0) {
      return false;
    }
    swap_block_transfer_infos_[dp_rank].emplace_back(src_blocks.back().id(),
                                                     new_blocks[0].id());
    sequence->kv_state().process_beam_search(new_blocks[0]);
  } else {
    sequence->kv_state().process_beam_search(std::nullopt);
  }
  return true;
}

void BlockManagerPool::allocate_shared(Sequence* sequence) {
  if (!options_.enable_prefix_cache()) {
    return;
  }
  int32_t dp_rank = get_dp_rank(sequence);
  static_cast<CompositeBlockManager*>(block_managers_[dp_rank].get())
      ->allocate_shared_for_sequence(sequence);
}

void BlockManagerPool::cache(Sequence* sequence) {
  if (!options_.enable_prefix_cache()) {
    return;
  }
  int32_t dp_rank = get_dp_rank(sequence);
  static_cast<CompositeBlockManager*>(block_managers_[dp_rank].get())
      ->cache_for_sequence(sequence);
}

void BlockManagerPool::cache(Sequence* sequence, size_t num_tokens) {
  CHECK(sequence != nullptr);
  if (!options_.enable_prefix_cache()) {
    return;
  }
  int32_t dp_rank = get_dp_rank(sequence);
  // Fan out the in-batch publish to the prefix leaf (KV); the composite no-ops
  // for shapes without a prefix-capable KV leaf (DSV4 / xtensor).
  static_cast<CompositeBlockManager*>(block_managers_[dp_rank].get())
      ->cache_for_sequence(sequence, num_tokens);
}

float BlockManagerPool::get_gpu_cache_usage_perc() const {
  float perc = 0.0;
  for (int32_t i = 0; i < block_managers_.size(); ++i) {
    perc += block_managers_[i]->kv_cache_utilization();
  }
  return perc / block_managers_.size();
}

uint32_t BlockManagerPool::num_blocks() const { return options_.num_blocks(); }

int32_t BlockManagerPool::block_size() const { return options_.block_size(); }

void BlockManagerPool::reset_prefix_cache() {
  if (!options_.enable_prefix_cache()) {
    return;
  }
  for (auto& block_manager : block_managers_) {
    block_manager->reset_prefix_cache();
  }
}

std::vector<size_t> BlockManagerPool::num_blocks_in_prefix_cache() const {
  std::vector<size_t> num_blocks_in_prefix_cache(block_managers_.size());
  if (!options_.enable_prefix_cache()) {
    return num_blocks_in_prefix_cache;
  }
  for (size_t dp_rank = 0; dp_rank < block_managers_.size(); ++dp_rank) {
    num_blocks_in_prefix_cache[dp_rank] =
        block_managers_[dp_rank]->num_blocks_in_prefix_cache();
  }
  return num_blocks_in_prefix_cache;
}

std::vector<size_t> BlockManagerPool::num_free_blocks() const {
  std::vector<size_t> num_free_blocks(block_managers_.size());
  for (size_t dp_rank = 0; dp_rank < block_managers_.size(); ++dp_rank) {
    num_free_blocks[dp_rank] = block_managers_[dp_rank]->num_free_blocks();
  }
  return num_free_blocks;
}

std::vector<size_t> BlockManagerPool::num_used_blocks() const {
  std::vector<size_t> num_used_blocks(block_managers_.size());
  for (size_t dp_rank = 0; dp_rank < block_managers_.size(); ++dp_rank) {
    num_used_blocks[dp_rank] = block_managers_[dp_rank]->num_used_blocks();
  }
  return num_used_blocks;
}

double BlockManagerPool::kv_cache_utilization() const {
  int32_t dp_rank = get_manager_with_max_free_blocks();
  return block_managers_[dp_rank]->kv_cache_utilization();
}

// currently use only for profile, which not need prefix cache.
// If more often used in the future, can be integrated into deallocate function.
void BlockManagerPool::deallocate_without_cache(Sequence* sequence) {
  DCHECK(sequence != nullptr);
  int32_t dp_rank = get_dp_rank(sequence);
  // Release every leaf's blocks directly (no prefix-cache final flush). Routing
  // by Block::manager() inside composite::deallocate dispatches each block to
  // its owning leaf.
  for (const BlockType type : {BlockType::KV,
                               BlockType::SWA,
                               BlockType::C4,
                               BlockType::C128,
                               BlockType::SINGLE,
                               BlockType::LINEAR}) {
    const Slice<Block> blocks = sequence->kv_state().blocks(type);
    if (!blocks.empty()) {
      block_managers_[dp_rank]->deallocate(blocks);
    }
  }
  sequence->reset();
}

void BlockManagerPool::reserve_xtensor_padding_blocks() {
  if (!options_.enable_xtensor()) {
    return;
  }
  // The xtensor KV leaf is a CompositeBlockManager leaf now; fan the padding
  // reservation out through the composite (no dynamic_cast). Non-xtensor leaves
  // inherit the empty base default and are no-ops.
  for (auto& manager : block_managers_) {
    manager->reserve_xtensor_padding_blocks();
  }
  // Start prealloc thread once (PageAllocator is shared by all managers).
  PageAllocator::get_instance().start_prealloc_thread();
}

}  // namespace xllm

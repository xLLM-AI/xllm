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

#include "composite_block_manager.h"

#include <algorithm>
#include <limits>
#include <utility>

#include "block_manager_impl.h"
#include "concurrent_block_manager_impl.h"
#include "core/framework/config/kv_cache_config.h"
#include "framework/xtensor/xtensor_block_manager_impl.h"
#include "single_block_manager.h"
#include "sliding_window_block_manager.h"

namespace xllm {

namespace {

constexpr uint32_t kManagerTypeBlockManagerImpl = 0;
constexpr uint32_t kManagerTypeSlidingWindowBlockManager = 1;

uint32_t ceil_div(uint32_t numerator, uint32_t denominator) {
  CHECK_GT(denominator, 0u);
  return (numerator + denominator - 1) / denominator;
}

// Wrap a leaf in a concurrency adapter when sequence-level entry points may run
// off the scheduler thread (disagg PD / kvcache store prefill threadpools).
std::unique_ptr<BlockManager> maybe_concurrent(
    std::unique_ptr<BlockManager> leaf,
    const BlockManager::Options& options) {
  if (options.enable_disagg_pd() || options.enable_kvcache_store()) {
    return std::make_unique<ConcurrentBlockManagerImpl>(std::move(leaf));
  }
  return leaf;
}

// Build the KV leaf: an xtensor VMM manager when enable_xtensor, otherwise the
// flat free-list BlockManagerImpl. xtensor does not support prefix cache.
std::unique_ptr<BlockManager> make_kv_leaf(const BlockManager::Options& kv_opts,
                                           int32_t dp_rank) {
  if (!kv_opts.enable_xtensor()) {
    return std::make_unique<BlockManagerImpl>(kv_opts);
  }
  CHECK_GT(kv_opts.num_layers(), 0)
      << "num_layers must be set when enable_xtensor is true";
  CHECK_GT(kv_opts.slot_size(), 0)
      << "slot_size must be set when enable_xtensor is true";
  const size_t page_size =
      ::xllm::KVCacheConfig::get_instance().phy_page_granularity_size();
  // K and V are the same size in the current implementation, so divide by 2.
  const size_t block_mem_size =
      static_cast<size_t>(kv_opts.block_size()) * kv_opts.slot_size() / 2;
  return std::make_unique<XTensorBlockManagerImpl>(kv_opts,
                                                   kv_opts.num_layers(),
                                                   block_mem_size,
                                                   page_size,
                                                   dp_rank,
                                                   kv_opts.model_id());
}

}  // namespace

std::map<BlockType, CompositeBlockManager::LeafEntry> build_composite_leaves(
    const BlockManager::Options& options,
    int32_t dp_rank) {
  std::map<BlockType, CompositeBlockManager::LeafEntry> leaves;

  if (options.manager_types().empty()) {
    // Normal / Qwen / xtensor model: a single KV leaf. It is the admission
    // source; prefix cache is on only for the flat (non-xtensor) KV leaf.
    BlockManager::Options kv_opts = options;
    kv_opts.block_type(BlockType::KV);
    leaves.emplace(
        BlockType::KV,
        CompositeBlockManager::LeafEntry{
            maybe_concurrent(make_kv_leaf(kv_opts, dp_rank), options),
            /*participates_in_admission=*/true,
            /*supports_prefix_cache=*/
            options.enable_prefix_cache() && !options.enable_xtensor()});
    return leaves;
  }

  // DSV4: SWA + compressed (C4 / C128) leaves, derived from manager_types /
  // compress_ratios (kept identical to the previous in-composite construction).
  const size_t n = options.manager_types().size();
  CHECK_EQ(n, options.compress_ratios().size())
      << "manager_types and compress_ratios must have the same size";
  CHECK_GT(n, 0u) << "Composite requires at least one sub-manager";

  for (size_t i = 0; i < n; ++i) {
    const uint32_t type = options.manager_types()[i];
    const uint32_t compress_ratio = options.compress_ratios()[i];
    BlockManager::Options opts = options;

    if (type == kManagerTypeBlockManagerImpl) {
      opts.block_size(static_cast<uint32_t>(options.block_size()) *
                      compress_ratio);
      opts.num_blocks(static_cast<uint32_t>(options.num_blocks()) /
                      compress_ratio);
      CHECK(compress_ratio == 4 || compress_ratio == 128)
          << "unexpected compress_ratio " << compress_ratio
          << " for composite BlockManagerImpl sub-manager";
      const BlockType key =
          compress_ratio == 4 ? BlockType::C4 : BlockType::C128;
      opts.block_type(key);
      // Compressed groups are admission sources but do not serve prefix cache.
      leaves.emplace(key,
                     CompositeBlockManager::LeafEntry{
                         maybe_concurrent(
                             std::make_unique<BlockManagerImpl>(opts), options),
                         /*participates_in_admission=*/true,
                         /*supports_prefix_cache=*/false});
    } else if (type == kManagerTypeSlidingWindowBlockManager) {
      const uint32_t swa_blocks_per_seq = options.swa_blocks_per_seq();
      CHECK_GT(swa_blocks_per_seq, 0u) << "swa_blocks_per_seq must be positive";
      CHECK_GT(options.block_size(), 0) << "block_size must be positive";
      const uint32_t sliding_window_size =
          std::max(options.sliding_window_size(), 1u);
      const uint32_t max_seqs = std::max(options.max_seqs_per_batch(), 1u);
      const uint32_t burst_blocks =
          ceil_div(std::max(options.max_tokens_per_batch(), 1u),
                   static_cast<uint32_t>(options.block_size()));
      const uint32_t swa_total_blocks =
          swa_blocks_per_seq * max_seqs + burst_blocks + max_seqs + 2;
      opts.num_blocks(swa_total_blocks)
          .swa_blocks_per_seq(swa_blocks_per_seq)
          .sliding_window_size(sliding_window_size)
          .block_type(BlockType::SWA);
      // SWA neither participates in admission nor serves prefix cache.
      leaves.emplace(
          BlockType::SWA,
          CompositeBlockManager::LeafEntry{
              maybe_concurrent(
                  std::make_unique<SlidingWindowBlockManager>(opts), options),
              /*participates_in_admission=*/false,
              /*supports_prefix_cache=*/false});
    } else {
      LOG(FATAL) << "Unknown manager_type " << type;
    }
  }
  return leaves;
}

CompositeBlockManager::CompositeBlockManager(
    std::map<BlockType, LeafEntry> leaves)
    : BlockManager(BlockManager::Options()), leaves_(std::move(leaves)) {
  CHECK(!leaves_.empty()) << "CompositeBlockManager requires at least one leaf";
}

BlockManager* CompositeBlockManager::leaf_of(BlockType type) const {
  auto it = leaves_.find(type);
  return it == leaves_.end() ? nullptr : it->second.leaf.get();
}

bool CompositeBlockManager::allocate_sequence(Sequence* seq,
                                              size_t num_tokens) {
  if (seq == nullptr) {
    return false;
  }
  KVCacheState& kv_state = seq->kv_state();

  // Each leaf grows its own block_type() blocks for the sequence and returns
  // the blocks it newly allocated; it does not insert them. The composite
  // stages them keyed by BlockType and commits only after every leaf succeeds.
  // Rollback = deallocate every staged run (the sequence is never grown before
  // commit, so existing blocks stay intact).
  std::vector<std::pair<BlockType, std::vector<Block>>> staged;

  for (auto& [type, entry] : leaves_) {
    std::optional<std::vector<Block>> blocks =
        entry.leaf->allocate_for_sequence(seq, num_tokens);
    if (!blocks.has_value()) {
      for (auto& [staged_type, staged_blocks] : staged) {
        leaf_of(staged_type)->deallocate(staged_blocks);
      }
      return false;
    }
    if (!blocks->empty()) {
      staged.emplace_back(type, std::move(*blocks));
    }
  }

  // Commit: append staged blocks to the sequence under each leaf's block type.
  for (auto& [type, blocks] : staged) {
    kv_state.add_blocks(type, blocks);
  }
  // Post-commit: release blocks that have slid out of the window (SWA only;
  // other leaves are a no-op). Runs only after a fully successful round, so a
  // failed allocation never releases the sequence's existing blocks.
  for (auto& [type, entry] : leaves_) {
    entry.leaf->release_out_of_window(seq);
  }
  return true;
}

void CompositeBlockManager::deallocate_for_sequence(Sequence* seq) {
  if (seq == nullptr) {
    return;
  }
  // Publish prefix cache first (prefix leaves only), then release every leaf's
  // blocks by key. seq->reset() stays with the pool caller.
  cache_for_sequence(seq);
  for (auto& [type, entry] : leaves_) {
    entry.leaf->deallocate(seq->kv_state().blocks(type));
  }
}

void CompositeBlockManager::allocate_shared_for_sequence(Sequence* seq) {
  // Today only the flat-KV model supports prefix cache. Other shapes (DSV4 /
  // xtensor) leave the KV entry's supports_prefix_cache=false (or have no KV
  // leaf at all), so this is a no-op for them.
  if (seq == nullptr) {
    return;
  }
  auto it = leaves_.find(BlockType::KV);
  if (it == leaves_.end() || !it->second.supports_prefix_cache) {
    return;
  }
  BlockManager& kv_leaf = *it->second.leaf;
  seq->update_block_hashes(static_cast<uint32_t>(kv_leaf.block_size()),
                           kv_leaf.options().hasher_type());
  KVCacheState& kv_state = seq->kv_state();
  const auto existed = kv_state.blocks(BlockType::KV)
                           .slice(0, kv_state.shared_blocks_num(BlockType::KV));
  std::vector<Block> shared = kv_leaf.allocate_shared(
      seq->tokens(), existed, seq->mm_data(), seq->block_hashes());
  seq->add_shared_blocks(BlockType::KV, std::move(shared));
}

void CompositeBlockManager::cache_for_sequence(Sequence* seq) {
  if (seq == nullptr) {
    return;
  }
  auto it = leaves_.find(BlockType::KV);
  if (it == leaves_.end() || !it->second.supports_prefix_cache) {
    return;
  }
  BlockManager& kv_leaf = *it->second.leaf;
  seq->update_block_hashes(static_cast<uint32_t>(kv_leaf.block_size()),
                           kv_leaf.options().hasher_type());
  KVCacheState& kv_state = seq->kv_state();
  std::vector<Block>* blocks = kv_state.mutable_blocks(BlockType::KV);
  kv_leaf.cache(seq->cached_tokens(),
                *blocks,
                kv_state.shared_blocks_num(BlockType::KV),
                seq->mm_data(),
                seq->block_hashes());
}

void CompositeBlockManager::cache_for_sequence(Sequence* seq,
                                               size_t num_tokens) {
  if (seq == nullptr) {
    return;
  }
  auto it = leaves_.find(BlockType::KV);
  if (it == leaves_.end() || !it->second.supports_prefix_cache) {
    return;
  }
  BlockManager& kv_leaf = *it->second.leaf;
  KVCacheState& kv_state = seq->kv_state();
  const size_t block_size = kv_leaf.block_size();
  // Clamp to blocks actually allocated and to the sequence's own tokens; the
  // last partial block is dropped by the prefix-cache insert.
  const size_t available_tokens_num =
      std::min({num_tokens,
                kv_state.num_blocks(BlockType::KV) * block_size,
                seq->tokens().size()});
  const size_t existed_shared_blocks_num =
      kv_state.shared_blocks_num(BlockType::KV);
  if (available_tokens_num <= existed_shared_blocks_num * block_size) {
    return;
  }
  seq->update_block_hashes(static_cast<uint32_t>(block_size),
                           kv_leaf.options().hasher_type());
  std::vector<Block>* blocks = kv_state.mutable_blocks(BlockType::KV);
  CHECK_GE(blocks->size(), existed_shared_blocks_num);
  kv_leaf.cache(seq->tokens().slice(0, available_tokens_num),
                *blocks,
                existed_shared_blocks_num,
                seq->mm_data(),
                seq->block_hashes());
}

std::vector<Block> CompositeBlockManager::allocate_blocks(BlockType type,
                                                          size_t num_blocks) {
  BlockManager* leaf = leaf_of(type);
  CHECK(leaf != nullptr) << "CompositeBlockManager has no leaf for block type "
                         << static_cast<int>(type);
  return leaf->allocate(num_blocks);
}

void CompositeBlockManager::deallocate(const Slice<Block>& blocks) {
  // Route each run of blocks to its owning leaf by Block::manager().
  if (blocks.empty()) {
    return;
  }
  size_t run_start = 0;
  BlockManager* run_manager = nullptr;
  for (size_t i = 0; i < blocks.size(); ++i) {
    const auto& block = blocks[i];
    if (!block.is_valid()) {
      if (run_manager != nullptr) {
        run_manager->deallocate(blocks.slice(run_start, i));
        run_manager = nullptr;
      }
      run_start = i + 1;
      continue;
    }
    BlockManager* manager = block.manager();
    CHECK(manager != nullptr)
        << "CompositeBlockManager got a valid block without owner manager";
    if (run_manager == nullptr) {
      run_manager = manager;
      run_start = i;
    } else if (run_manager != manager) {
      run_manager->deallocate(blocks.slice(run_start, i));
      run_manager = manager;
      run_start = i;
    }
  }
  if (run_manager != nullptr) {
    run_manager->deallocate(blocks.slice(run_start, blocks.size()));
  }
}

std::vector<Block> CompositeBlockManager::allocate(size_t /*num_blocks*/) {
  NOT_IMPLEMENTED();
  return {};
}

std::optional<std::vector<Block>> CompositeBlockManager::allocate_for_sequence(
    Sequence* /*seq*/,
    size_t /*num_tokens*/) {
  // The composition is driven via allocate_sequence(), which fans out to each
  // leaf's allocate_for_sequence. The leaf-level entry point is meaningless on
  // the composite itself.
  NOT_IMPLEMENTED();
  return std::nullopt;
}

std::vector<Block> CompositeBlockManager::allocate_shared(
    const Slice<int32_t>& /*tokens_ids*/,
    const Slice<Block>& /*existed_shared_blocks*/,
    const MMData& /*mm_data*/,
    const Slice<XXH3Key>& /*block_hashes*/) {
  NOT_IMPLEMENTED();
  return {};
}

void CompositeBlockManager::cache(const Slice<int32_t>& /*token_ids*/,
                                  std::vector<Block>& /*blocks*/,
                                  size_t /*existed_shared_blocks_num*/,
                                  const MMData& /*mm_data*/,
                                  const Slice<XXH3Key>& /*block_hashes*/) {
  NOT_IMPLEMENTED();
}

void CompositeBlockManager::cache(const std::vector<Block>& /*blocks*/) {
  NOT_IMPLEMENTED();
}

void CompositeBlockManager::reset_prefix_cache() {
  for (auto& [type, entry] : leaves_) {
    entry.leaf->reset_prefix_cache();
  }
}

size_t CompositeBlockManager::num_blocks_in_prefix_cache() const {
  size_t total = 0;
  for (const auto& [type, entry] : leaves_) {
    total += entry.leaf->num_blocks_in_prefix_cache();
  }
  return total;
}

const CompositeBlockManager::LeafEntry* CompositeBlockManager::capacity_leaf()
    const {
  const LeafEntry* chosen = nullptr;
  for (const auto& [type, entry] : leaves_) {
    if (!entry.participates_in_admission) {
      continue;
    }
    // Smallest block_size = finest granularity = closest to the base block the
    // scheduler assumes. For normal models this is the lone KV leaf; for DSV4
    // (C4 bs=4*base, C128 bs=128*base) this is C4.
    if (chosen == nullptr ||
        entry.leaf->block_size() < chosen->leaf->block_size()) {
      chosen = &entry;
    }
  }
  return chosen;
}

size_t CompositeBlockManager::num_free_blocks() const {
  // Scheduler-facing capacity is reported as a single admission leaf's raw
  // block count (see capacity_leaf): mixing leaves of different block_size
  // would make num_free * block_size() meaningless. This reproduces the
  // pre-refactor single-group (`sub_managers_[1]`) semantics and is identical
  // to the old BlockManagerImpl for normal models (KV is the only admission
  // leaf).
  const LeafEntry* leaf = capacity_leaf();
  return leaf == nullptr ? 0 : leaf->leaf->num_free_blocks();
}

size_t CompositeBlockManager::num_used_blocks() const {
  const LeafEntry* leaf = capacity_leaf();
  return leaf == nullptr ? 0 : leaf->leaf->num_used_blocks();
}

double CompositeBlockManager::kv_cache_utilization() const {
  const size_t total = num_total_blocks();
  if (total == 0) {
    return 0.0;
  }
  return static_cast<double>(num_used_blocks()) / static_cast<double>(total);
}

void CompositeBlockManager::free(int32_t /*block_id*/) {
  LOG(FATAL) << "CompositeBlockManager::free should not be called";
}

Block CompositeBlockManager::allocate() {
  LOG(FATAL) << "CompositeBlockManager::allocate should not be called";
  return Block();
}

size_t CompositeBlockManager::num_total_blocks() const {
  const LeafEntry* leaf = capacity_leaf();
  return leaf == nullptr ? 0 : leaf->leaf->num_total_blocks();
}

void CompositeBlockManager::reserve_xtensor_padding_blocks() {
  for (auto& [type, entry] : leaves_) {
    entry.leaf->reserve_xtensor_padding_blocks();
  }
}

}  // namespace xllm

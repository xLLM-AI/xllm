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

#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "block_manager.h"

namespace xllm {

// Generic composition of multiple BlockManager leaves keyed by BlockType. The
// map key decides which KVCacheState slot a leaf's blocks land in; the leaf
// itself is type-free. The composite is the only block-side class that touches
// Sequence: it extracts parameters, drives the leaves' pure planners and
// type-free primitives, and writes results back by key. It holds no
// model-specific logic.
class CompositeBlockManager : public BlockManager {
 public:
  // Per-leaf entry. The admission / prefix roles live here (on the composite),
  // not on the leaf: the same BlockManagerImpl is a prefix-capable admission
  // leaf under the KV key but a non-prefix admission leaf under C4/C128.
  struct LeafEntry {
    std::unique_ptr<BlockManager> leaf;
    bool participates_in_admission = true;
    bool supports_prefix_cache = false;
  };

  explicit CompositeBlockManager(std::map<BlockType, LeafEntry> leaves);
  ~CompositeBlockManager() override = default;

  bool is_composite() const override { return true; }

  // —— Sequence-level orchestration (the only Sequence-aware surface) ——
  // Drives every leaf's allocate_for_sequence(seq, num_tokens), stages the
  // blocks each leaf returns, and commits them into the sequence under the
  // leaf's block_type() once all succeed (rolling back on any failure).
  // Distinct from the leaf-level BlockManager::allocate_for_sequence, which
  // returns the blocks for a single leaf without inserting them.
  bool allocate_sequence(Sequence* seq, size_t num_tokens);
  void deallocate_for_sequence(Sequence* seq);
  void allocate_shared_for_sequence(Sequence* seq);
  void cache_for_sequence(Sequence* seq);
  void cache_for_sequence(Sequence* seq, size_t num_tokens);

  // Typed block-level allocation routed to the leaf under `type`. Used by the
  // pool for beam copy-on-write (which needs exactly one KV block).
  std::vector<Block> allocate_blocks(BlockType type, size_t num_blocks);

  // Type-ambiguous block-level primitives are not meaningful on a composition.
  void deallocate(const Slice<Block>& blocks) override;
  std::vector<Block> allocate(size_t num_blocks) override;
  // Leaf-level growth is not meaningful on the composition: the pool drives the
  // composite via allocate_sequence() (which fans out to each leaf's
  // allocate_for_sequence). Satisfies the pure-virtual base; never called.
  std::optional<std::vector<Block>> allocate_for_sequence(
      Sequence* seq,
      size_t num_tokens) override;
  std::vector<Block> allocate_shared(
      const Slice<int32_t>& tokens_ids,
      const Slice<Block>& existed_shared_blocks = {},
      const MMData& mm_data = MMData(),
      const Slice<XXH3Key>& block_hashes = {},
      size_t* matched_tokens = nullptr) override;
  void cache(const Slice<int32_t>& token_ids,
             std::vector<Block>& blocks,
             size_t existed_shared_blocks_num = 0,
             const MMData& mm_data = MMData(),
             const Slice<XXH3Key>& block_hashes = {}) override;
  void cache(const std::vector<Block>& blocks) override;

  // RL sleep/wakeup: fan out to every leaf (non-prefix leaves are a no-op).
  void reset_prefix_cache() override;

  // Stats reported from the single capacity leaf (see capacity_leaf()).
  size_t num_blocks_in_prefix_cache() const override;  // sum over all leaves
  size_t num_free_blocks() const override;             // from capacity leaf
  size_t num_used_blocks() const override;             // from capacity leaf
  double kv_cache_utilization() const override;
  void free(int32_t block_id) override;
  Block allocate() override;
  size_t num_total_blocks() const override;  // from capacity leaf

  void reserve_xtensor_padding_blocks() override;

  size_t num_sub_managers() const { return leaves_.size(); }

 private:
  // Test-only reach into the LINEAR leaf, mirroring the friend pattern the leaf
  // itself uses: the test peer seeds checkpoints the way the scheduler does
  // while resolving cache ops. No production caller reaches a leaf by type --
  // the composite drives every leaf through its own type-free orchestration --
  // so leaf_of stays private.
  friend class BlockManagerPoolTestPeer;

  // Leaf serving `type`, or nullptr if none. Internal helper for the
  // composite's own orchestration; not a Sequence-aware or externally driven
  // surface.
  BlockManager* leaf_of(BlockType type) const;

  // The single admission leaf whose raw block count defines the pool's
  // scheduler-facing capacity unit. Schedulers treat num_free/used/total_blocks
  // as counts of base (block_size()) blocks, so we must report one leaf's raw
  // count rather than mixing leaves of different block sizes. Picks the
  // admission leaf with the smallest block_size (the finest-grained, closest to
  // base): KV for normal models, C4 for DSV4. Reproduces the pre-refactor
  // single-`sub_managers_[1]` capacity semantics. nullptr if none.
  const LeafEntry* capacity_leaf() const;

  std::map<BlockType, LeafEntry> leaves_;
};

// Build the leaf map for one DP rank from the pool options (per-model:
// normal/Qwen -> {KV, SINGLE}; DSV4 -> {SWA, C4, C128, SINGLE}; xtensor ->
// {KV(XTensorBlockManagerImpl), SINGLE}). Leaves are wrapped in
// ConcurrentBlockManagerImpl when disagg-PD / kvcache store is enabled. The
// SINGLE entry is appended by the caller (pool). dp_rank is needed by the
// xtensor KV leaf (per-rank VMM page pool).
std::map<BlockType, CompositeBlockManager::LeafEntry> build_composite_leaves(
    const BlockManager::Options& options,
    int32_t dp_rank = 0);

}  // namespace xllm

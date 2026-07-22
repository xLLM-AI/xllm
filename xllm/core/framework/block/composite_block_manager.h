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
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "block_manager.h"

namespace xllm {

// Composition of BlockManager leaves keyed by BlockType. The map key decides
// which KVCacheState slot a leaf's blocks land in; the leaf itself is
// type-free. The composite is the only block-side class that touches
// Sequence.
class CompositeBlockManager : public BlockManager {
 public:
  // The admission / prefix roles live on the composite, not on the leaf: the
  // same BlockManagerImpl is prefix-capable under KV but non-prefix under
  // C4/C128.
  struct LeafEntry {
    std::unique_ptr<BlockManager> leaf;
    bool participates_in_admission = true;
    bool supports_prefix_cache = false;
  };

  // Which prefix-cache-supported leaf shape this composite carries. Classified
  // once at construction and cached in `combination_`; the sequence-level
  // orchestrators dispatch on it to pick the trim / mount strategy.
  //   FLAT_KV        Plain KV (normal / Qwen). No trim.
  //   FLAT_KV_LINEAR KV + LINEAR restore (Qwen3.5-Next GDN). Composite clamps
  //                  KV to LINEAR's recoverable budget and pulls out its
  //                  deepest checkpoint as a restore source.
  //   SWA_COMPRESSED SWA + C4 + C128 (DSV4). Cross-leaf min, C128-stride
  //                  clamp, SWA tail-continuity, exact-repeat pop.
  //   UNSUPPORTED    Prefix cache off (xtensor / --enable_prefix_cache=false).
  enum class LeafCombination : int8_t {
    FLAT_KV,
    FLAT_KV_LINEAR,
    SWA_COMPRESSED,
    UNSUPPORTED,
  };

  explicit CompositeBlockManager(std::map<BlockType, LeafEntry> leaves);
  ~CompositeBlockManager() override = default;

  bool is_composite() const override { return true; }

  // Sequence-level orchestration (the only Sequence-aware surface). Drives
  // each leaf's own primitives and writes results back into KVCacheState
  // under the leaf's block_type().
  bool allocate_sequence(Sequence* seq, size_t num_tokens);
  void deallocate_for_sequence(Sequence* seq);
  void allocate_shared_for_sequence(Sequence* seq);
  void cache_for_sequence(Sequence* seq);
  void cache_for_sequence(Sequence* seq, size_t num_tokens);

  // Typed block-level allocation routed to the leaf under `type`. Used for
  // beam copy-on-write.
  std::vector<Block> allocate_blocks(BlockType type, size_t num_blocks);

  // Type-ambiguous block-level primitives are not meaningful on a composition.
  void deallocate(const Slice<Block>& blocks) override;
  std::vector<Block> allocate(size_t num_blocks) override;
  std::optional<std::vector<Block>> allocate_for_sequence(
      Sequence* seq,
      size_t num_tokens) override;
  std::vector<Block> allocate_shared(
      const Slice<int32_t>& tokens_ids,
      const Slice<Block>& existed_shared_blocks = {},
      const MMData& mm_data = MMData(),
      const Slice<XXH3Key>& block_hashes = {}) override;
  void cache(const Slice<int32_t>& token_ids,
             std::vector<Block>& blocks,
             size_t existed_shared_blocks_num = 0,
             const MMData& mm_data = MMData(),
             const Slice<XXH3Key>& block_hashes = {}) override;
  void cache(const std::vector<Block>& blocks) override;

  // RL sleep/wakeup: fan out to every leaf (non-prefix leaves are a no-op).
  void reset_prefix_cache() override;

  // Stats reported from the single capacity leaf (see capacity_leaf()).
  size_t num_blocks_in_prefix_cache() const override;
  size_t num_free_blocks() const override;
  size_t num_used_blocks() const override;
  double kv_cache_utilization() const override;
  void free(int32_t block_id) override;
  Block allocate() override;
  size_t num_total_blocks() const override;

  void reserve_xtensor_padding_blocks() override;

  size_t num_sub_managers() const { return leaves_.size(); }

 private:
  friend class BlockManagerPoolTestPeer;

  BlockManager* leaf_of(BlockType type) const;

  // The admission leaf whose raw block count defines the pool's
  // scheduler-facing capacity unit. Picks the finest-grained admission leaf
  // (smallest block_size): KV for normal models, C4 for DSV4.
  const LeafEntry* capacity_leaf() const;

  static LeafCombination classify_leaf_combination(
      const std::map<BlockType, LeafEntry>& leaves);

  // Pre-grow / final-flush cache hook. Inserts every leaf's newly-forwarded
  // full blocks into its own prefix cache incrementally (cursor lives in
  // KVCacheState::num_cached_blocks). Only fires for leaves that carry token
  // cache and are not the KV leaf -- KV has its own final-flush semantics
  // via cache_for_sequence at deallocate time; SINGLE / LINEAR hold no token
  // cache.
  void cache_full_blocks_for_sequence(Sequence* seq);

  std::map<BlockType, LeafEntry> leaves_;
  LeafCombination combination_;
};

// Build the leaf map for one DP rank. Per model:
//   normal / Qwen -> {KV, SINGLE}
//   DSV4          -> {SWA, C4, C128, SINGLE}
//   xtensor       -> {KV(XTensorBlockManagerImpl), SINGLE}
// Leaves are wrapped in ConcurrentBlockManagerImpl for disagg-PD / kvcache
// store. SINGLE is appended by the pool caller. dp_rank is used by the
// xtensor KV leaf (per-rank VMM page pool).
std::map<BlockType, CompositeBlockManager::LeafEntry> build_composite_leaves(
    const BlockManager::Options& options,
    int32_t dp_rank = 0);

}  // namespace xllm

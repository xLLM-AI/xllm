/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "block.h"
#include "common/global_flags.h"
#include "common/macros.h"
#include "common/metrics.h"
#include "common/types.h"
#include "core/framework/multimodal/mm_data.h"
#include "framework/prefix_cache/prefix_cache.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"
#include "util/timer.h"

namespace xllm {

class BlockManager {
 public:
  struct Options {
    PROPERTY(uint32_t, num_blocks) = 0;
    PROPERTY(int32_t, block_size) = 0;
    PROPERTY(bool, enable_prefix_cache) = true;
    PROPERTY(bool, enable_disagg_pd) = false;
    // Token-level sliding window size for composite SWA allocation.
    PROPERTY(uint32_t, sliding_window_size) = 0;
    // Base SWA/cache-state block rows retained per sequence.
    PROPERTY(uint32_t, swa_blocks_per_seq) = 0;
    // Scheduler token budget used to size the shared SWA burst pool.
    PROPERTY(uint32_t, max_tokens_per_batch) = 0;
    // For CompositeBlockManager (passed from upstream).
    PROPERTY(std::vector<uint32_t>, manager_types) = {};
    PROPERTY(std::vector<uint32_t>, compress_ratios) = {};
    PROPERTY(uint32_t, max_seqs_per_batch) = 0;
    // Hasher type bound to the engine (TEXT for LLM, MM for VLM).
    PROPERTY(BlockHasherType, hasher_type) = BlockHasherType::TEXT;
    // The block category used as the composite's map key for this leaf. The
    // leaf itself is type-free (no block_type() accessor); the spec builder
    // carries this value to decide the map key. Flat KV uses KV.
    PROPERTY(BlockType, block_type) = BlockType::KV;
    // Whether the kvcache-store (host offload) path is enabled. Composite
    // leaves are wrapped in ConcurrentBlockManagerImpl when this (or
    // enable_disagg_pd) is set.
    PROPERTY(bool, enable_kvcache_store) = false;
    // Whether host prefix-cache offload (host_blocks_factor > 1) is enabled.
    // The D2H offload-completion callback frees device/host blocks from a folly
    // executor thread, so leaves are wrapped in ConcurrentBlockManagerImpl when
    // this is set to make those mutations thread-safe against the scheduler.
    PROPERTY(bool, enable_host_offload) = false;
    // xtensor (VMM) KV leaf parameters. When enable_xtensor is set, the KV leaf
    // is an XTensorBlockManagerImpl instead of a flat BlockManagerImpl; these
    // carry the construction args the spec builder needs.
    PROPERTY(bool, enable_xtensor) = false;
    PROPERTY(int64_t, num_layers) = 0;
    PROPERTY(int64_t, slot_size) = 0;
    PROPERTY(std::string, model_id);
    // Linear-state (Qwen3.5 GDN) resource leaf. When enable_linear_state is
    // set, build_composite_leaves appends a LINEAR leaf on top of the KV
    // family. linear_state_num_slots is the total physical slot count [0, N)
    // for the unified slot pool. Both are ignored unless linear state is on.
    PROPERTY(bool, enable_linear_state) = false;
    PROPERTY(int32_t, linear_state_num_slots) = 0;
    // Linear-state checkpoint stride in tokens (one prefill chunk), forwarded
    // to the LINEAR leaf's prefix cache as its hash-domain step. Ignored unless
    // linear state is on; -1 makes the linear probe bail out.
    PROPERTY(int32_t, linear_chunk_stride) = -1;
    // Number of speculative tokens for MTP decode (passed from
    // runtime::Options). Used by CompositeBlockManager to adjust SWA block
    // release accounting.
    PROPERTY(uint32_t, num_speculative_tokens) = 0;
  };

  explicit BlockManager(Options options) : options_(options) {}
  virtual ~BlockManager() = default;

  virtual void deallocate(const Slice<Block>& blocks) = 0;

  virtual std::vector<Block> allocate(size_t num_blocks) = 0;

  // `matched_tokens`, when non-null, receives the matched prefix length in
  // TOKENS (forwarded straight to PrefixCache::match: the flat-KV leaf writes
  // blocks.size() * block_size; the LINEAR leaf writes the recoverable
  // checkpoint prefix in its own chunk-strided hash domain). Callers that only
  // want the blocks leave it null.
  virtual std::vector<Block> allocate_shared(
      const Slice<int32_t>& token_ids,
      const Slice<Block>& existed_shared_blocks = {},
      const MMData& mm_data = MMData(),
      const Slice<XXH3Key>& block_hashes = {},
      size_t* matched_tokens = nullptr) = 0;

  virtual void cache(const Slice<int32_t>& token_ids,
                     std::vector<Block>& blocks,
                     size_t existed_shared_blocks_num = 0,
                     const MMData& mm_data = MMData(),
                     const Slice<XXH3Key>& block_hashes = {}) = 0;
  virtual void cache(const std::vector<Block>& blocks) = 0;

  virtual size_t num_blocks_in_prefix_cache() const = 0;
  virtual size_t num_free_blocks() const = 0;
  virtual size_t num_used_blocks() const = 0;
  virtual double kv_cache_utilization() const = 0;

  // Evict all entries from the prefix cache (returning their blocks to the free
  // pool). Used by RL sleep/wakeup: after deep sleep the KV physical memory is
  // discarded, so any cached prefix would point to garbage and must be dropped.
  virtual void reset_prefix_cache() {}

  // get the options for the block manager
  const Options& options() const { return options_; }

  // get number of slots per block
  size_t block_size() const { return options_.block_size(); }

  // The block category this leaf serves (KV / SWA / C4 / C128 / SINGLE). A leaf
  // reads its own held-block count from the sequence under this type, and the
  // CompositeBlockManager inserts the returned blocks into the sequence under
  // this type. Carried via Options by the spec builder.
  BlockType block_type() const { return options_.block_type(); }

  // call BlockManager to free block used by Block.
  virtual void free(int32_t block_id) = 0;

  // allocate a list of blocks, used for unit test
  // virtual std::vector<Block> allocate(uint32_t n_blocks) = 0;

  // allocate a block, used for unit test
  virtual Block allocate() = 0;

  // get number of total blocks
  virtual size_t num_total_blocks() const = 0;

  // True only for CompositeBlockManager. Leaves and flat managers return false.
  virtual bool is_composite() const { return false; }

  // —— Sequence-level growth interface (pure: every concrete manager defines
  // its own policy) ——
  // Grows this manager's block_type() blocks for a sequence and returns the
  // blocks it newly allocated this round (empty when nothing is needed),
  // std::nullopt on shortage. The flat-KV default lives in BlockManagerImpl;
  // SlidingWindow / Single override; CompositeBlockManager fans out to leaves.
  // Most managers do NOT insert into the sequence -- the composite stages the
  // returned blocks and commits them under block_type() (the SWA leaf also
  // releases its own slid-out blocks in place as part of this call).
  virtual std::optional<std::vector<Block>> allocate_for_sequence(
      Sequence* seq,
      size_t num_tokens) = 0;

  // Sliding-window hook: release leading blocks that have slid out of the
  // window. The composite calls this on every leaf AFTER a successful
  // allocate_sequence commit; non-SWA leaves keep the empty default (no-op).
  // Running post-commit means a failed round never releases existing blocks.
  virtual void release_out_of_window(Sequence* /*seq*/) {}

  // Post-construction init hook: only the xtensor leaf needs it (KV tensors
  // must be created on the worker before VMM physical pages can be mapped to
  // reserve the padding block). Empty base default; the composite fans it out
  // to every leaf, so non-xtensor leaves are a no-op. This keeps the
  // out-of-band timing free of any dynamic_cast through the composite.
  virtual void reserve_xtensor_padding_blocks() {}

 protected:
  // the options for the block manager
  Options options_;
};

}  // namespace xllm

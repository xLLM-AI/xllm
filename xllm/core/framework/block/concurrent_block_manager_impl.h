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

#include <memory>
#include <mutex>

#include "block_manager.h"

namespace xllm {

// Concurrency adapter: wraps an inner leaf BlockManager and serializes every
// entry point under one recursive mutex. Used for modes whose sequence-level
// calls run off the scheduler thread (disagg PD / kvcache store prefill
// threadpools).
//
// Composition, not inheritance: the inner manager is always a leaf
// (BlockManagerImpl / SingleBlockManager / ...), never a CompositeBlockManager.
// The composite constructs this wrapper around the leaves that need it.
class ConcurrentBlockManagerImpl : public BlockManager {
 public:
  explicit ConcurrentBlockManagerImpl(std::unique_ptr<BlockManager> inner);
  ~ConcurrentBlockManagerImpl() override = default;

  // Block-level primitives.
  std::vector<Block> allocate(size_t num_blocks) override;
  Block allocate() override;
  void deallocate(const Slice<Block>& blocks) override;
  void free(int32_t block_id) override;

  std::vector<Block> allocate_shared(
      const Slice<int32_t>& token_ids,
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

  // Sequence-level growth forwards under the lock (touches the inner leaf's
  // free list / pool).
  std::optional<std::vector<Block>> allocate_for_sequence(
      Sequence* seq,
      size_t num_tokens) override;
  void release_out_of_window(Sequence* seq) override;

  void reset_prefix_cache() override;

  size_t num_blocks_in_prefix_cache() const override;
  size_t num_free_blocks() const override;
  size_t num_used_blocks() const override;
  size_t num_total_blocks() const override;
  double kv_cache_utilization() const override;

 private:
  std::unique_ptr<BlockManager> inner_;  // always a leaf, never a composite
  // Recursive: Block dtors call free() through this wrapper (we rewrite each
  // returned Block's manager_ to this), and inner_->allocate() may trigger
  // prefix-cache eviction which destroys cached Blocks while we still hold the
  // lock from the outer wrapper call.
  mutable std::recursive_mutex mutex_;
};

}  // namespace xllm

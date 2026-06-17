/* Copyright 2025 The xLLM Authors. All Rights Reserved.
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
    PROPERTY(bool, enable_cache_upload) = false;
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
  };

  explicit BlockManager(Options options) : options_(options) {}
  virtual ~BlockManager() = default;

  virtual void deallocate(const Slice<Block>& blocks) = 0;

  virtual std::vector<Block> allocate(size_t num_blocks) = 0;

  virtual std::vector<Block> allocate_shared(
      const Slice<int32_t>& token_ids,
      const Slice<Block>& existed_shared_blocks = {},
      const MMData& mm_data = MMData()) = 0;

  virtual void cache(const Slice<int32_t>& token_ids,
                     std::vector<Block>& blocks,
                     size_t existed_shared_blocks_num = 0,
                     const MMData& mm_data = MMData()) = 0;
  virtual void cache(const std::vector<Block>& blocks) = 0;

  // get merged all dp rank KVCacheEvent
  virtual void get_merged_kvcache_event(KvCacheEvent* event) const = 0;

  virtual size_t num_blocks_in_prefix_cache() const = 0;
  virtual size_t num_free_blocks() const = 0;
  virtual size_t num_used_blocks() const = 0;
  virtual double kv_cache_utilization() const = 0;

  // get the options for the block manager
  const Options& options() const { return options_; }

  // get number of slots per block
  size_t block_size() const { return options_.block_size(); }

  // call BlockManager to free block used by Block.
  virtual void free(int32_t block_id) = 0;

  // allocate a list of blocks, used for unit test
  // virtual std::vector<Block> allocate(uint32_t n_blocks) = 0;

  // allocate a block, used for unit test
  virtual Block allocate() = 0;

  // get number of total blocks
  virtual size_t num_total_blocks() const = 0;

  // CompositeBlockManager: Pool calls these when is_composite() is true.
  virtual bool is_composite() const { return false; }
  virtual bool allocate_for_sequence(Sequence* seq, size_t num_tokens) {
    return false;
  }
  virtual void deallocate_sequence(Sequence* seq) {}

 protected:
  // the options for the block manager
  Options options_;
};

}  // namespace xllm

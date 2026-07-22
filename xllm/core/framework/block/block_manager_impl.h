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

#include "block_manager.h"

namespace xllm {

class BlockManagerImpl : public BlockManager {
 public:
  explicit BlockManagerImpl(const Options& options);
  virtual ~BlockManagerImpl() {
    prefix_cache_.reset();
    CHECK_EQ(num_free_blocks_, free_blocks_.size() - 1)
        << "Not all blocks have been freed";
  };

  // Try to allocate blocks with num_blocks,
  // return {} if not enough blocks
  std::vector<Block> allocate(size_t num_blocks) override;

  void deallocate(const Slice<Block>& blocks) override;

  // Flat incremental growth: allocate ceil(num_tokens/block_size) - held blocks
  // and return them (does not insert into the sequence). The shared default for
  // flat-KV / compressed / xtensor leaves; SlidingWindow and Single override.
  std::optional<std::vector<Block>> allocate_for_sequence(
      Sequence* seq,
      size_t num_tokens) override;

  // allocate shared blocks when enable prefix cache
  std::vector<Block> allocate_shared(
      const Slice<int32_t>& token_ids,
      const Slice<Block>& existed_shared_blocks = {},
      const MMData& mm_data = MMData(),
      const Slice<XXH3Key>& block_hashes = {}) override;

  // cache blocks when enable prefix cache
  void cache(const Slice<int32_t>& token_ids,
             std::vector<Block>& blocks,
             size_t existed_shared_blocks_num = 0,
             const MMData& mm_data = MMData(),
             const Slice<XXH3Key>& block_hashes = {}) override;
  void cache(const std::vector<Block>& blocks) override;

  size_t num_blocks_in_prefix_cache() const override {
    if (options_.enable_prefix_cache()) {
      CHECK(prefix_cache_);
      return prefix_cache_->num_blocks();
    }
    return 0;
  }

  void reset_prefix_cache() override {
    if (options_.enable_prefix_cache() && prefix_cache_) {
      prefix_cache_->evict(prefix_cache_->num_blocks());
    }
  }

  // free blocks num
  size_t num_free_blocks() const override { return num_free_blocks_; }

  // used blocks num
  size_t num_used_blocks() const override { return num_used_blocks_; }

  // current kv cache utilization.
  double kv_cache_utilization() const override {
    return static_cast<double>(num_used_blocks_) / num_total_blocks();
  }

  // call BlockManager to free block used by Block.
  void free(int32_t block_id) override;

  // allocate a list of blocks
  // std::vector<Block> allocate(uint32_t n_blocks) override;

  // allocate a block
  Block allocate() override;

  // total blocks num
  size_t num_total_blocks() const override { return free_blocks_.size() - 1; }

 protected:
  // Flip a block's entry in `usage_ids` from 0 to 1. Returns true if the flip
  // happened; false if the entry was already 1 (i.e. block was already marked
  // used). Shared with subclasses (e.g. SlidingWindowBlockManager) that need
  // to reproduce the base allocate_shared refcount bookkeeping over their own
  // custom probe path. Static-friendly signature keeps callers free of `this`.
  static bool mark_used(std::vector<uint8_t>* usage_ids, int32_t block_id);

 private:
  // check if has enough slots, if not, try to evict some blocks
  // from the prefix cache
  bool has_enough_blocks(uint32_t num_blocks);

 protected:
  // prefix cache
  std::unique_ptr<PrefixCache> prefix_cache_;

  // reserved block id for padding
  Block padding_block_;

  // number of used blocks
  std::atomic<size_t> num_used_blocks_{0};

  // free block count
  std::atomic<size_t> num_free_blocks_{0};

  // block size
  size_t block_size_ = 0;

  // free block list
  std::vector<int32_t> free_blocks_;

  // Whether a block is already counted in num_used_blocks_.
  std::vector<uint8_t> usage_accounted_ids_;
};

}  // namespace xllm

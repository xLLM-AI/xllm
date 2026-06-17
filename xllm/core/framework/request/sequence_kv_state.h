/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <cstddef>
#include <optional>
#include <vector>

#include "core/common/types.h"
#include "core/util/slice.h"
#include "framework/block/block.h"

namespace xllm {

class KVCacheState {
 public:
  // get the number of tokens in the kvcache
  size_t kv_cache_tokens_num() const;
  void set_kv_cache_tokens_num(size_t num);
  void incr_kv_cache_tokens_num(size_t num);
  // get the number of shared blocks.
  size_t shared_kv_blocks_num() const;
  size_t shared_kv_tokens_num() const;

  void add_kv_blocks(const std::vector<Block>& new_blocks);
  void add_shared_kv_blocks(std::vector<Block>&& blocks,
                            size_t current_total_num_tokens);
  void incr_shared_kv_blocks_num(size_t num);
  void set_slice_window_size(uint32_t size);
  void update_slice_window_pos();

  size_t current_max_tokens_capacity() const;

  // returns allocated cache blocks
  Slice<Block> kv_blocks() const;
  std::vector<Block>* mutable_kv_blocks();

  Slice<Block> src_blocks() const { return src_blocks_; };

  void set_src_blocks(const std::vector<Block>& src_blocks,
                      bool need_swap = false) {
    src_blocks_ = std::move(src_blocks);
    need_swap_ = need_swap;
  };

  bool need_swap() const { return need_swap_; }

  // get the number of blocks
  size_t num_kv_blocks() const;
  std::vector<int32_t> kv_cache_slots(int32_t pos_start, int32_t pos_end);

  // composite block managers: blocks per sub-manager index (for
  // CompositeBlockManager)
  const std::vector<std::vector<Block>>& composite_blocks() const {
    return composite_blocks_;
  }
  std::vector<std::vector<Block>>* mutable_composite_blocks() {
    return &composite_blocks_;
  }

  void set_transfer_kv_info(TransferKVInfo&& info);
  std::optional<TransferKVInfo>& transfer_kv_info();

  uint32_t pushed_local_block_count() const {
    return pushed_local_block_count_;
  }
  void set_pushed_local_block_count(uint32_t n) {
    pushed_local_block_count_ = n;
  }

  size_t next_transfer_block_idx() const;
  void set_next_transfer_block_idx(size_t idx);
  void advance_transfer_block_idx(size_t idx);

  void reset();

  void process_beam_search(std::optional<Block> new_block = std::nullopt);

 private:
  // number of tokens in kv cache
  size_t kv_cache_tokens_num_ = 0;

  // kv cache blocks.
  std::vector<Block> blocks_;

  // source kv cache blocks for swap
  std::vector<Block> src_blocks_;

  // if need to swap last block
  bool need_swap_ = false;

  // transfer kv info for disaggregated PD mode.
  std::optional<TransferKVInfo> transfer_kv_info_;

  // next logical prompt block index that needs PD PUSH transfer.
  size_t next_transfer_block_idx_ = 0;

  // shared blocks number of the sequence.
  uint32_t num_owned_shared_blocks_ = 0;

  // blocks allocated per composite sub-manager index (used when BlockManager is
  // CompositeBlockManager). DeepSeek V4 leans on this so each per-manager
  // block table can be assembled independently of the legacy `blocks_` slot.
  std::vector<std::vector<Block>> composite_blocks_;

  // Sliding-window bookkeeping (DSA SWA: keeps the physical block order
  // stable so DSA metadata's modulo-based logical->physical mapping holds).
  uint32_t slice_window_pos_ = 0;
  uint32_t slice_window_size_ = 0;
  uint32_t slice_window_buffer_ = 0;

  // Number of local KV blocks already pushed to the decode instance.
  // Used for incremental push in chunked prefill + PD disagg mode.
  uint32_t pushed_local_block_count_ = 0;
};

}  // namespace xllm

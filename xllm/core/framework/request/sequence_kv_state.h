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

#include <cstddef>
#include <map>
#include <optional>
#include <utility>
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

  // Blocks held under `type`; empty slice when the type is absent.
  Slice<Block> blocks(BlockType type) const;
  // Mutable block list for `type`, created on demand.
  std::vector<Block>* mutable_blocks(BlockType type);
  // Number of blocks held under `type`.
  size_t num_blocks(BlockType type) const;
  // True if the sequence holds any cache-bearing blocks (KV / SWA / C4 / C128).
  // Excludes SINGLE, which is a per-sequence resource, not token cache. Used to
  // decide whether an allocation that started from an empty sequence should be
  // fully rolled back on failure (vs. a grow on an already-populated sequence).
  bool has_any_blocks() const;
  // token <-> physical slot mapping for `type` (paged attention). CHECKs the
  // type is present.
  std::vector<int32_t> cache_slots(BlockType type,
                                   int32_t pos_start,
                                   int32_t pos_end);

  void add_blocks(BlockType type, const std::vector<Block>& new_blocks);
  void add_shared_blocks(BlockType type,
                         std::vector<Block>&& blocks,
                         size_t current_total_num_tokens);
  void incr_shared_blocks_num(BlockType type, size_t num);
  // Drop all blocks held under `type` (releases their Block refs and removes
  // the map entry).
  void erase_blocks(BlockType type);
  // Number of shared (prefix-cache-hit) blocks held under `type`.
  size_t shared_blocks_num(BlockType type) const;
  // Number of shared tokens for this sequence. Sequence-level: the value is the
  // same across block types, so it takes no BlockType.
  size_t shared_tokens_num() const;

  void set_slice_window_size(uint32_t size);
  void update_slice_window_pos();

  size_t current_max_tokens_capacity() const;

  Slice<Block> src_blocks() const { return src_blocks_; };

  void set_src_blocks(const std::vector<Block>& src_blocks,
                      bool need_swap = false) {
    src_blocks_ = std::move(src_blocks);
    need_swap_ = need_swap;
  };

  bool need_swap() const { return need_swap_; }

  // Groups exported to the worker's multi_block_tables, in the fixed
  // kMultiBlockExportOrder. Each pair is (type, &blocks); only types currently
  // present in the map are returned. Empty for the flat KV path.
  std::vector<std::pair<BlockType, const std::vector<Block>*>>
  multi_block_export_view() const;
  // True when this sequence holds any multi_block_tables-exported group
  // (SWA / C4 / C128).
  bool has_multi_block_export() const;

  // Single per-sequence resource block (BlockType::SINGLE). The only per-type
  // id convenience accessor: returns the Single block id, or -1 when absent.
  // Used for linear_state_ids / embedding_ids export and disagg-PD's
  // linear_state_id. Other block types read via blocks(type).
  int32_t get_single_block_id() const;

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

  // KV cache blocks keyed by cache role. The flat attention KV lives under
  // BlockType::KV; DSV4 keeps its SWA / C4 / C128 groups here; the per-sequence
  // linear/embedding resource block lives under BlockType::SINGLE. std::map
  // keeps deterministic iteration for reset / dealloc / debugging, but worker
  // export order is governed by kMultiBlockExportOrder, not by map order.
  std::map<BlockType, std::vector<Block>> composite_blocks_;

  // source kv cache blocks for swap
  std::vector<Block> src_blocks_;

  // if need to swap last block
  bool need_swap_ = false;

  // transfer kv info for disaggregated PD mode.
  std::optional<TransferKVInfo> transfer_kv_info_;

  // next logical prompt block index that needs PD PUSH transfer.
  size_t next_transfer_block_idx_ = 0;

  // shared blocks number per block type.
  std::map<BlockType, uint32_t> num_owned_shared_blocks_;

  // Sliding-window cursor for legacy callers. CompositeBlockManager keeps DSA
  // SWA block vectors in absolute logical block order and leaves expired
  // logical positions invalid.
  uint32_t slice_window_pos_ = 0;
  uint32_t slice_window_size_ = 0;
  uint32_t slice_window_buffer_ = 0;

  // Number of local KV blocks already pushed to the decode instance.
  // Used for incremental push in chunked prefill + PD disagg mode.
  uint32_t pushed_local_block_count_ = 0;
};

}  // namespace xllm

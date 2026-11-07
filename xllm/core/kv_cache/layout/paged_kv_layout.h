/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "core/kv_cache/layout/seq_lens_layout.h"
#include "core/util/slice.h"
#include "core/util/utils.h"

namespace xllm {

// The allocator reserves block 0 as the padding target, so a block table entry
// that must resolve to a real address without touching live KV uses it.
inline constexpr int32_t kPaddingBlockId = 0;

// Slots are physical cache addresses. A negative slot is the kernel-side skip
// sentinel: the KV write is dropped instead of landing in block 0.
inline constexpr int32_t kSkippedSlotId = -1;

// A zero-length sequence still owns one page with last-page length one.
inline int32_t kv_page_count(int32_t kv_seq_len, int32_t block_size) {
  return util::ceil_div(std::max(kv_seq_len, 1), block_size);
}

inline int32_t kv_last_page_len(int32_t kv_seq_len, int32_t block_size) {
  return (std::max(kv_seq_len, 1) - 1) % block_size + 1;
}

inline int32_t append_paged_kv_geometry(
    int32_t kv_seq_len,
    int64_t max_blocks_per_seq,
    int32_t block_size,
    std::vector<int32_t>& paged_kv_indptr,
    std::vector<int32_t>& paged_kv_last_page_len) {
  CHECK_GT(block_size, 0);
  const int32_t page_count = kv_page_count(kv_seq_len, block_size);
  CHECK_LE(page_count, max_blocks_per_seq)
      << "KV length exceeds block-table capacity, kv_seq_len=" << kv_seq_len
      << ", page_count=" << page_count
      << ", max_blocks_per_seq=" << max_blocks_per_seq;
  CuSeqLens::append_cumulative(paged_kv_indptr, page_count);
  paged_kv_last_page_len.emplace_back(kv_last_page_len(kv_seq_len, block_size));
  return page_count;
}

// The indptr always carries a leading zero on every backend, unlike
// seq-lens vectors.
inline void append_paged_kv_metadata(
    int32_t kv_seq_len,
    const Slice<int32_t>& block_table,
    int32_t block_size,
    std::vector<int32_t>& paged_kv_indices,
    std::vector<int32_t>& paged_kv_indptr,
    std::vector<int32_t>& paged_kv_last_page_len) {
  const int32_t page_count =
      append_paged_kv_geometry(kv_seq_len,
                               static_cast<int64_t>(block_table.size()),
                               block_size,
                               paged_kv_indptr,
                               paged_kv_last_page_len);
  paged_kv_indices.insert(paged_kv_indices.end(),
                          block_table.begin(),
                          block_table.begin() + page_count);
}

// The ring flavor wraps the block index modulo the table width; out-of-range
// or negative block ids yield the -1 padding slot.
template <bool kRing, typename BlockIds>
inline int32_t slot_for_position(const BlockIds& block_ids,
                                 int64_t pos,
                                 int64_t block_size,
                                 int64_t block_table_width) {
  if (block_table_width <= 0) {
    return kSkippedSlotId;
  }
  int64_t block_idx = pos / block_size;
  if (kRing) {
    block_idx %= block_table_width;
  } else if (block_idx >= block_table_width) {
    return kSkippedSlotId;
  }
  const int32_t block_id = block_ids[block_idx];
  if (block_id < 0) {
    return kSkippedSlotId;
  }
  return static_cast<int32_t>(static_cast<int64_t>(block_id) * block_size +
                              pos % block_size);
}

template <bool kRing>
inline int32_t compute_slot_mapping(const Slice<int32_t>& block_table,
                                    int64_t pos,
                                    int64_t block_size) {
  CHECK_GT(block_size, 0);
  const int32_t slot = slot_for_position<kRing>(
      block_table, pos, block_size, static_cast<int64_t>(block_table.size()));
  CHECK_NE(slot, kSkippedSlotId)
      << "block table index out of range or invalid block id,"
      << " position=" << pos << ", block_size=" << block_size
      << ", block_table_size=" << block_table.size();
  return slot;
}

}  // namespace xllm

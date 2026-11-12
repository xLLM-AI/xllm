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
#include <torch/torch.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <vector>

#include "core/common/constants.h"
#include "core/util/slice.h"
#include "core/util/tensor_helper.h"
#include "core/util/utils.h"

namespace xllm::util {

inline std::vector<torch::Tensor> gather_block_tables(
    std::vector<std::vector<std::vector<int32_t>>>& tables) {
  std::vector<torch::Tensor> out;
  out.reserve(tables.size());
  for (auto& manager_tables : tables) {
    pad_2d_vector(manager_tables, -1);
    out.emplace_back(create_2d_tensor(manager_tables, torch::kInt));
  }
  return out;
}

inline int32_t max_len(const Slice<int32_t>& lens) {
  return lens.empty() ? 0 : *std::max_element(lens.begin(), lens.end());
}

template <typename T = int32_t>
void fill_query_start_loc(std::vector<T>& dst, const Slice<int32_t>& lens) {
  dst.assign(lens.size() + 1, /*value=*/T{0});
  std::partial_sum(lens.begin(), lens.end(), dst.begin() + 1, std::plus<T>());
}

inline std::vector<int32_t> query_start_loc(const Slice<int32_t>& lens) {
  std::vector<int32_t> cumulative_lens;
  fill_query_start_loc(cumulative_lens, lens);
  return cumulative_lens;
}

template <typename T = int32_t>
std::vector<T> cu_num_tokens(const Slice<int32_t>& lens) {
  std::vector<T> offsets(lens.size());
  std::partial_sum(lens.begin(), lens.end(), offsets.begin(), std::plus<T>());
  return offsets;
}

inline void append_indptr(std::vector<int32_t>& cumulative_lens,
                          int32_t length) {
  if (cumulative_lens.empty()) {
    cumulative_lens.emplace_back(/*value=*/0);
  }
  cumulative_lens.emplace_back(cumulative_lens.back() + length);
}

// Padded rows keep a decode-width query but own no KV page: a non-zero KV
// length would read the zero-filled dummy block-table rows.
inline void pad_uniform_decode_lens(std::vector<int32_t>& q_seq_lens,
                                    std::vector<int32_t>& kv_seq_lens,
                                    size_t num_tokens_padded,
                                    int32_t uniform_decode_query_len) {
  if (q_seq_lens.empty() && kv_seq_lens.empty()) {
    return;
  }
  q_seq_lens.resize(q_seq_lens.size() + num_tokens_padded,
                    uniform_decode_query_len);
  kv_seq_lens.resize(kv_seq_lens.size() + num_tokens_padded, /*value=*/0);
}

inline void fill_query_start_loc(const torch::Tensor& seq_lens,
                                 int64_t num_requests,
                                 torch::Tensor& dst) {
  torch::Tensor cumsum_dst =
      dst.slice(/*dim=*/0, /*start=*/1, /*end=*/num_requests + 1);
  torch::cumsum_out(
      cumsum_dst,
      seq_lens.slice(/*dim=*/0, /*start=*/0, /*end=*/num_requests),
      /*dim=*/0,
      /*dtype=*/torch::kInt32);
}

inline torch::Tensor make_query_start_loc(const torch::Tensor& seq_lens,
                                          int64_t num_requests = -1) {
  if (!seq_lens.defined()) {
    return {};
  }
  num_requests = num_requests < 0 ? seq_lens.numel() : num_requests;
  torch::Tensor cumulative =
      torch::empty({num_requests + 1}, seq_lens.options().dtype(torch::kInt32));
  cumulative[0].fill_(/*value=*/0);
  fill_query_start_loc(seq_lens, num_requests, cumulative);
  return cumulative;
}

inline void pad_query_start_loc(torch::Tensor& cumulative,
                                int64_t filled_size,
                                int64_t total_size) {
  if (total_size <= filled_size) {
    return;
  }
  cumulative.slice(/*dim=*/0, /*start=*/filled_size, /*end=*/total_size)
      .copy_(cumulative
                 .slice(/*dim=*/0,
                        /*start=*/filled_size - 1,
                        /*end=*/filled_size)
                 .expand({total_size - filled_size}),
             /*non_blocking=*/true);
}

inline int32_t kv_page_count(int32_t kv_seq_len, int32_t block_size) {
  return ceil_div(std::max(kv_seq_len, /*b=*/1), block_size);
}

inline int32_t append_paged_kv_indptr(int32_t kv_seq_len,
                                      int64_t max_blocks_per_seq,
                                      int32_t block_size,
                                      std::vector<int32_t>& paged_kv_indptr,
                                      std::vector<int32_t>& last_page_len_out) {
  const int32_t page_count = kv_page_count(kv_seq_len, block_size);
  CHECK_LE(page_count, max_blocks_per_seq)
      << "KV length exceeds block-table capacity, kv_seq_len=" << kv_seq_len
      << ", page_count=" << page_count
      << ", max_blocks_per_seq=" << max_blocks_per_seq;
  append_indptr(paged_kv_indptr, page_count);
  last_page_len_out.emplace_back(
      (std::max(kv_seq_len, /*b=*/1) - 1) % block_size + 1);
  return page_count;
}

template <typename BlockIds>
inline void append_paged_kv_entry(const BlockIds& block_ids,
                                  int32_t kv_seq_len,
                                  int32_t block_size,
                                  std::vector<int32_t>& paged_kv_indptr,
                                  std::vector<int32_t>& paged_kv_indices,
                                  std::vector<int32_t>& last_page_len_out) {
  const int32_t page_count =
      append_paged_kv_indptr(kv_seq_len,
                             static_cast<int64_t>(block_ids.size()),
                             block_size,
                             paged_kv_indptr,
                             last_page_len_out);
  paged_kv_indices.insert(paged_kv_indices.end(),
                          block_ids.begin(),
                          block_ids.begin() + page_count);
}

template <bool Ring, typename BlockIds>
inline int32_t slot_for_position(const BlockIds& block_ids,
                                 int64_t pos,
                                 int64_t block_size,
                                 int64_t block_table_width) {
  if (block_table_width <= 0) {
    return kPadSlotId;
  }
  int64_t block_idx = pos / block_size;
  if (Ring) {
    block_idx %= block_table_width;
  } else if (block_idx >= block_table_width) {
    return kPadSlotId;
  }
  const int32_t block_id = block_ids[block_idx];
  if (block_id < 0) {
    return kPadSlotId;
  }
  return static_cast<int32_t>(static_cast<int64_t>(block_id) * block_size +
                              pos % block_size);
}

template <bool Ring>
inline int32_t compute_slot_mapping(const Slice<int32_t>& block_table,
                                    int64_t pos,
                                    int64_t block_size) {
  CHECK_GT(block_size, 0);
  const int32_t slot = slot_for_position<Ring>(
      block_table, pos, block_size, static_cast<int64_t>(block_table.size()));
  CHECK_NE(slot, kPadSlotId)
      << "block table index out of range or invalid block id,"
      << " position=" << pos << ", block_size=" << block_size
      << ", block_table_size=" << block_table.size();
  return slot;
}

}  // namespace xllm::util

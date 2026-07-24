/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "framework/parallel_state/npu_cp_plan.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

#include "core/framework/model/model_input_params.h"
#include "core/framework/parallel_state/parallel_state.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/util/tensor_helper.h"

namespace xllm {
namespace {

torch::Tensor generate_query_balance_indices(
    const torch::Tensor& input_lengths) {
  CHECK_EQ(input_lengths.scalar_type(), torch::kInt32)
      << "input_lengths must be int32 tensor";
  CHECK_EQ(input_lengths.dim(), 1) << "input_lengths must be 1D tensor";

  std::vector<int32_t> lengths_vec;
  int32_t* lengths_ptr = input_lengths.data_ptr<int32_t>();
  const int64_t n = input_lengths.numel();
  lengths_vec.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    lengths_vec.push_back(lengths_ptr[i]);
  }

  std::vector<int32_t> query_balance_indices_first;
  std::vector<int32_t> query_balance_indices_last;
  int32_t base = 0;
  for (int32_t length : lengths_vec) {
    std::vector<int32_t> length_range(length);
    std::iota(length_range.begin(), length_range.end(), base);
    int32_t divider = length / 2;
    query_balance_indices_first.insert(query_balance_indices_first.end(),
                                       length_range.begin(),
                                       length_range.begin() + divider);
    query_balance_indices_last.insert(query_balance_indices_last.end(),
                                      length_range.begin() + divider,
                                      length_range.end());
    base += length;
  }

  query_balance_indices_first.insert(query_balance_indices_first.end(),
                                     query_balance_indices_last.begin(),
                                     query_balance_indices_last.end());

  auto tensor = torch::tensor(query_balance_indices_first,
                              torch::dtype(torch::kInt32).device(torch::kCPU));
  return tensor;
}

torch::Tensor generate_attention_output_reorder_indices(
    const std::vector<int32_t>& chunk_lengths) {
  std::vector<int32_t> attention_output_reorder_indices;
  int32_t base = 0;
  int32_t chunk_lengths_sum =
      std::accumulate(chunk_lengths.begin(), chunk_lengths.end(), 0);

  for (int32_t chunk_len : chunk_lengths) {
    std::vector<int32_t> length_range(chunk_len);
    std::iota(length_range.begin(), length_range.end(), base);
    attention_output_reorder_indices.insert(
        attention_output_reorder_indices.end(),
        length_range.begin(),
        length_range.end());
    std::vector<int32_t> last_part(length_range.size());
    std::transform(
        length_range.begin(),
        length_range.end(),
        last_part.begin(),
        [chunk_lengths_sum](int32_t x) { return x + chunk_lengths_sum; });
    attention_output_reorder_indices.insert(
        attention_output_reorder_indices.end(),
        last_part.begin(),
        last_part.end());
    base += chunk_len;
  }

  return torch::tensor(attention_output_reorder_indices,
                       torch::dtype(torch::kInt32).device(torch::kCPU));
}

torch::Tensor generate_kv_reorder_indices(
    int32_t cp_size,
    int32_t input_ids_size,
    const std::vector<int32_t>& chunk_lengths) {
  std::vector<int32_t> kv_reorder_indices;
  int32_t req_offset = 0;

  for (int32_t req_chunk_len : chunk_lengths) {
    std::vector<std::vector<int32_t>> gather_idx_per_chunk(cp_size * 2);
    for (int32_t cp_rank_id = 0; cp_rank_id < cp_size; ++cp_rank_id) {
      int32_t rank_offset = cp_rank_id * input_ids_size;
      std::vector<int32_t> first_part(req_chunk_len);
      std::iota(first_part.begin(), first_part.end(), rank_offset + req_offset);
      gather_idx_per_chunk[cp_rank_id] = first_part;

      std::vector<int32_t> last_part(req_chunk_len);
      std::iota(last_part.begin(),
                last_part.end(),
                rank_offset + req_offset + req_chunk_len);
      gather_idx_per_chunk[cp_size * 2 - 1 - cp_rank_id] = last_part;
    }

    for (const auto& vec : gather_idx_per_chunk) {
      kv_reorder_indices.insert(
          kv_reorder_indices.end(), vec.begin(), vec.end());
    }
    req_offset += req_chunk_len * 2;
  }

  return torch::tensor(kv_reorder_indices,
                       torch::dtype(torch::kInt32).device(torch::kCPU));
}

std::pair<torch::Tensor, torch::Tensor> compute_input_lengths_cumsum_cp(
    const torch::Tensor& input_lengths_cumsum) {
  CHECK_EQ(input_lengths_cumsum.scalar_type(), torch::kInt32)
      << "input_lengths_cumsum must be int32 tensor";
  CHECK_EQ(input_lengths_cumsum.dim(), 1)
      << "input_lengths_cumsum must be 1D tensor";

  const int64_t n = input_lengths_cumsum.numel();
  auto input_lengths_cumsum_cp_prev =
      torch::zeros({n}, torch::dtype(torch::kInt32).device(torch::kCPU));
  auto input_lengths_cumsum_cp_next =
      torch::zeros({n}, torch::dtype(torch::kInt32).device(torch::kCPU));

  int32_t offset = 0;
  auto cumsum_data = input_lengths_cumsum.data_ptr<int32_t>();
  auto prev_data = input_lengths_cumsum_cp_prev.data_ptr<int32_t>();
  auto next_data = input_lengths_cumsum_cp_next.data_ptr<int32_t>();

  for (int64_t i = 0; i < n; ++i) {
    prev_data[i] = offset + (cumsum_data[i] - offset) / 2;
    next_data[i] = cumsum_data[i];
    offset = cumsum_data[i];
  }

  return {input_lengths_cumsum_cp_prev, input_lengths_cumsum_cp_next};
}
/* |ctx 0|ctx 1|ctx 2|new 0|new 1|new 2| */
std::pair<torch::Tensor, torch::Tensor> generate_k_gather_index(
    const torch::Tensor& actual_seq_lengths_kv_cp_prev,
    const torch::Tensor& actual_seq_lengths_kv_cp_next,
    const torch::Tensor& input_lengths,
    int32_t cp_size) {
  CHECK_EQ(actual_seq_lengths_kv_cp_prev.dim(), 1)
      << "actual_seq_lengths_kv_cp_prev must be 1D";
  CHECK_EQ(actual_seq_lengths_kv_cp_next.dim(), 1)
      << "actual_seq_lengths_kv_cp_next must be 1D";
  CHECK_EQ(input_lengths.dim(), 1) << "input_lengths must be 1D";

  std::vector<int32_t> prev_kv_gather_indices;
  std::vector<int32_t> next_kv_gather_indices;
  int32_t k_offset = 0;
  const int64_t n = input_lengths.numel();

  auto prev_len_data = actual_seq_lengths_kv_cp_prev.data_ptr<int32_t>();
  auto next_len_data = actual_seq_lengths_kv_cp_next.data_ptr<int32_t>();
  auto input_len_data = input_lengths.data_ptr<int32_t>();

  for (int64_t i = 0; i < n; ++i) {
    std::vector<int32_t> prev_range(prev_len_data[i]);
    std::iota(prev_range.begin(), prev_range.end(), k_offset);
    prev_kv_gather_indices.insert(
        prev_kv_gather_indices.end(), prev_range.begin(), prev_range.end());

    std::vector<int32_t> next_range(next_len_data[i]);
    std::iota(next_range.begin(), next_range.end(), k_offset);
    next_kv_gather_indices.insert(
        next_kv_gather_indices.end(), next_range.begin(), next_range.end());

    k_offset += input_len_data[i] * cp_size;
  }

  auto prev_tensor = torch::tensor(
      prev_kv_gather_indices, torch::dtype(torch::kInt32).device(torch::kCPU));
  auto next_tensor = torch::tensor(
      next_kv_gather_indices, torch::dtype(torch::kInt32).device(torch::kCPU));
  return {prev_tensor, next_tensor};
}

// Build the per-rank prefix geometry shared by prefix slot preparation and
// attention gather-index construction:
//   - real_len[i] = (kv_cache_tokens_per_seq[i] / cp_size / block_size) *
//   block_size
//   - cache_len[i] = real_len[i] if real_len[i] > 0 else 1   (1-slot padding)
//   - offset_in_rank[i] = sum_{k<i} cache_len[k]
//   - rank_block_size  = sum_i cache_len[i]
// Real vs cache len is the source of truth for whether a seq's ctx tokens
// actually exist in `prefix_kv_allgather` (real_len == 0 means only the
// padding slot is present and must not appear in the gather result).
struct PrefixRankGeometry {
  std::vector<int32_t> real_len_in_rank;
  std::vector<int32_t> cache_len_in_rank;
  std::vector<int32_t> offset_in_rank;
  int64_t rank_block_size = 0;
};

PrefixRankGeometry compute_prefix_rank_geometry(
    const std::vector<int32_t>& kv_cache_tokens_per_seq,
    int32_t cp_size,
    int32_t block_size) {
  CHECK_GT(cp_size, 0) << "cp_size must be positive";
  CHECK_GT(block_size, 0) << "block_size must be positive";

  const int64_t n = static_cast<int64_t>(kv_cache_tokens_per_seq.size());
  PrefixRankGeometry geom;
  geom.real_len_in_rank.resize(n);
  geom.cache_len_in_rank.resize(n);
  geom.offset_in_rank.resize(n);
  for (int64_t i = 0; i < n; ++i) {
    const int32_t per_rank_prefix_tokens =
        (kv_cache_tokens_per_seq[i] / cp_size / block_size) * block_size;
    geom.real_len_in_rank[i] = per_rank_prefix_tokens;
    geom.cache_len_in_rank[i] =
        per_rank_prefix_tokens == 0 ? 1 : per_rank_prefix_tokens;
    geom.offset_in_rank[i] =
        (i == 0 ? 0
                : geom.offset_in_rank[i - 1] + geom.cache_len_in_rank[i - 1]);
    geom.rank_block_size += geom.cache_len_in_rank[i];
  }
  return geom;
}

torch::Tensor build_prefix_cache_slots(
    const torch::Tensor& block_tables,
    const std::vector<int32_t>& prefix_token_counts,
    int32_t block_size,
    int32_t kv_split_size) {
  CHECK(block_tables.defined());
  CHECK(block_tables.device().is_cpu());
  CHECK_EQ(block_tables.scalar_type(), torch::kInt32);
  CHECK_EQ(block_tables.dim(), 2) << "block_tables must be a 2D tensor";
  CHECK_EQ(static_cast<size_t>(block_tables.size(0)),
           prefix_token_counts.size())
      << "block_tables rows must match prefix_token_counts size";

  const PrefixRankGeometry geometry = compute_prefix_rank_geometry(
      prefix_token_counts, kv_split_size, block_size);
  if (block_tables.size(0) == 0) {
    return torch::tensor({0}, torch::kInt32);
  }

  std::vector<int32_t> prefix_cache_slots;
  prefix_cache_slots.reserve(geometry.rank_block_size);
  auto block_table = block_tables.accessor<int32_t, 2>();
  for (int64_t sequence = 0; sequence < block_tables.size(0); ++sequence) {
    const int32_t prefix_block_count =
        geometry.real_len_in_rank[sequence] / block_size;
    if (prefix_block_count == 0) {
      prefix_cache_slots.emplace_back(0);
      continue;
    }
    CHECK_LE(prefix_block_count, block_tables.size(1));
    for (int32_t block = 0; block < prefix_block_count; ++block) {
      const int32_t first_slot = block_table[sequence][block] * block_size;
      for (int32_t offset = 0; offset < block_size; ++offset) {
        prefix_cache_slots.emplace_back(first_slot + offset);
      }
    }
  }
  return torch::tensor(prefix_cache_slots, torch::kInt32);
}

// Gather index over `intermediate_kv` (current segment of `merged_kv`, which
// is per-seq grouped after `kv_reorder_indices` reorder; each seq occupies
// `cp_size * input_lengths[i]` slots). Offsets are local to the current
// segment (start at 0). `merge_context_and_current_k_gather_index` rebases
// them onto `merged_kv` by adding the prefix segment total length.
//
// current_lengths_kv_cp_prev[i] = max(0, actual_seq_lengths_kv_cp_prev[i]
//                                       - per_rank_prefix_tokens[i] * cp_size)
// current_lengths_kv_cp_next[i] = max(0, actual_seq_lengths_kv_cp_next[i]
//                                       - per_rank_prefix_tokens[i] * cp_size)
std::pair<torch::Tensor, torch::Tensor> generate_current_k_gather_index(
    const torch::Tensor& current_lengths_kv_cp_prev,
    const torch::Tensor& current_lengths_kv_cp_next,
    const torch::Tensor& input_lengths,
    int32_t cp_size) {
  CHECK_EQ(current_lengths_kv_cp_prev.dim(), 1)
      << "current_lengths_kv_cp_prev must be 1D";
  CHECK_EQ(current_lengths_kv_cp_next.dim(), 1)
      << "current_lengths_kv_cp_next must be 1D";
  CHECK_EQ(input_lengths.dim(), 1) << "input_lengths must be 1D";
  CHECK_EQ(current_lengths_kv_cp_prev.numel(), input_lengths.numel())
      << "current_lengths_kv_cp_prev size mismatch";
  CHECK_EQ(current_lengths_kv_cp_next.numel(), input_lengths.numel())
      << "current_lengths_kv_cp_next size mismatch";
  CHECK_GT(cp_size, 0) << "cp_size must be positive";

  const int64_t n = input_lengths.numel();
  auto prev_len_data = current_lengths_kv_cp_prev.data_ptr<int32_t>();
  auto next_len_data = current_lengths_kv_cp_next.data_ptr<int32_t>();
  auto input_len_data = input_lengths.data_ptr<int32_t>();

  std::vector<int32_t> prev_idx;
  std::vector<int32_t> next_idx;
  int32_t k_offset = 0;
  for (int64_t i = 0; i < n; ++i) {
    std::vector<int32_t> prev_range(prev_len_data[i]);
    std::iota(prev_range.begin(), prev_range.end(), k_offset);
    prev_idx.insert(prev_idx.end(), prev_range.begin(), prev_range.end());

    std::vector<int32_t> next_range(next_len_data[i]);
    std::iota(next_range.begin(), next_range.end(), k_offset);
    next_idx.insert(next_idx.end(), next_range.begin(), next_range.end());

    k_offset += input_len_data[i] * cp_size;
  }

  return {
      torch::tensor(prev_idx, torch::dtype(torch::kInt32).device(torch::kCPU)),
      torch::tensor(next_idx, torch::dtype(torch::kInt32).device(torch::kCPU))};
}

// Gather index over `prefix_kv_allgather` (prefix segment of `merged_kv`,
// which is rank-grouped after AllGather: kv_split_size rank segments back-
// to-back, each segment = concat of per-rank prefix slices over all seqs,
// including 1-token padding slots for prefix-less seqs).
//
// For each seq with a real prefix, this emits its full prefix by stitching
// the same `cache_len_in_rank` slice from each of the kv_split_size rank
// segments. Prefix-less seqs are skipped entirely so their padding slots
// never appear in the gather output.
//
// prev and next halves both attend to the full prefix, so this function
// returns the same tensor (cloned) for the two output slots.
//
// Note: the parameter was historically called `cp_size`. After the KV-split
// / CP decoupling refactor it should be passed `kv_split_size` since the
// prefix geometry is shard-aligned, not token-CP-aligned. When the two
// happen to be equal (the legacy default) behavior is unchanged.
std::pair<torch::Tensor, torch::Tensor> generate_context_k_gather_index(
    const std::vector<int32_t>& kv_cache_tokens_per_seq,
    int32_t kv_split_size,
    int32_t block_size) {
  const auto geom = compute_prefix_rank_geometry(
      kv_cache_tokens_per_seq, kv_split_size, block_size);
  const int64_t n = static_cast<int64_t>(kv_cache_tokens_per_seq.size());

  std::vector<int32_t> ctx_idx;
  for (int64_t i = 0; i < n; ++i) {
    if (geom.real_len_in_rank[i] == 0) {
      // Prefix-less seq: only the padding slot lives in merged_kv's prefix
      // segment and it must not be gathered.
      continue;
    }
    for (int64_t j = 0; j < kv_split_size; ++j) {
      std::vector<int32_t> prefix_range(geom.cache_len_in_rank[i]);
      std::iota(prefix_range.begin(),
                prefix_range.end(),
                geom.offset_in_rank[i] + geom.rank_block_size * j);
      ctx_idx.insert(ctx_idx.end(), prefix_range.begin(), prefix_range.end());
    }
  }

  auto tensor =
      torch::tensor(ctx_idx, torch::dtype(torch::kInt32).device(torch::kCPU));
  return {tensor, tensor.clone()};
}

// Stitch the per-seq context (prefix) and current slices into the final
// gather indices over `merged_kv`. The final layout is per-seq interleaved:
//   prev: |ctx_0|cur_0_prev|ctx_1|cur_1_prev|...|
//   next: |ctx_0|cur_0_next|ctx_1|cur_1_next|...|
// where ctx_i is empty for prefix-less seqs.
//
// Context indices are taken verbatim from history_* (already absolute
// offsets into merged_kv's prefix segment). Current indices are local to
// the current segment so we rebase them by adding the prefix total length
// (= rank_block_size * cp_size).
std::pair<torch::Tensor, torch::Tensor>
merge_context_and_current_k_gather_index(
    const torch::Tensor& current_prev_kv_gather_indices,
    const torch::Tensor& current_next_kv_gather_indices,
    const torch::Tensor& history_prev_kv_gather_indices,
    const torch::Tensor& history_next_kv_gather_indices,
    const torch::Tensor& current_lengths_kv_cp_prev,
    const torch::Tensor& current_lengths_kv_cp_next,
    const torch::Tensor& input_lengths,
    const std::vector<int32_t>& kv_cache_tokens_per_seq,
    int32_t kv_split_size,
    int32_t block_size) {
  // NOTE: `kv_split_size` (was `cp_size`) only governs the PREFIX-segment
  // geometry: rank stride for AllGather slices and `prefix_total_len`. The
  // CURRENT-segment indices passed in via current_k_gather_index_* are
  // generated upstream with token-CP cp_size, so the two widths can differ.
  CHECK_EQ(input_lengths.dim(), 1) << "input_lengths must be 1D";
  CHECK_EQ(current_lengths_kv_cp_prev.dim(), 1)
      << "current_lengths_kv_cp_prev must be 1D";
  CHECK_EQ(current_lengths_kv_cp_next.dim(), 1)
      << "current_lengths_kv_cp_next must be 1D";
  CHECK_EQ(static_cast<size_t>(input_lengths.numel()),
           kv_cache_tokens_per_seq.size())
      << "input_lengths must equal kv_cache_tokens_per_seq size";

  const auto geom = compute_prefix_rank_geometry(
      kv_cache_tokens_per_seq, kv_split_size, block_size);
  const int32_t prefix_total_len =
      static_cast<int32_t>(geom.rank_block_size * kv_split_size);
  const int64_t n = input_lengths.numel();

  auto current_prev_len_data = current_lengths_kv_cp_prev.data_ptr<int32_t>();
  auto current_next_len_data = current_lengths_kv_cp_next.data_ptr<int32_t>();
  auto current_prev_data = current_prev_kv_gather_indices.data_ptr<int32_t>();
  auto current_next_data = current_next_kv_gather_indices.data_ptr<int32_t>();
  auto history_prev_data = history_prev_kv_gather_indices.data_ptr<int32_t>();
  auto history_next_data = history_next_kv_gather_indices.data_ptr<int32_t>();

  std::vector<int32_t> merged_prev;
  std::vector<int32_t> merged_next;
  int64_t history_off = 0;
  int64_t current_off_prev = 0;
  int64_t current_off_next = 0;

  for (int64_t i = 0; i < n; ++i) {
    const int32_t ctx_len_i = geom.real_len_in_rank[i] > 0
                                  ? geom.real_len_in_rank[i] * kv_split_size
                                  : 0;
    for (int32_t k = 0; k < ctx_len_i; ++k) {
      merged_prev.push_back(history_prev_data[history_off + k]);
      merged_next.push_back(history_next_data[history_off + k]);
    }
    history_off += ctx_len_i;

    const int32_t cur_prev_len_i = current_prev_len_data[i];
    for (int32_t k = 0; k < cur_prev_len_i; ++k) {
      merged_prev.push_back(current_prev_data[current_off_prev + k] +
                            prefix_total_len);
    }
    current_off_prev += cur_prev_len_i;

    const int32_t cur_next_len_i = current_next_len_data[i];
    for (int32_t k = 0; k < cur_next_len_i; ++k) {
      merged_next.push_back(current_next_data[current_off_next + k] +
                            prefix_total_len);
    }
    current_off_next += cur_next_len_i;
  }

  return {torch::tensor(merged_prev,
                        torch::dtype(torch::kInt32).device(torch::kCPU)),
          torch::tensor(merged_next,
                        torch::dtype(torch::kInt32).device(torch::kCPU))};
}

CpAttentionMeta build_attention_tensor_meta(
    int32_t cp_size,
    int64_t local_token_num,
    const torch::Tensor& position_ids,
    const torch::Tensor& input_lengths,
    bool have_prefix_slots,
    const std::vector<int32_t>& kv_cache_tokens_per_seq,
    int32_t block_size,
    int32_t kv_split_size) {
  CHECK_GT(cp_size, 0) << "cp_size must be positive";
  // Default kv_split_size to cp_size to preserve legacy behavior (prefix
  // geometry was implicitly bound to cp_size before the KV-split / CP
  // decoupling refactor).
  if (kv_split_size <= 0) {
    kv_split_size = cp_size;
  }
  CHECK_GT(kv_split_size, 0) << "kv_split_size must resolve to positive";
  CHECK_EQ(cp_size % kv_split_size, 0)
      << "cp_size (" << cp_size << ") must be divisible by kv_split_size ("
      << kv_split_size << ").";
  CpAttentionMeta inputs;

  std::vector<int32_t> chunk_lengths;
  chunk_lengths.reserve(input_lengths.numel());
  auto input_len_data = input_lengths.data_ptr<int32_t>();
  for (int64_t i = 0; i < input_lengths.numel(); ++i) {
    chunk_lengths.push_back(input_len_data[i] / 2);
  }

  inputs.query_balance_indices = generate_query_balance_indices(input_lengths);

  inputs.attention_output_reorder_indices =
      generate_attention_output_reorder_indices(chunk_lengths);

  inputs.kv_reorder_indices =
      generate_kv_reorder_indices(cp_size, local_token_num, chunk_lengths);

  auto input_lengths_cumsum = torch::cumsum(input_lengths, 0, torch::kInt32);
  auto [input_lengths_cumsum_cp_prev, input_lengths_cumsum_cp_next] =
      compute_input_lengths_cumsum_cp(input_lengths_cumsum);

  auto gather_index_prev = (input_lengths_cumsum_cp_prev - 1).to(torch::kLong);
  auto gather_index_next = (input_lengths_cumsum_cp_next - 1).to(torch::kLong);
  // Guard zero-length seqs: input_lengths[i] == 0 -> cumsum_cp_prev/next == 0
  // -> gather_index == -1, out of range for index_select. Clamp to 0; the seq
  // contributes no prev/next KV and downstream actual_seq_lengths == 0 masks
  // the bogus position_ids[0] lookup away. (For local_padded-based inputs a
  // zero length only occurs when the seq has no real token, which the batch
  // builder rejects before reaching here; this is purely defensive.)
  gather_index_prev = torch::clamp(gather_index_prev, /*min=*/0);
  gather_index_next = torch::clamp(gather_index_next, /*min=*/0);
  auto position_ids_prev = position_ids.index_select(0, gather_index_prev) + 1;
  auto position_ids_next = position_ids.index_select(0, gather_index_next) + 1;
  auto actual_seq_lengths_kv_cp_prev = position_ids_prev.to(torch::kInt32);
  auto actual_seq_lengths_kv_cp_next = position_ids_next.to(torch::kInt32);

  if (have_prefix_slots) {
    // Strip the per-seq full-prefix length from the SFA-logical KV lengths to
    // obtain how much each seq's prev/next half needs from merged_kv's current
    // segment. Prefix-less seqs get prefix_kv_len_total == 0 and fall through
    // unchanged. Prefix slots and these gather indices share
    // `compute_prefix_rank_geometry` as their source of truth.
    const int64_t n = input_lengths.numel();
    CHECK_EQ(static_cast<size_t>(n), kv_cache_tokens_per_seq.size())
        << "input_lengths must equal kv_cache_tokens_per_seq size";
    // Prefix geometry is shard-aligned to `kv_split_size`, not cp_size:
    //   per_rank_prefix_tokens = total_prefix / kv_split_size
    //   prefix_kv_len_total    = per_rank_prefix * kv_split_size
    // When kv_split_size == cp_size (legacy) this is byte-identical to the
    // previous implementation.
    const auto geom = compute_prefix_rank_geometry(
        kv_cache_tokens_per_seq, kv_split_size, block_size);
    auto prev_total_data = actual_seq_lengths_kv_cp_prev.data_ptr<int32_t>();
    auto next_total_data = actual_seq_lengths_kv_cp_next.data_ptr<int32_t>();
    std::vector<int32_t> current_prev_vec(n);
    std::vector<int32_t> current_next_vec(n);
    for (int64_t i = 0; i < n; ++i) {
      const int32_t prefix_kv_len_total =
          geom.real_len_in_rank[i] * kv_split_size;
      current_prev_vec[i] =
          std::max(0, prev_total_data[i] - prefix_kv_len_total);
      current_next_vec[i] =
          std::max(0, next_total_data[i] - prefix_kv_len_total);
    }
    auto current_lengths_kv_cp_prev = torch::tensor(
        current_prev_vec, torch::dtype(torch::kInt32).device(torch::kCPU));
    auto current_lengths_kv_cp_next = torch::tensor(
        current_next_vec, torch::dtype(torch::kInt32).device(torch::kCPU));

    // The CURRENT segment (intermediate_kv) is still rearranged by token-CP,
    // so generate_current_k_gather_index uses cp_size (not kv_split_size) for
    // its per-seq stride k_offset += input_len * cp_size.
    auto current_pair =
        generate_current_k_gather_index(current_lengths_kv_cp_prev,
                                        current_lengths_kv_cp_next,
                                        input_lengths,
                                        cp_size);
    // The PREFIX segment (prefix_kv_allgather) is rank-grouped by KV-split,
    // so context/merge use kv_split_size for per-rank slice offsets and
    // `prefix_total_len = rank_block_size * kv_split_size`.
    auto history_pair = generate_context_k_gather_index(
        kv_cache_tokens_per_seq, kv_split_size, block_size);

    std::tie(inputs.prev_kv_gather_indices, inputs.next_kv_gather_indices) =
        merge_context_and_current_k_gather_index(current_pair.first,
                                                 current_pair.second,
                                                 history_pair.first,
                                                 history_pair.second,
                                                 current_lengths_kv_cp_prev,
                                                 current_lengths_kv_cp_next,
                                                 input_lengths,
                                                 kv_cache_tokens_per_seq,
                                                 kv_split_size,
                                                 block_size);
  } else {
    std::tie(inputs.prev_kv_gather_indices, inputs.next_kv_gather_indices) =
        generate_k_gather_index(actual_seq_lengths_kv_cp_prev,
                                actual_seq_lengths_kv_cp_next,
                                input_lengths,
                                cp_size);
  }

  auto actual_seq_lengths_kv_cp_prev_cumsum =
      torch::cumsum(actual_seq_lengths_kv_cp_prev, 0, torch::kInt32);
  auto actual_seq_lengths_kv_cp_next_cumsum =
      torch::cumsum(actual_seq_lengths_kv_cp_next, 0, torch::kInt32);
  inputs.prev_key_cu_seq_lens = actual_seq_lengths_kv_cp_prev_cumsum;
  inputs.next_key_cu_seq_lens = actual_seq_lengths_kv_cp_next_cumsum;

  auto input_lengths_cumsum_half = torch::floor_divide(input_lengths_cumsum, 2);
  inputs.prev_query_cu_seq_lens = input_lengths_cumsum_half;
  inputs.next_query_cu_seq_lens = input_lengths_cumsum_half;
  return inputs;
}

CpAttentionMeta build_attention_meta(
    const CpInputShardMeta& shard_meta,
    int32_t cp_size,
    bool have_prefix_slots,
    const std::vector<int32_t>& kv_cache_tokens_per_seq,
    const torch::Tensor& block_tables,
    int32_t block_size,
    int32_t kv_split_size) {
  CHECK_GT(cp_size, 1) << "cp_size must be > 1";
  const torch::Tensor local_padded_seq_lens =
      torch::tensor(shard_meta.local_padded_seq_lens,
                    torch::dtype(torch::kInt32).device(torch::kCPU));
  CpAttentionMeta meta =
      build_attention_tensor_meta(cp_size,
                                  shard_meta.local_padded_token_count,
                                  shard_meta.local_position_ids,
                                  local_padded_seq_lens,
                                  have_prefix_slots,
                                  kv_cache_tokens_per_seq,
                                  block_size,
                                  kv_split_size);
  if (have_prefix_slots && block_tables.defined()) {
    meta.prefix_cache_slots = build_prefix_cache_slots(
        block_tables, kv_cache_tokens_per_seq, block_size, kv_split_size);
  }
  return meta;
}

CpInputShardMeta build_input_shard_meta(
    int32_t cp_size,
    int32_t cp_rank,
    const std::vector<int32_t>& global_q_seq_lens,
    const torch::Tensor& global_positions) {
  CHECK_GT(cp_size, 1) << "NPU CP plan requires cp_size > 1";
  CHECK_GE(cp_rank, 0) << "cp_rank out of range";
  CHECK_LT(cp_rank, cp_size) << "cp_rank out of range";

  CpInputShardMeta shard_meta;

  const int32_t num_chunks = cp_size * 2;
  const int32_t num_sequences = static_cast<int32_t>(global_q_seq_lens.size());

  int64_t global_real_token_count = 0;
  int64_t local_padded_token_count = 0;
  int64_t local_real_token_count = 0;

  std::vector<int64_t> source_vec;
  std::vector<int64_t> dest_vec;
  std::vector<int32_t> virtual_pos_vec;
  std::vector<int32_t> local_q_seq_lens;
  std::vector<int32_t> local_padded_seq_lens;
  local_q_seq_lens.reserve(num_sequences);
  local_padded_seq_lens.reserve(num_sequences);

  const int32_t* positions_data = global_positions.defined()
                                      ? global_positions.data_ptr<int32_t>()
                                      : nullptr;

  int64_t seq_token_offset = 0;  // global-real token offset of current seq
  int64_t local_offset = 0;  // offset within this rank's local-padded hidden
  for (int32_t i = 0; i < num_sequences; ++i) {
    const int32_t q_i = std::max(0, global_q_seq_lens[i]);
    const int32_t p_i =
        ((q_i + num_chunks - 1) / num_chunks) * num_chunks;  // align_up to 2*cp
    const int32_t chunk_i = p_i / num_chunks;
    const int32_t local_len_i = 2 * chunk_i;  // constant across ranks
    int32_t local_real_len_i = 0;  // real tokens owned by this rank for seq i

    const int32_t pos_start = (positions_data != nullptr &&
                               seq_token_offset < global_positions.numel())
                                  ? positions_data[seq_token_offset]
                                  : static_cast<int32_t>(seq_token_offset);

    // First half: chunk `cp_rank`.
    const int32_t c_first = cp_rank;
    for (int32_t j = 0; j < chunk_i; ++j) {
      const int32_t intra = c_first * chunk_i + j;
      const int32_t vpos = pos_start + c_first * chunk_i + j;
      virtual_pos_vec.push_back(vpos);
      if (intra < q_i) {
        source_vec.push_back(seq_token_offset + intra);
        dest_vec.push_back(local_offset + j);
        ++local_real_token_count;
        ++local_real_len_i;
      }
    }
    // Second half: chunk `num_chunks - 1 - cp_rank`.
    const int32_t c_second = num_chunks - 1 - cp_rank;
    for (int32_t j = 0; j < chunk_i; ++j) {
      const int32_t intra = c_second * chunk_i + j;
      const int32_t vpos = pos_start + c_second * chunk_i + j;
      virtual_pos_vec.push_back(vpos);
      if (intra < q_i) {
        source_vec.push_back(seq_token_offset + intra);
        dest_vec.push_back(local_offset + chunk_i + j);
        ++local_real_token_count;
        ++local_real_len_i;
      }
    }

    // q_seq_lens / kv_seq_lens carry the real per-seq token count. The padded
    // layout is conveyed separately
    // via attn_padding_idx / local_padded_token_count so the ATB graph can
    // unpad.
    local_q_seq_lens.push_back(local_real_len_i);
    local_padded_seq_lens.push_back(local_len_i);

    global_real_token_count += q_i;
    local_padded_token_count += local_len_i;
    seq_token_offset += q_i;
    local_offset += local_len_i;
  }

  shard_meta.global_real_token_count = global_real_token_count;
  shard_meta.local_padded_token_count = local_padded_token_count;
  shard_meta.local_real_token_count = local_real_token_count;
  shard_meta.local_real_seq_lens = std::move(local_q_seq_lens);
  shard_meta.local_padded_seq_lens = std::move(local_padded_seq_lens);

  shard_meta.input_source_indices = torch::tensor(
      source_vec, torch::dtype(torch::kInt64).device(torch::kCPU));
  shard_meta.input_destination_indices =
      torch::tensor(dest_vec, torch::dtype(torch::kInt64).device(torch::kCPU));
  shard_meta.local_position_ids = torch::tensor(
      virtual_pos_vec, torch::dtype(torch::kInt32).device(torch::kCPU));

  return shard_meta;
}

CpOutputMergeMeta build_output_merge_meta(
    int32_t cp_size,
    const std::vector<int32_t>& global_q_seq_lens,
    int64_t local_padded_token_count) {
  CpOutputMergeMeta merge_meta;
  const int32_t num_chunks = cp_size * 2;
  const int32_t num_sequences = static_cast<int32_t>(global_q_seq_lens.size());

  // output_restore_indices[g] = offset into the CP rank-major all-gathered
  // hidden (size global_padded_token_count) where global-real token g lives.
  std::vector<int64_t> restore_vec;
  int64_t global_padded_token_count = 0;
  int64_t gather_seq_offset =
      0;  // per-rank offset of seq i (same on all ranks)
  for (int32_t i = 0; i < num_sequences; ++i) {
    const int32_t q_i = std::max(0, global_q_seq_lens[i]);
    const int32_t p_i = ((q_i + num_chunks - 1) / num_chunks) * num_chunks;
    const int32_t chunk_i = p_i / num_chunks;
    for (int32_t t = 0; t < q_i; ++t) {
      const int32_t c = t / chunk_i;
      const int32_t k = t % chunk_i;
      int32_t r_owner;
      int32_t row_in_seq;
      if (c < cp_size) {
        r_owner = c;  // first half on rank c
        row_in_seq = k;
      } else {
        r_owner = num_chunks - 1 - c;  // second half on rank (num_chunks-1-c)
        row_in_seq = chunk_i + k;
      }
      const int64_t gathered_idx =
          static_cast<int64_t>(r_owner) * local_padded_token_count +
          gather_seq_offset + row_in_seq;
      restore_vec.push_back(gathered_idx);
    }
    gather_seq_offset += 2 * chunk_i;
    global_padded_token_count += p_i;
  }
  merge_meta.global_padded_token_count = global_padded_token_count;
  merge_meta.output_restore_indices = torch::tensor(
      restore_vec, torch::dtype(torch::kInt64).device(torch::kCPU));
  return merge_meta;
}

std::vector<int32_t> preserve_length_layout(const std::vector<int32_t>& lengths,
                                            bool cumulative) {
  if (!cumulative) {
    return lengths;
  }
  std::vector<int32_t> cumulative_lengths = {0};
  cumulative_lengths.reserve(lengths.size() + 1);
  for (int32_t length : lengths) {
    cumulative_lengths.push_back(cumulative_lengths.back() + length);
  }
  return cumulative_lengths;
}

float get_cp_ep_buffer_factor(int64_t length, int32_t attention_cp_size) {
  length *= attention_cp_size;
  const std::vector<std::pair<int64_t, float>> thresholds = {{1048576, 1.32f},
                                                             {524288, 1.4f},
                                                             {262144, 1.53f},
                                                             {131072, 1.8f},
                                                             {32768, 3.0f},
                                                             {8192, 5.2f},
                                                             {0, 8.0f}};
  for (const auto& threshold : thresholds) {
    if (length >= threshold.first) {
      return threshold.second;
    }
  }
  return 8.0f;
}

CpEpMeta build_cp_ep_meta(int64_t local_padded_token_count,
                          const CpPlanConfig& config) {
  CHECK_GT(config.attention_tp_size, 0);
  CHECK_GT(config.attention_cp_size, 0);
  CHECK_GT(config.attention_cp_group_size, 0);
  CHECK_GE(config.attention_tp_rank, 0);
  CHECK_LT(config.attention_tp_rank, config.attention_tp_size);

  const int64_t input_length = std::max<int64_t>(local_padded_token_count, 1);
  const int64_t padding_length =
      (config.attention_tp_size - input_length % config.attention_tp_size) %
      config.attention_tp_size;
  const int64_t padded_group_length = input_length + padding_length;
  const int64_t padded_rank_length =
      padded_group_length / config.attention_tp_size;

  CpEpMeta meta;
  meta.attention_tp_padding_indices =
      torch::cat({torch::arange(input_length, torch::kInt32),
                  torch::zeros({padding_length}, torch::kInt32)});
  meta.prenorm_gather_indices = meta.attention_tp_padding_indices.slice(
      /*dim=*/0,
      config.attention_tp_rank * padded_rank_length,
      (config.attention_tp_rank + 1) * padded_rank_length);

  std::vector<torch::Tensor> all_gather_indices;
  all_gather_indices.reserve(config.attention_cp_group_size);
  for (int32_t cp_rank = 0; cp_rank < config.attention_cp_group_size;
       ++cp_rank) {
    all_gather_indices.emplace_back(torch::arange(input_length, torch::kInt32) +
                                    cp_rank * padded_group_length);
  }
  torch::Tensor skip_padding_indices = torch::cat(all_gather_indices, 0);

  const bool dynamic_ep =
      config.moe_ep_size > 1 &&
      (config.expert_parallel_degree == 2 ||
       (config.expert_parallel_degree == 3 && config.is_prefill));
  if (dynamic_ep) {
    meta.attention_tp_unpadding_indices =
        torch::arange(padded_rank_length, torch::kInt32);
    meta.ffn_padding_indices = meta.attention_tp_unpadding_indices;
  } else {
    meta.attention_tp_unpadding_indices = skip_padding_indices;
    std::vector<torch::Tensor> ffn_padding_indices;
    ffn_padding_indices.reserve(config.attention_cp_group_size);
    for (int32_t cp_rank = 0; cp_rank < config.attention_cp_group_size;
         ++cp_rank) {
      ffn_padding_indices.emplace_back(
          torch::cat({torch::arange(input_length * cp_rank,
                                    input_length * (cp_rank + 1),
                                    torch::kInt32),
                      torch::zeros({padding_length}, torch::kInt32)}));
    }
    meta.ffn_padding_indices = torch::cat(ffn_padding_indices, 0);
  }

  meta.attention_padding_indices = meta.attention_tp_padding_indices;
  meta.attention_unpadding_indices = torch::zeros({1}, torch::kInt32);
  meta.ffn_unpadding_indices = torch::arange(input_length, torch::kInt32);
  meta.lm_head_skip_padding_indices = skip_padding_indices;

  if (!dynamic_ep) {
    meta.dynamic_ep_indices = torch::zeros({1}, torch::kInt32);
    meta.moe_indices = torch::zeros({1}, torch::kInt32);
    meta.expert_array = torch::tensor({0});
    return meta;
  }

  if (config.attention_tp_size == 1) {
    meta.dynamic_ep_indices = torch::arange(
        input_length * config.num_experts_per_token, torch::kInt32);
  } else {
    meta.dynamic_ep_indices =
        torch::arange(meta.attention_tp_unpadding_indices.size(0) *
                          config.num_experts_per_token,
                      torch::kInt32);
  }
  const int64_t base_length =
      config.attention_tp_size == 1
          ? meta.attention_tp_unpadding_indices.size(0) *
                config.num_experts_per_token
          : meta.dynamic_ep_indices.size(0);
  const float buffer_factor =
      get_cp_ep_buffer_factor(base_length, config.attention_cp_size);
  int64_t ep_input_length =
      static_cast<int64_t>(static_cast<float>(base_length) * buffer_factor);
  const int64_t all_to_all_remainder = ep_input_length % config.moe_ep_size;
  if (all_to_all_remainder != 0) {
    ep_input_length += config.moe_ep_size - all_to_all_remainder;
  }
  meta.moe_indices = torch::arange(1, ep_input_length + 1, torch::kInt32);
  meta.expert_array =
      torch::ones({ep_input_length}, config.dtype).view({-1, 1});
  return meta;
}

CpInputShardMeta copy_input_shard_meta_to(const CpInputShardMeta& meta,
                                          const torch::Device& device) {
  CpInputShardMeta result = meta;
  result.input_source_indices =
      safe_to(meta.input_source_indices, device, true);
  result.input_destination_indices =
      safe_to(meta.input_destination_indices, device, true);
  result.local_position_ids = safe_to(meta.local_position_ids, device, true);
  return result;
}

CpOutputMergeMeta copy_output_merge_meta_to(const CpOutputMergeMeta& meta,
                                            const torch::Device& device) {
  CpOutputMergeMeta result = meta;
  result.output_restore_indices =
      safe_to(meta.output_restore_indices, device, true);
  return result;
}

CpAttentionMeta copy_attention_meta_to(const CpAttentionMeta& meta,
                                       const torch::Device& device) {
  CpAttentionMeta result = meta;
  result.q_seq_lens = safe_to(meta.q_seq_lens, device, true);
  result.kv_seq_lens = safe_to(meta.kv_seq_lens, device, true);
  result.q_cu_seq_lens = safe_to(meta.q_cu_seq_lens, device, true);
  result.query_balance_indices =
      safe_to(meta.query_balance_indices, device, true);
  result.attention_output_reorder_indices =
      safe_to(meta.attention_output_reorder_indices, device, true);
  result.kv_reorder_indices = safe_to(meta.kv_reorder_indices, device, true);
  result.prev_kv_gather_indices =
      safe_to(meta.prev_kv_gather_indices, device, true);
  result.next_kv_gather_indices =
      safe_to(meta.next_kv_gather_indices, device, true);
  result.prev_query_cu_seq_lens =
      safe_to(meta.prev_query_cu_seq_lens, device, true);
  result.next_query_cu_seq_lens =
      safe_to(meta.next_query_cu_seq_lens, device, true);
  result.prev_key_cu_seq_lens =
      safe_to(meta.prev_key_cu_seq_lens, device, true);
  result.next_key_cu_seq_lens =
      safe_to(meta.next_key_cu_seq_lens, device, true);
  result.prefix_cache_slots = safe_to(meta.prefix_cache_slots, device, true);
  return result;
}

CpEpMeta copy_cp_ep_meta_to(const CpEpMeta& meta, const torch::Device& device) {
  CpEpMeta result;
  result.attention_tp_padding_indices =
      safe_to(meta.attention_tp_padding_indices, device, true);
  result.attention_tp_unpadding_indices =
      safe_to(meta.attention_tp_unpadding_indices, device, true);
  result.ffn_padding_indices = safe_to(meta.ffn_padding_indices, device, true);
  result.ffn_unpadding_indices =
      safe_to(meta.ffn_unpadding_indices, device, true);
  result.lm_head_skip_padding_indices =
      safe_to(meta.lm_head_skip_padding_indices, device, true);
  result.prenorm_gather_indices =
      safe_to(meta.prenorm_gather_indices, device, true);
  result.attention_padding_indices =
      safe_to(meta.attention_padding_indices, device, true);
  result.attention_unpadding_indices =
      safe_to(meta.attention_unpadding_indices, device, true);
  result.dynamic_ep_indices = safe_to(meta.dynamic_ep_indices, device, true);
  result.moe_indices = safe_to(meta.moe_indices, device, true);
  result.expert_array = safe_to(meta.expert_array, device, true);
  return result;
}

torch::Tensor map_cache_slots_to_kv_shard(
    const torch::Tensor& recovered_logical_slots,
    int32_t block_size,
    int32_t kv_split_size,
    int32_t kv_split_rank) {
  CHECK_GT(block_size, 0);
  CHECK_GT(kv_split_size, 0);
  CHECK_GE(kv_split_rank, 0);
  CHECK_LT(kv_split_rank, kv_split_size);

  const int32_t logical_block_size = block_size * kv_split_size;
  torch::Tensor row_indices =
      torch::arange(recovered_logical_slots.numel(), torch::kCPU);
  torch::Tensor logical_block_offsets = row_indices % logical_block_size;
  torch::Tensor row_kv_split_ranks =
      torch::floor_divide(logical_block_offsets, block_size);
  torch::Tensor local_row_indices =
      torch::nonzero(row_kv_split_ranks == kv_split_rank).flatten();

  torch::Tensor physical_slots = torch::full_like(recovered_logical_slots, -1);
  if (local_row_indices.numel() == 0) {
    return physical_slots;
  }

  const torch::Tensor device_row_indices = local_row_indices.to(
      recovered_logical_slots.device(), /*non_blocking=*/false);
  torch::Tensor logical_slots =
      recovered_logical_slots.index_select(/*dim=*/0, device_row_indices)
          .to(torch::kInt32);
  torch::Tensor logical_block_ids =
      torch::floor_divide(logical_slots, logical_block_size);
  torch::Tensor physical_block_offsets = logical_slots % block_size;
  torch::Tensor local_physical_slots =
      logical_block_ids * block_size + physical_block_offsets;
  physical_slots.index_put_({device_row_indices},
                            local_physical_slots.to(physical_slots.dtype()));
  return physical_slots;
}

}  // namespace

NpuCpPlan NpuCpPlan::build(const CpPlanInput& input,
                           const CpPlanConfig& config) {
  CHECK_GT(config.cp_size, 1) << "NPU CP plan requires cp_size > 1";
  CHECK_GE(config.cp_rank, 0);
  CHECK_LT(config.cp_rank, config.cp_size);
  CHECK_GT(config.kv_split_size, 0);
  CHECK_GE(config.kv_split_rank, 0);
  CHECK_LT(config.kv_split_rank, config.kv_split_size);
  CHECK_GT(config.block_size, 0);
  if (input.has_prefix_slots) {
    CHECK_EQ(input.q_seq_lens.size(), input.prefix_token_counts.size());
  }
  int64_t global_token_count = 0;
  for (int32_t seq_len : input.q_seq_lens) {
    CHECK_GE(seq_len, 0);
    global_token_count += seq_len;
  }
  CHECK(input.position_ids.defined());
  CHECK(input.position_ids.device().is_cpu());
  CHECK_EQ(input.position_ids.scalar_type(), torch::kInt32);
  CHECK_EQ(input.position_ids.numel(), global_token_count)
      << "NPU CP plan must be built from global, unsharded positions";

  NpuCpPlan plan;
  plan.enabled_ = true;
  plan.cp_size_ = config.cp_size;
  plan.cp_rank_ = config.cp_rank;
  plan.kv_split_size_ = config.kv_split_size;
  plan.kv_split_rank_ = config.kv_split_rank;
  plan.block_size_ = config.block_size;
  plan.input_shard_meta_ = build_input_shard_meta(
      config.cp_size, config.cp_rank, input.q_seq_lens, input.position_ids);
  plan.attention_meta_ = build_attention_meta(plan.input_shard_meta_,
                                              config.cp_size,
                                              input.has_prefix_slots,
                                              input.prefix_token_counts,
                                              input.block_tables,
                                              config.block_size,
                                              config.kv_split_size);
  plan.attention_meta_.host_q_seq_lens =
      preserve_length_layout(plan.input_shard_meta_.local_padded_seq_lens,
                             input.q_seq_lens_are_cumulative);
  plan.attention_meta_.host_kv_seq_lens =
      preserve_length_layout(plan.input_shard_meta_.local_padded_seq_lens,
                             input.kv_seq_lens_are_cumulative);
  plan.attention_meta_.host_q_cu_seq_lens.reserve(
      plan.input_shard_meta_.local_padded_seq_lens.size());
  int32_t cumulative_length = 0;
  int32_t max_seq_len = 0;
  for (int32_t length : plan.input_shard_meta_.local_padded_seq_lens) {
    cumulative_length += length;
    plan.attention_meta_.host_q_cu_seq_lens.push_back(cumulative_length);
    max_seq_len = std::max(max_seq_len, length);
  }
  plan.attention_meta_.q_max_seq_len = max_seq_len;
  plan.attention_meta_.kv_max_seq_len = max_seq_len;

  const torch::TensorOptions cpu_int32 =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
  plan.attention_meta_.q_seq_lens =
      torch::tensor(plan.attention_meta_.host_q_seq_lens, cpu_int32);
  plan.attention_meta_.kv_seq_lens =
      torch::tensor(plan.attention_meta_.host_kv_seq_lens, cpu_int32);
  plan.attention_meta_.q_cu_seq_lens =
      torch::tensor(plan.attention_meta_.host_q_cu_seq_lens, cpu_int32);
  plan.cp_ep_meta_ =
      build_cp_ep_meta(plan.input_shard_meta_.local_padded_token_count, config);
  plan.output_merge_meta_ =
      build_output_merge_meta(config.cp_size,
                              input.q_seq_lens,
                              plan.input_shard_meta_.local_padded_token_count);
  return plan.to(config.device);
}

NpuCpPlan NpuCpPlan::to(const torch::Device& device) const {
  NpuCpPlan result = *this;
  result.input_shard_meta_ =
      copy_input_shard_meta_to(input_shard_meta_, device);
  result.attention_meta_ = copy_attention_meta_to(attention_meta_, device);
  result.cp_ep_meta_ = copy_cp_ep_meta_to(cp_ep_meta_, device);
  result.output_merge_meta_ =
      copy_output_merge_meta_to(output_merge_meta_, device);
  return result;
}

CpInputShard NpuCpPlan::shard_model_input(
    const torch::Tensor& global_hidden_states,
    const torch::Tensor& global_position_ids) const {
  if (!enabled()) {
    return {global_hidden_states, global_position_ids};
  }
  CHECK_EQ(global_hidden_states.dim(), 2);
  if (input_shard_meta_.global_real_token_count == 0) {
    CHECK_EQ(input_shard_meta_.local_padded_token_count, 0);
    return {global_hidden_states.slice(/*dim=*/0, /*start=*/0, /*end=*/0),
            input_shard_meta_.local_position_ids};
  }
  CHECK_EQ(global_hidden_states.size(0),
           input_shard_meta_.global_real_token_count)
      << "NPU CP model input must be sharded exactly once from global layout";
  CHECK_EQ(global_position_ids.numel(),
           input_shard_meta_.global_real_token_count)
      << "NPU CP positions must be sharded exactly once from global layout";

  torch::Tensor local_hidden_states =
      torch::zeros({input_shard_meta_.local_padded_token_count,
                    global_hidden_states.size(1)},
                   global_hidden_states.options());
  torch::Tensor local_source = global_hidden_states.index_select(
      /*dim=*/0, input_shard_meta_.input_source_indices);
  local_hidden_states.index_put_({input_shard_meta_.input_destination_indices},
                                 local_source);
  return {local_hidden_states, input_shard_meta_.local_position_ids};
}

void NpuCpPlan::apply_attention_meta(ModelInputParams& params) const {
  if (!enabled()) {
    return;
  }
  params.attention.host.q_seq_lens = attention_meta_.host_q_seq_lens;
  params.attention.host.kv_seq_lens = attention_meta_.host_kv_seq_lens;
  params.attention.host.q_cu_seq_lens = attention_meta_.host_q_cu_seq_lens;
  params.attention.device.q_seq_lens = attention_meta_.q_seq_lens;
  params.attention.device.kv_seq_lens = attention_meta_.kv_seq_lens;
  params.attention.device.q_cu_seq_lens = attention_meta_.q_cu_seq_lens;
  if (attention_meta_.prefix_cache_slots.defined()) {
    params.attention.device.in_prefix_slots =
        attention_meta_.prefix_cache_slots;
  }
  params.meta.q_max_seq_len = attention_meta_.q_max_seq_len;
  params.meta.kv_max_seq_len = attention_meta_.kv_max_seq_len;
}

torch::Tensor NpuCpPlan::prepare_cache_slots(
    const torch::Tensor& global_logical_slots) const {
  if (!enabled()) {
    return global_logical_slots;
  }
  CHECK_EQ(global_logical_slots.dim(), 1);
  const int64_t gathered_rows = recovered_token_count();
  CHECK_EQ(global_logical_slots.numel(),
           input_shard_meta_.global_real_token_count)
      << "NPU CP cache slots must use the global-real logical layout";

  torch::Tensor gathered_slots =
      torch::full({gathered_rows}, -1, global_logical_slots.options());
  gathered_slots.index_put_({output_merge_meta_.output_restore_indices},
                            global_logical_slots);
  torch::Tensor recovered_logical_slots = gathered_slots.index_select(
      /*dim=*/0, attention_meta_.kv_reorder_indices.to(torch::kLong));
  return map_cache_slots_to_kv_shard(
      recovered_logical_slots, block_size_, kv_split_size_, kv_split_rank_);
}

torch::Tensor NpuCpPlan::merge_model_output(
    const torch::Tensor& local_hidden_states,
    ProcessGroup* process_group) const {
  if (!enabled()) {
    return local_hidden_states;
  }
  CHECK_EQ(local_hidden_states.size(0),
           input_shard_meta_.local_padded_token_count)
      << "NPU CP output must be merged exactly once from local layout";
  CHECK(process_group != nullptr)
      << "NPU CP output merge requires a process_group";
  CHECK_EQ(process_group->world_size(), cp_size_)
      << "NPU CP output merge process_group size mismatch";
  CHECK_EQ(process_group->rank(), cp_rank_)
      << "NPU CP output merge process_group rank mismatch";
  torch::Tensor gathered =
      parallel_state::gather(local_hidden_states, process_group, /*dim=*/0);
  return gathered.index_select(/*dim=*/0,
                               output_merge_meta_.output_restore_indices);
}

void NpuCpPlan::replace_cp_ep_meta_storage(CpEpMeta meta) {
  cp_ep_meta_ = std::move(meta);
}

}  // namespace xllm

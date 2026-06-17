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

#include "framework/parallel_state/npu_cp_prepare.h"

#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace xllm {

torch::Tensor generate_cp_load_balance_idx(const torch::Tensor& input_lengths) {
  TORCH_CHECK(input_lengths.dtype() == torch::kInt32,
              "input_lengths must be int32 tensor");
  TORCH_CHECK(input_lengths.dim() == 1, "input_lengths must be 1D tensor");

  std::vector<int> lengths_vec;
  int* lengths_ptr = input_lengths.data_ptr<int>();
  int64_t n = input_lengths.numel();
  for (int64_t i = 0; i < n; ++i) {
    lengths_vec.push_back(lengths_ptr[i]);
  }

  std::vector<int> cp_load_balance_idx_first, cp_load_balance_idx_last;
  int base = 0;
  for (int length : lengths_vec) {
    std::vector<int> length_range(length);
    std::iota(length_range.begin(), length_range.end(), base);
    int divider = length / 2;
    cp_load_balance_idx_first.insert(cp_load_balance_idx_first.end(),
                                     length_range.begin(),
                                     length_range.begin() + divider);
    cp_load_balance_idx_last.insert(cp_load_balance_idx_last.end(),
                                    length_range.begin() + divider,
                                    length_range.end());
    base += length;
  }

  cp_load_balance_idx_first.insert(cp_load_balance_idx_first.end(),
                                   cp_load_balance_idx_last.begin(),
                                   cp_load_balance_idx_last.end());

  auto tensor = torch::tensor(cp_load_balance_idx_first,
                              torch::dtype(torch::kInt32).device(torch::kCPU));
  return tensor;
}

torch::Tensor generate_cp_o_recover_idx(const std::vector<int>& chunk_lengths) {
  std::vector<int> cp_o_recover_idx;
  int base = 0;
  int chunk_lengths_sum =
      std::accumulate(chunk_lengths.begin(), chunk_lengths.end(), 0);

  for (int chunk_len : chunk_lengths) {
    std::vector<int> length_range(chunk_len);
    std::iota(length_range.begin(), length_range.end(), base);
    cp_o_recover_idx.insert(
        cp_o_recover_idx.end(), length_range.begin(), length_range.end());
    std::vector<int> last_part(length_range.size());
    std::transform(
        length_range.begin(),
        length_range.end(),
        last_part.begin(),
        [chunk_lengths_sum](int x) { return x + chunk_lengths_sum; });
    cp_o_recover_idx.insert(
        cp_o_recover_idx.end(), last_part.begin(), last_part.end());
    base += chunk_len;
  }

  return torch::tensor(cp_o_recover_idx,
                       torch::dtype(torch::kInt32).device(torch::kCPU));
}

torch::Tensor generate_cp_kv_recover_idx(
    int cp_size,
    int input_ids_size,
    const std::vector<int>& chunk_lengths) {
  std::vector<int> cp_kv_recover_idx;
  int req_offset = 0;

  for (int req_chunk_len : chunk_lengths) {
    std::vector<std::vector<int>> gather_idx_per_chunk(cp_size * 2);
    for (int cp_rank_id = 0; cp_rank_id < cp_size; ++cp_rank_id) {
      int rank_offset = cp_rank_id * input_ids_size;
      std::vector<int> first_part(req_chunk_len);
      std::iota(first_part.begin(), first_part.end(), rank_offset + req_offset);
      gather_idx_per_chunk[cp_rank_id] = first_part;

      std::vector<int> last_part(req_chunk_len);
      std::iota(last_part.begin(),
                last_part.end(),
                rank_offset + req_offset + req_chunk_len);
      gather_idx_per_chunk[cp_size * 2 - 1 - cp_rank_id] = last_part;
    }

    for (const auto& vec : gather_idx_per_chunk) {
      cp_kv_recover_idx.insert(cp_kv_recover_idx.end(), vec.begin(), vec.end());
    }
    req_offset += req_chunk_len * 2;
  }

  return torch::tensor(cp_kv_recover_idx,
                       torch::dtype(torch::kInt32).device(torch::kCPU));
}

std::pair<torch::Tensor, torch::Tensor> compute_input_lengths_cumsum_cp(
    const torch::Tensor& input_lengths_cumsum) {
  TORCH_CHECK(input_lengths_cumsum.dtype() == torch::kInt32,
              "input_lengths_cumsum must be int32 tensor");
  TORCH_CHECK(input_lengths_cumsum.dim() == 1,
              "input_lengths_cumsum must be 1D tensor");

  int64_t n = input_lengths_cumsum.numel();
  auto input_lengths_cumsum_cp_prev =
      torch::zeros({n}, torch::dtype(torch::kInt32).device(torch::kCPU));
  auto input_lengths_cumsum_cp_next =
      torch::zeros({n}, torch::dtype(torch::kInt32).device(torch::kCPU));

  int offset = 0;
  auto cumsum_data = input_lengths_cumsum.data_ptr<int>();
  auto prev_data = input_lengths_cumsum_cp_prev.data_ptr<int>();
  auto next_data = input_lengths_cumsum_cp_next.data_ptr<int>();

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
    int cp_size) {
  TORCH_CHECK(actual_seq_lengths_kv_cp_prev.dim() == 1,
              "actual_seq_lengths_kv_cp_prev must be 1D");
  TORCH_CHECK(actual_seq_lengths_kv_cp_next.dim() == 1,
              "actual_seq_lengths_kv_cp_next must be 1D");
  TORCH_CHECK(input_lengths.dim() == 1, "input_lengths must be 1D");

  std::vector<int> k_gather_index_prev, k_gather_index_next;
  int k_offset = 0;
  int64_t n = input_lengths.numel();

  auto prev_len_data = actual_seq_lengths_kv_cp_prev.data_ptr<int>();
  auto next_len_data = actual_seq_lengths_kv_cp_next.data_ptr<int>();
  auto input_len_data = input_lengths.data_ptr<int>();

  for (int64_t i = 0; i < n; ++i) {
    std::vector<int> prev_range(prev_len_data[i]);
    std::iota(prev_range.begin(), prev_range.end(), k_offset);
    k_gather_index_prev.insert(
        k_gather_index_prev.end(), prev_range.begin(), prev_range.end());

    std::vector<int> next_range(next_len_data[i]);
    std::iota(next_range.begin(), next_range.end(), k_offset);
    k_gather_index_next.insert(
        k_gather_index_next.end(), next_range.begin(), next_range.end());

    k_offset += input_len_data[i] * cp_size;
  }

  auto prev_tensor = torch::tensor(
      k_gather_index_prev, torch::dtype(torch::kInt32).device(torch::kCPU));
  auto next_tensor = torch::tensor(
      k_gather_index_next, torch::dtype(torch::kInt32).device(torch::kCPU));
  return {prev_tensor, next_tensor};
}

namespace {

// Build the per-rank prefix geometry that mirrors
// `WorkerImpl::compute_in_prefix_slots`:
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
    int cp_size,
    int block_size) {
  TORCH_CHECK(cp_size > 0, "cp_size must be positive");
  TORCH_CHECK(block_size > 0, "block_size must be positive");

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

}  // namespace

// Gather index over `intermediate_kv` (current segment of `merged_kv`, which
// is per-seq grouped after `cp_kv_recover_idx` reorder; each seq occupies
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
    int cp_size) {
  TORCH_CHECK(current_lengths_kv_cp_prev.dim() == 1,
              "current_lengths_kv_cp_prev must be 1D");
  TORCH_CHECK(current_lengths_kv_cp_next.dim() == 1,
              "current_lengths_kv_cp_next must be 1D");
  TORCH_CHECK(input_lengths.dim() == 1, "input_lengths must be 1D");
  TORCH_CHECK(current_lengths_kv_cp_prev.numel() == input_lengths.numel(),
              "current_lengths_kv_cp_prev size mismatch");
  TORCH_CHECK(current_lengths_kv_cp_next.numel() == input_lengths.numel(),
              "current_lengths_kv_cp_next size mismatch");
  TORCH_CHECK(cp_size > 0, "cp_size must be positive");

  const int64_t n = input_lengths.numel();
  auto prev_len_data = current_lengths_kv_cp_prev.data_ptr<int>();
  auto next_len_data = current_lengths_kv_cp_next.data_ptr<int>();
  auto input_len_data = input_lengths.data_ptr<int>();

  std::vector<int> prev_idx, next_idx;
  int k_offset = 0;
  for (int64_t i = 0; i < n; ++i) {
    std::vector<int> prev_range(prev_len_data[i]);
    std::iota(prev_range.begin(), prev_range.end(), k_offset);
    prev_idx.insert(prev_idx.end(), prev_range.begin(), prev_range.end());

    std::vector<int> next_range(next_len_data[i]);
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
    int kv_split_size,
    int block_size) {
  const auto geom = compute_prefix_rank_geometry(
      kv_cache_tokens_per_seq, kv_split_size, block_size);
  const int64_t n = static_cast<int64_t>(kv_cache_tokens_per_seq.size());

  std::vector<int> ctx_idx;
  for (int64_t i = 0; i < n; ++i) {
    if (geom.real_len_in_rank[i] == 0) {
      // Prefix-less seq: only the padding slot lives in merged_kv's prefix
      // segment and it must not be gathered.
      continue;
    }
    for (int64_t j = 0; j < kv_split_size; ++j) {
      std::vector<int> prefix_range(geom.cache_len_in_rank[i]);
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
    const torch::Tensor& current_k_gather_index_prev,
    const torch::Tensor& current_k_gather_index_next,
    const torch::Tensor& history_k_gather_index_prev,
    const torch::Tensor& history_k_gather_index_next,
    const torch::Tensor& current_lengths_kv_cp_prev,
    const torch::Tensor& current_lengths_kv_cp_next,
    const torch::Tensor& input_lengths,
    const std::vector<int32_t>& kv_cache_tokens_per_seq,
    int kv_split_size,
    int block_size) {
  // NOTE: `kv_split_size` (was `cp_size`) only governs the PREFIX-segment
  // geometry: rank stride for AllGather slices and `prefix_total_len`. The
  // CURRENT-segment indices passed in via current_k_gather_index_* are
  // generated upstream with token-CP cp_size, so the two widths can differ.
  TORCH_CHECK(input_lengths.dim() == 1, "input_lengths must be 1D");
  TORCH_CHECK(current_lengths_kv_cp_prev.dim() == 1,
              "current_lengths_kv_cp_prev must be 1D");
  TORCH_CHECK(current_lengths_kv_cp_next.dim() == 1,
              "current_lengths_kv_cp_next must be 1D");
  TORCH_CHECK(static_cast<size_t>(input_lengths.numel()) ==
                  kv_cache_tokens_per_seq.size(),
              "input_lengths must equal kv_cache_tokens_per_seq size");

  const auto geom = compute_prefix_rank_geometry(
      kv_cache_tokens_per_seq, kv_split_size, block_size);
  const int32_t prefix_total_len =
      static_cast<int32_t>(geom.rank_block_size * kv_split_size);
  const int64_t n = input_lengths.numel();

  auto current_prev_len_data = current_lengths_kv_cp_prev.data_ptr<int>();
  auto current_next_len_data = current_lengths_kv_cp_next.data_ptr<int>();
  auto current_prev_data = current_k_gather_index_prev.data_ptr<int>();
  auto current_next_data = current_k_gather_index_next.data_ptr<int>();
  auto history_prev_data = history_k_gather_index_prev.data_ptr<int>();
  auto history_next_data = history_k_gather_index_next.data_ptr<int>();

  std::vector<int> merged_prev, merged_next;
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

CpPrefillInputs prepare_cp_prefill_inputs(
    int cp_size,
    const torch::Tensor& input_ids,
    const torch::Tensor& position_ids,
    const torch::Tensor& input_lengths,
    bool have_prefix_slots,
    const std::vector<int32_t>& kv_cache_tokens_per_seq,
    int block_size,
    int kv_split_size) {
  TORCH_CHECK(cp_size > 0, "cp_size must be positive");
  // Default kv_split_size to cp_size to preserve legacy behavior (prefix
  // geometry was implicitly bound to cp_size before the KV-split / CP
  // decoupling refactor).
  if (kv_split_size <= 0) {
    kv_split_size = cp_size;
  }
  TORCH_CHECK(kv_split_size > 0, "kv_split_size must resolve to positive");
  TORCH_CHECK(cp_size % kv_split_size == 0,
              "cp_size (",
              cp_size,
              ") must be divisible by kv_split_size (",
              kv_split_size,
              ").");
  CpPrefillInputs inputs;

  std::vector<int> chunk_lengths;
  auto input_len_data = input_lengths.data_ptr<int>();
  for (int64_t i = 0; i < input_lengths.numel(); ++i) {
    chunk_lengths.push_back(input_len_data[i] / 2);
  }

  inputs.cp_load_balance_idx = generate_cp_load_balance_idx(input_lengths);

  inputs.cp_o_recover_idx = generate_cp_o_recover_idx(chunk_lengths);

  inputs.cp_kv_recover_idx =
      generate_cp_kv_recover_idx(cp_size, input_ids.numel(), chunk_lengths);

  auto input_lengths_cumsum = torch::cumsum(input_lengths, 0, torch::kInt32);
  auto [input_lengths_cumsum_cp_prev, input_lengths_cumsum_cp_next] =
      compute_input_lengths_cumsum_cp(input_lengths_cumsum);

  auto gather_index_prev = (input_lengths_cumsum_cp_prev - 1).to(torch::kLong);
  auto gather_index_next = (input_lengths_cumsum_cp_next - 1).to(torch::kLong);
  auto position_ids_prev = position_ids.index_select(0, gather_index_prev) + 1;
  auto position_ids_next = position_ids.index_select(0, gather_index_next) + 1;
  auto actual_seq_lengths_kv_cp_prev = position_ids_prev.to(torch::kInt32);
  auto actual_seq_lengths_kv_cp_next = position_ids_next.to(torch::kInt32);

  if (have_prefix_slots) {
    // Strip the per-seq full-prefix length from the SFA-logical KV lengths to
    // obtain how much each seq's prev/next half needs from merged_kv's current
    // segment. Prefix-less seqs get prefix_kv_len_total == 0 and fall through
    // unchanged. The prefix geometry here MUST stay aligned with
    // `WorkerImpl::compute_in_prefix_slots` (single source of truth) - we go
    // through `compute_prefix_rank_geometry` to ensure that.
    const int64_t n = input_lengths.numel();
    TORCH_CHECK(static_cast<size_t>(n) == kv_cache_tokens_per_seq.size(),
                "input_lengths must equal kv_cache_tokens_per_seq size");
    // Prefix geometry is shard-aligned to `kv_split_size`, not cp_size:
    //   per_rank_prefix_tokens = total_prefix / kv_split_size
    //   prefix_kv_len_total    = per_rank_prefix * kv_split_size
    // When kv_split_size == cp_size (legacy) this is byte-identical to the
    // previous implementation.
    const auto geom = compute_prefix_rank_geometry(
        kv_cache_tokens_per_seq, kv_split_size, block_size);
    auto prev_total_data = actual_seq_lengths_kv_cp_prev.data_ptr<int>();
    auto next_total_data = actual_seq_lengths_kv_cp_next.data_ptr<int>();
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

    std::tie(inputs.k_gather_index_prev, inputs.k_gather_index_next) =
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
    std::tie(inputs.k_gather_index_prev, inputs.k_gather_index_next) =
        generate_k_gather_index(actual_seq_lengths_kv_cp_prev,
                                actual_seq_lengths_kv_cp_next,
                                input_lengths,
                                cp_size);
  }

  auto actual_seq_lengths_kv_cp_prev_cumsum =
      torch::cumsum(actual_seq_lengths_kv_cp_prev, 0, torch::kInt32);
  auto actual_seq_lengths_kv_cp_next_cumsum =
      torch::cumsum(actual_seq_lengths_kv_cp_next, 0, torch::kInt32);
  inputs.actual_seq_lengths_key_prev = actual_seq_lengths_kv_cp_prev_cumsum;
  inputs.actual_seq_lengths_key_next = actual_seq_lengths_kv_cp_next_cumsum;

  auto input_lengths_cumsum_half = torch::floor_divide(input_lengths_cumsum, 2);
  inputs.actual_seq_lengths_query_prev = input_lengths_cumsum_half;
  inputs.actual_seq_lengths_query_next = input_lengths_cumsum_half;
  return inputs;
}

}  // namespace xllm

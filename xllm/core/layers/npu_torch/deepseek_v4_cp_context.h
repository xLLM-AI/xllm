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

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include "framework/parallel_state/parallel_state.h"
#include "framework/parallel_state/process_group.h"

namespace xllm::layer::v4_cp {

// Per-forward prefill context-parallel plan for DeepSeek-V4.
//
// Why not reuse the framework's NpuCpPlan: that substrate localizes
// kv_seq_lens in apply_attention_meta(), but V4 derives every cache group's
// slot_mapping from kv_seq_lens (expand_blocks_to_slots / compute_slot_num
// scale it by the group's ratio and block_size). Shortening kv lengths would
// write KV to the wrong slots. This plan keeps kv global on purpose and
// localizes only the query axis. NpuCpPlan also splits zigzag (two chunks per
// rank, so a sequence's local rows are non-contiguous, needing two metadata
// rows) and asserts 2D hidden, while V4 metadata is one row per sequence and
// its hidden is 3D.
struct DeepseekV4CpContext {
  // Global row indices this rank owns, in ascending (global) order.
  torch::Tensor local_row_indices;
  // Maps global row -> its position in the rank-major gathered buffer, so
  // gathered.index_select(0, restore_indices) restores global order.
  torch::Tensor restore_indices;
  // Row count per rank; segments may differ in length, so the gather is uneven.
  std::vector<int32_t> tokens_per_rank;
  // True global positions of the local rows (never renumbered), so query RoPE
  // stays correct without a separate position table.
  torch::Tensor local_positions;

  // Query axis is localized; KV axis stays global. Both are needed because the
  // KV/compressor/index-cache write path runs on all tokens while attention
  // runs on this rank's queries only.
  std::vector<int32_t> global_q_seq_lens;
  std::vector<int32_t> global_kv_seq_lens;
  std::vector<int32_t> local_q_seq_lens;
  // Per-sequence kv extent this rank's queries may attend to, in global
  // coordinates: cached prefix + one past this rank's last local row.
  //
  // sparse_attn_sharedkv and the lightning indexer align the query block to the
  // END of the kv window, so the query at local row j is treated as sitting at
  // global position kv_len - local_q + j. With a contiguous split that only
  // holds for the last cp_rank. Handing every rank the global kv length shifts
  // rank r's queries to the tail of the whole sequence: rank 0 of a 145-token
  // prompt at cp=2 attends as if its rows were at 72..144 instead of 0..72, so
  // its causal mask, its window and its top-k all address the wrong prefix.
  // Only the attention / indexer read path uses this; slot_mapping, the block
  // tables, the compressor and the index-cache writes stay global because each
  // rank holds a full KV replica written from all tokens.
  std::vector<int32_t> local_kv_seq_lens;
  // (batch+1,) leading-zero cumsum of local_kv_seq_lens, matching the layout
  // DSAMetadataBuilder gives kv_cu_seq_lens.
  torch::Tensor local_kv_cu_seq_lens;
  // Global q cumulative lengths (leading 0), matching the layout
  // DSAMetadataBuilder gives actual_seq_lengths_query. The model localizes that
  // field for the attention path, so the compressor / index-cache writes --
  // which run on all tokens -- read their global view from here instead. Cached
  // at build time so the layer loop does not redo a cumsum per layer.
  torch::Tensor global_q_cu_seq_lens;

  // Global-position RoPE tables keyed by the layer's compression ratio, saved
  // before the query axis is localized. A c4 or c128 layer swaps dsa.cos to its
  // own ratio table, so the KV path needs the global table of that same ratio;
  // capturing only the ratio-1 table would apply the wrong RoPE on compressed
  // layers. Populated by the model; empty when CP is off.
  std::unordered_map<int32_t, std::pair<torch::Tensor, torch::Tensor>>
      global_rope_by_ratio;

  // Returns the global RoPE pair for `ratio`, falling back to ratio 1.
  std::pair<torch::Tensor, torch::Tensor> global_rope(int32_t ratio) const {
    auto it = global_rope_by_ratio.find(ratio);
    if (it == global_rope_by_ratio.end()) {
      it = global_rope_by_ratio.find(1);
    }
    if (it == global_rope_by_ratio.end()) {
      return {torch::Tensor(), torch::Tensor()};
    }
    return it->second;
  }

  int32_t cp_size = 1;
  int32_t cp_rank = 0;
  int32_t global_token_count = 0;
  int32_t local_token_count = 0;
  ProcessGroup* cp_group = nullptr;

  bool enabled() const { return cp_size > 1 && cp_group != nullptr; }

  // Select this rank's rows from a global-ordered tensor (dim 0).
  torch::Tensor shard_rows(const torch::Tensor& global_tensor) const {
    if (!global_tensor.defined() || !enabled()) {
      return global_tensor;
    }
    return global_tensor.index_select(/*dim=*/0,
                                      local_row_indices.to(torch::kLong));
  }

  // AllGather local rows across the CP group, then restore global order.
  torch::Tensor gather_restore(const torch::Tensor& local_tensor) const {
    if (!local_tensor.defined() || !enabled()) {
      return local_tensor;
    }
    torch::Tensor gathered = xllm::parallel_state::gather(
        local_tensor.contiguous(), cp_group, tokens_per_rank);
    return gathered.index_select(/*dim=*/0, restore_indices.to(torch::kLong));
  }
};

// Splits each sequence into cp_size contiguous segments and keeps segment
// cp_rank. Contiguous (rather than zigzag) keeps every sequence at one
// metadata row, which is what V4's c1/c4/c128/qli metadata expects. The cost is
// causal load imbalance: rank 0 gets short prefixes, the last rank long ones.
//
// Short sequences yield empty segments on the higher ranks. That is legal and
// must stay working -- chunked prefill routinely schedules chunks smaller than
// cp_size.
// Pure split arithmetic, factored out so it is unit-testable without a
// ProcessGroup or an accelerator: returns the global row indices owned by each
// rank, outer index = cp_rank.
inline std::vector<std::vector<int64_t>> compute_cp_rows_by_rank(
    int32_t cp_size,
    const std::vector<int32_t>& global_q_seq_lens) {
  CHECK_GT(cp_size, 0);
  std::vector<std::vector<int64_t>> rows_by_rank(cp_size);
  int64_t seq_base = 0;
  for (const int32_t q_i : global_q_seq_lens) {
    const int32_t seg = (q_i + cp_size - 1) / cp_size;
    for (int32_t r = 0; r < cp_size; ++r) {
      const int32_t start = std::min(r * seg, q_i);
      const int32_t end = std::min(start + seg, q_i);
      for (int32_t j = start; j < end; ++j) {
        rows_by_rank[r].push_back(seq_base + j);
      }
    }
    seq_base += q_i;
  }
  return rows_by_rank;
}

// Per-sequence local q length and the kv extent that q block ends at, for one
// rank. Factored out of build_deepseek_v4_cp_context for the same reason as
// compute_cp_rows_by_rank: it is pure arithmetic and the accuracy of the whole
// CP path hinges on it, so it must be unit-testable without a ProcessGroup.
//
// local_kv[i] is a GLOBAL position count, not a row count: cached prefix +
// one past this rank's last row in sequence i. That is what makes the kernel's
// end-of-window query alignment land on this rank's true global positions.
inline void compute_cp_local_seq_lens(
    int32_t cp_size,
    int32_t cp_rank,
    const std::vector<int32_t>& global_q_seq_lens,
    const std::vector<int32_t>& global_kv_seq_lens,
    std::vector<int32_t>* local_q_seq_lens,
    std::vector<int32_t>* local_kv_seq_lens) {
  CHECK_GT(cp_size, 0);
  CHECK_EQ(global_q_seq_lens.size(), global_kv_seq_lens.size());
  local_q_seq_lens->clear();
  local_kv_seq_lens->clear();
  local_q_seq_lens->reserve(global_q_seq_lens.size());
  local_kv_seq_lens->reserve(global_q_seq_lens.size());
  for (size_t i = 0; i < global_q_seq_lens.size(); ++i) {
    const int32_t q_i = global_q_seq_lens[i];
    const int32_t seg = (q_i + cp_size - 1) / cp_size;
    const int32_t start = std::min(cp_rank * seg, q_i);
    const int32_t end = std::min(start + seg, q_i);
    local_q_seq_lens->push_back(end - start);
    // prefix = global_kv - global_q is the cached context in front of this
    // forward's query rows; end is one past this rank's last row within them.
    const int32_t prefix = global_kv_seq_lens[i] - q_i;
    local_kv_seq_lens->push_back(prefix + end);
  }
}

inline DeepseekV4CpContext build_deepseek_v4_cp_context(
    int32_t cp_size,
    int32_t cp_rank,
    ProcessGroup* cp_group,
    const std::vector<int32_t>& global_q_seq_lens,
    const std::vector<int32_t>& global_kv_seq_lens,
    const torch::Tensor& global_positions) {
  DeepseekV4CpContext ctx;
  ctx.cp_size = cp_size;
  ctx.cp_rank = cp_rank;
  ctx.cp_group = cp_group;
  ctx.global_q_seq_lens = global_q_seq_lens;
  ctx.global_kv_seq_lens = global_kv_seq_lens;
  if (cp_size <= 1 || cp_group == nullptr) {
    return ctx;
  }
  CHECK_GE(cp_rank, 0);
  CHECK_LT(cp_rank, cp_size);
  CHECK_EQ(global_q_seq_lens.size(), global_kv_seq_lens.size())
      << "V4 CP expects one kv length per q length";

  const int32_t global_token_count = std::accumulate(
      global_q_seq_lens.begin(), global_q_seq_lens.end(), int32_t{0});
  ctx.global_token_count = global_token_count;

  const std::vector<std::vector<int64_t>> rows_by_rank =
      compute_cp_rows_by_rank(cp_size, global_q_seq_lens);

  // Local per-sequence q lengths, in the same sequence order as the input, and
  // the kv extent each of those local query blocks ends at.
  compute_cp_local_seq_lens(cp_size,
                            cp_rank,
                            global_q_seq_lens,
                            global_kv_seq_lens,
                            &ctx.local_q_seq_lens,
                            &ctx.local_kv_seq_lens);

  // gather() concatenates rank-major, so walk ranks in order to learn where
  // each global row lands, then invert: index_select needs
  // restore[global_row] = gathered_position.
  std::vector<int64_t> restore(static_cast<size_t>(global_token_count), 0);
  ctx.tokens_per_rank.reserve(cp_size);
  int64_t gathered_pos = 0;
  for (int32_t r = 0; r < cp_size; ++r) {
    ctx.tokens_per_rank.push_back(static_cast<int32_t>(rows_by_rank[r].size()));
    for (const int64_t global_row : rows_by_rank[r]) {
      restore[static_cast<size_t>(global_row)] = gathered_pos++;
    }
  }
  CHECK_EQ(gathered_pos, global_token_count)
      << "V4 CP segments must cover every global row exactly once";

  const auto cpu_i64 =
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  const torch::Device device =
      global_positions.defined() ? global_positions.device() : torch::kCPU;
  ctx.local_token_count = static_cast<int32_t>(rows_by_rank[cp_rank].size());
  ctx.local_row_indices =
      torch::tensor(rows_by_rank[cp_rank], cpu_i64).to(device);
  ctx.restore_indices = torch::tensor(restore, cpu_i64).to(device);

  // Leading 0 then cumsum, matching DSAMetadataBuilder's convention.
  std::vector<int32_t> q_cu;
  q_cu.reserve(global_q_seq_lens.size() + 1);
  q_cu.push_back(0);
  int32_t q_running = 0;
  for (const int32_t q_i : global_q_seq_lens) {
    q_running += q_i;
    q_cu.push_back(q_running);
  }
  const auto cpu_i32 =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
  ctx.global_q_cu_seq_lens = torch::tensor(q_cu, cpu_i32).to(device);

  std::vector<int32_t> kv_cu;
  kv_cu.reserve(ctx.local_kv_seq_lens.size() + 1);
  kv_cu.push_back(0);
  int32_t kv_running = 0;
  for (const int32_t kv_i : ctx.local_kv_seq_lens) {
    kv_running += kv_i;
    kv_cu.push_back(kv_running);
  }
  ctx.local_kv_cu_seq_lens = torch::tensor(kv_cu, cpu_i32).to(device);
  if (global_positions.defined()) {
    ctx.local_positions = global_positions.index_select(
        /*dim=*/0, ctx.local_row_indices.to(torch::kLong));
  }
  return ctx;
}

}  // namespace xllm::layer::v4_cp

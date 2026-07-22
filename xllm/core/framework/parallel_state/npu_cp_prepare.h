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

#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "util/tensor_helper.h"

namespace xllm {

struct ModelInputParams;

// Worker-local CP prefill plan for the NPU model-side CP closure. Built once
// per forward from the global-real layout (never from a per-rank slice) and
// consumed read-only by the model to localize hidden states after embedding
// and restore global-real order after the last decoder layer. This struct is
// intentionally worker/model-local: it never crosses the RPC/proto/shared
// memory boundary. `ParallelInput` carries a device-ready copy alongside the
// existing `cp_prefill_inputs` / `cp_ep_padding_data` so the decoder ATB graph
// keeps consuming only those two legacy fields.
struct NpuCpPrefillPlan {
  bool enabled = false;
  int32_t cp_size = 1;
  int32_t cp_rank = 0;

  // Sum of per-seq real q_seq_lens.
  int64_t global_real_token_num = 0;
  // Sum of p_i = align_up(q_i, 2 * cp_size) over all seqs.
  int64_t global_padded_token_num = 0;
  // Sum of p_i / cp_size over all seqs; identical across CP ranks (virtual
  // pad).
  int64_t local_padded_token_num = 0;
  // Number of real rows owned by this rank (sum of per-seq zigzag real counts).
  int64_t local_real_token_num = 0;

  // [local_real_token_num] int64: global-real row index of each owned real row.
  torch::Tensor local_source_indices;
  // [local_real_token_num] int64: offset in local-padded hidden to write it.
  torch::Tensor local_destination_indices;
  // [local_padded_token_num] int32: full local-padded positions (real + virtual
  // pad), contiguous per seq from its real start position, matching legacy
  // Batch physical padding byte-for-byte for aligned cases.
  torch::Tensor local_virtual_positions;
  // [global_real_token_num] int64: index into the CP rank-major all-gathered
  // hidden (size global_padded_token_num) that recovers global-real order.
  torch::Tensor restore_indices;

  // Local-padded per-seq host metadata fed to the decoder ATB graph.
  std::vector<int32_t> local_q_seq_lens;
  std::vector<int32_t> local_kv_seq_lens;
  std::vector<int32_t> local_q_cu_seq_lens;
  // Per-seq local-PADDED length (= 2 * chunk_i = p_i / cp_size), constant
  // across CP ranks. Used to build cp_kv_recover_idx at cp_size*local_padded
  // length so it matches the AllGathered KV that ReshapeAndCache caches
  // (critical for non-aligned cases where local_real < local_padded).
  std::vector<int32_t> local_padded_seq_lens;
  int32_t local_q_max_seq_len = 0;
  int32_t local_kv_max_seq_len = 0;

  NpuCpPrefillPlan to(const torch::Device& device) const {
    NpuCpPrefillPlan out = *this;
    out.local_source_indices = safe_to(local_source_indices, device, true);
    out.local_destination_indices =
        safe_to(local_destination_indices, device, true);
    out.local_virtual_positions =
        safe_to(local_virtual_positions, device, true);
    out.restore_indices = safe_to(restore_indices, device, true);
    return out;
  }
};

struct CpPrefillInputs {
  torch::Tensor cp_load_balance_idx;
  torch::Tensor cp_o_recover_idx;
  torch::Tensor cp_kv_recover_idx;

  torch::Tensor k_gather_index_prev;
  torch::Tensor k_gather_index_next;

  torch::Tensor actual_seq_lengths_query_prev;
  torch::Tensor actual_seq_lengths_query_next;
  torch::Tensor actual_seq_lengths_key_prev;
  torch::Tensor actual_seq_lengths_key_next;

  CpPrefillInputs to(const torch::Device& device) const {
    CpPrefillInputs inputs;
    inputs.cp_load_balance_idx = safe_to(cp_load_balance_idx, device, true);
    inputs.cp_o_recover_idx = safe_to(cp_o_recover_idx, device, true);
    inputs.cp_kv_recover_idx = safe_to(cp_kv_recover_idx, device, true);
    inputs.k_gather_index_prev = safe_to(k_gather_index_prev, device, true);
    inputs.k_gather_index_next = safe_to(k_gather_index_next, device, true);
    inputs.actual_seq_lengths_query_prev =
        safe_to(actual_seq_lengths_query_prev, device, true);
    inputs.actual_seq_lengths_query_next =
        safe_to(actual_seq_lengths_query_next, device, true);
    inputs.actual_seq_lengths_key_prev =
        safe_to(actual_seq_lengths_key_prev, device, true);
    inputs.actual_seq_lengths_key_next =
        safe_to(actual_seq_lengths_key_next, device, true);
    return inputs;
  }
};

// Build the worker-local CP prefill plan from the global-real layout. Inputs
// are the GLOBAL (un-sliced, un-padded) per-seq q_seq_lens, the global
// positions (one row per real token), and the per-seq KV cache token counts.
// The plan is rank-specific: `cp_rank` selects the zigzag chunks this rank
// owns. Virtual padding makes `local_padded_token_num` identical across ranks
// so the model can issue an equal-length all-gather after the last decoder
// layer.
NpuCpPrefillPlan build_npu_cp_prefill_plan(
    int cp_size,
    int cp_rank,
    const std::vector<int32_t>& global_q_seq_lens,
    const torch::Tensor& global_positions,
    bool have_prefix_slots,
    const std::vector<int32_t>& kv_cache_tokens_per_seq,
    int block_size,
    int kv_split_size);

// Produce the decoder-facing `CpPrefillInputs` from a plan instead of a
// per-rank token slice. For aligned cases (every q_i divisible by 2*cp_size)
// the output is byte-identical to the legacy `prepare_cp_prefill_inputs`
// called on the post-`cp_partition_inplace` per-rank slice.
CpPrefillInputs prepare_cp_prefill_inputs_from_plan(
    const NpuCpPrefillPlan& plan,
    bool have_prefix_slots,
    const std::vector<int32_t>& kv_cache_tokens_per_seq,
    int block_size,
    int kv_split_size);

// Override the rank-local attention metadata (q/kv seq lens, cu seq lens, max
// seq len) on a `ModelInputParams` so the decoder ATB graph and the attention
// metadata builder see the local-padded layout encoded by `plan`. This is the
// model-side CP counterpart of the metadata portion of worker
// `cp_partition_inplace`: the model keeps the global token stream intact (it
// localizes hidden states after embedding via `npu_cp::localize`) but, before
// running the decoder loop, rewrites the attention metadata to the per-rank
// local view. Device tensors are materialized on `device`.
//
// `plan` must be enabled and built from the same global q_seq_lens the worker
// received. The cumsum / non-cumsum layout of the existing host vectors is
// preserved, matching `cp_partition_inplace`.
void apply_cp_local_metadata_from_plan(ModelInputParams& params,
                                       const NpuCpPrefillPlan& plan,
                                       const torch::Device& device);

}  // namespace xllm

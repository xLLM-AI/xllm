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

#pragma once

#include <torch/torch.h>

#include <cstdint>

#include "util/tensor_helper.h"

namespace xllm {

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

// `cp_size`        : token-CP width. Drives current-segment k_offset (each
//                    seq occupies cp_size * input_lengths[i] slots in the
//                    rearranged kv buffer) and load-balance / kv-recover
//                    index generation.
// `kv_split_size`  : KV-shard width. Drives prefix geometry (per-rank prefix
//                    tokens = total_prefix / kv_split_size) and the
//                    prefix-segment offsets in `merged_kv`. Pass <= 0 (the
//                    default) to mean "follow cp_size" - this preserves the
//                    legacy behavior byte-for-byte.
CpPrefillInputs prepare_cp_prefill_inputs(
    int cp_size,
    const torch::Tensor& input_ids,
    const torch::Tensor& position_ids,
    const torch::Tensor& input_lengths,
    bool have_prefix_slots,
    const std::vector<int32_t>& kv_cache_tokens_per_seq,
    int block_size,
    int kv_split_size = -1);

}  // namespace xllm

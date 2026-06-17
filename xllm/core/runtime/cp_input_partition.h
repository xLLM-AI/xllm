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

#include <cstdint>

#include "runtime/forward_params.h"

namespace xllm::cp {

// Apply Context-Parallel input slicing in place on a deserialized
// ForwardInput.
//
// The function reproduces the per-rank slicing previously done by
// `RawForwardInput::cp_partition` at the engine side, so the engine can
// broadcast the same RawForwardInput to every CP worker and let each worker
// produce its own per-rank view locally.
//
// Trigger conditions (no-op otherwise):
//   - cp_size > 1
//   - !input.input_params.batch_forward_type.is_decode() (prefill / chunked
//     prefill / mixed; pure decode is skipped)
//     prefill batches; decode batches keep the global view)
//   - input.input_params.num_sequences > 0
//   - input.token_ids is defined and non-empty
//
// Fields rewritten:
//   - token_ids, positions (token-level: index_select on dim 0)
//   - input_params.embedding.mtp_shifted_token_ids (token-level; sliced when
//     defined)
//   - input_params.q_seq_lens(_vec), kv_seq_lens(_vec), q_cu_seq_lens
//     (rebuilt from per-seq cp_q_lens; the cumsum / non-cumsum layout is
//     preserved per the existing convention)
//   - input_params.q_max_seq_len, kv_max_seq_len (set to the max chunk pair
//     length across cp ranks, identical to old behavior)
//   - sampling_params.selected_token_idxes (remapped onto the per-rank view
//     using the same algorithm as the old cp_partition)
//
// Fields explicitly NOT touched (parity with old cp_partition):
//   - input_params.new_cache_slots / new_cache_slot_offsets
//   - input_params.block_tables
//   - input_params.kv_cache_tokens_nums(_host)
//   - input_params.dp_global_token_nums / dp_is_decode
//   - transfer_kv_infos
//
// This function is CPU-only (operates on CPU tensors before the
// `to(device)` step in WorkerImpl::prepare_work_before_execute).
void cp_partition_inplace(ForwardInput& input,
                          int32_t cp_rank,
                          int32_t cp_size);

}  // namespace xllm::cp

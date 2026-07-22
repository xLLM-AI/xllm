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

#include "core/framework/parallel_state/npu_cp_prepare.h"
#include "core/framework/parallel_state/parallel_state.h"

namespace xllm {
namespace npu_cp {

// Localize global-real hidden `[global_real, H]` into the rank-local padded
// layout `[local_padded, H]` consumed by the decoder ATB graph, using the
// zigzag scatter encoded in `plan`. Virtual-pad rows stay zero so the decoder
// padding indices (built from the same `local_padded_token_num`) mask them out.
//
// No-op (returns `global_hidden` unchanged) when the plan is disabled, so the
// model forward can call this unconditionally regardless of cp_size.
inline torch::Tensor localize(const torch::Tensor& global_hidden,
                              const NpuCpPrefillPlan& plan) {
  if (!plan.enabled) {
    return global_hidden;
  }
  const int64_t hidden = global_hidden.size(-1);
  torch::Tensor local = torch::zeros({plan.local_padded_token_num, hidden},
                                     global_hidden.options());
  torch::Tensor src =
      global_hidden.index_select(/*dim=*/0, plan.local_source_indices);
  local.index_put_({plan.local_destination_indices}, src);
  return local;
}

// Per-row positions for the localized layout. Returns
// `plan.local_virtual_positions` (`[local_padded]` int32, real rows carry their
// global position and virtual-pad rows carry a contiguous fill that is
// harmless because their hidden is zero) when enabled, otherwise the input
// `global_positions` unchanged. The caller feeds the result straight into the
// positional embedding op.
inline torch::Tensor localize_positions(const NpuCpPrefillPlan& plan,
                                        const torch::Tensor& global_positions) {
  if (!plan.enabled) {
    return global_positions;
  }
  return plan.local_virtual_positions;
}

// Localize global-real `new_cache_slots` `[global_real]` (int32, BlockManager
// logical slot space) into the rank-local padded layout `[local_padded]` that
// matches the decoder hidden row order encoded by `plan`. Real rows are
// scattered via `local_source_indices` -> `local_destination_indices`; virtual
// pad rows stay `-1` (the "skip" sentinel used throughout the KV cache path) so
// `recompute_new_cache_slots` maps them to `-1` and the decoder's
// `attn_padding_idx` masks them out of the cache write. No-op when disabled.
inline torch::Tensor localize_slots(const torch::Tensor& global_slots,
                                    const NpuCpPrefillPlan& plan) {
  if (!plan.enabled) {
    return global_slots;
  }
  TORCH_CHECK(global_slots.dim() == 1,
              "localize_slots expects a 1D global_slots tensor, got dim=",
              global_slots.dim());
  TORCH_CHECK(global_slots.numel() == plan.global_real_token_num,
              "localize_slots: global_slots numel (",
              global_slots.numel(),
              ") must equal plan.global_real_token_num (",
              plan.global_real_token_num,
              ")");
  torch::Tensor local =
      torch::full({plan.local_padded_token_num}, -1, global_slots.options());
  torch::Tensor src =
      global_slots.index_select(/*dim=*/0, plan.local_source_indices);
  local.index_put_({plan.local_destination_indices}, src);
  return local;
}

// Build the slot tensor matching the AllGathered+recovered KV that the ATB
// ReshapeAndCache node caches. The CP prefill graph AllGathers each rank's
// `[local_padded]` local KV (rank-major) and reorders it by `cp_kv_recover_idx`
// BEFORE ReshapeAndCache, so the cached KV has `cp_size * local_padded` rows in
// recovered order. The slots must match that length and order: real tokens
// carry their global slot id, virtual-pad rows carry -1 (ReshapeAndCache
// ignores -1).
//
// Construction (no collective needed: global_slots + plan are replicated across
// CP ranks, so every rank builds the identical tensor):
//   1. allgathered_slots[a] = global_slots[g] iff AllGathered row `a` holds
//      real token g (i.e. a == plan.restore_indices[g]); else -1.
//   2. recovered_slots[r] = allgathered_slots[cp_kv_recover_idx[r]].
// `plan.restore_indices` is [global_real] int64 (rank-major AllGathered offset
// per real token); `cp_kv_recover_idx` is [cp_size*local_padded] int32
// (recovered
// -> AllGathered index, from CpPrefillInputs). Handles non-aligned cases
// (global_real < cp_size*local_padded) correctly: virtual rows stay -1.
inline torch::Tensor localize_slots_recovered(
    const torch::Tensor& global_slots,
    const NpuCpPrefillPlan& plan,
    const torch::Tensor& cp_kv_recover_idx) {
  if (!plan.enabled) {
    return global_slots;
  }
  TORCH_CHECK(
      global_slots.dim() == 1,
      "localize_slots_recovered expects a 1D global_slots tensor, got dim=",
      global_slots.dim());
  const int64_t ag_rows =
      static_cast<int64_t>(plan.cp_size) * plan.local_padded_token_num;
  // The MTP draft model reuses the main model's already-recovered slots
  // (cp_size*local_padded long, in cp_kv_recover_idx order) for its own
  // ReshapeAndCache, so it arrives already in the recovered layout this
  // function produces. Accept that case as a no-op instead of forcing
  // global_real length (which only the main model's pre-localization slots
  // satisfy).
  if (global_slots.numel() == ag_rows) {
    return global_slots;
  }
  TORCH_CHECK(global_slots.numel() == plan.global_real_token_num,
              "localize_slots_recovered: global_slots numel (",
              global_slots.numel(),
              ") must equal plan.global_real_token_num (",
              plan.global_real_token_num,
              ") or cp_size*local_padded (",
              ag_rows,
              ")");
  // Step 1: scatter global slots into rank-major AllGathered layout; virtual
  // rows (not referenced by restore_indices) keep the -1 fill.
  torch::Tensor allgathered =
      torch::full({ag_rows}, -1, global_slots.options());
  allgathered.index_put_({plan.restore_indices.to(torch::kLong)}, global_slots);
  // Step 2: reorder to recovered order via cp_kv_recover_idx (int32 -> int64).
  torch::Tensor recovered =
      allgathered.index_select(/*dim=*/0, cp_kv_recover_idx.to(torch::kLong));
  return recovered;
}

// Restore global-real order from a rank-major gathered buffer
// `[cp_size * local_padded, H]` -> `[global_real, H]` by `index_select` on
// `plan.restore_indices`, which drops virtual-pad rows. No-op when disabled.
inline torch::Tensor restore_from_gathered(const torch::Tensor& gathered,
                                           const NpuCpPrefillPlan& plan) {
  if (!plan.enabled) {
    return gathered;
  }
  return gathered.index_select(/*dim=*/0, plan.restore_indices);
}

// All-gather the local-padded hidden across CP ranks (rank-major, matching
// `parallel_state::gather`'s `allgather_base_sync` + `cat(0)` layout) then
// restore global-real order. This is the model-side CP closure entry point
// called after the last decoder layer.
//
// No-op (returns `local_hidden` unchanged) when the plan is disabled. When the
// plan is enabled the CP group is mandatory and must match the plan's
// cp_size / cp_rank; a missing or mismatched group is a setup bug and is
// rejected with CHECK rather than silently returning the local-padded hidden
// (which would feed un-restored rows to final norm / LmHead).
inline torch::Tensor gather_restore(const torch::Tensor& local_hidden,
                                    const NpuCpPrefillPlan& plan,
                                    ProcessGroup* cp_group) {
  if (!plan.enabled) {
    return local_hidden;
  }
  CHECK(cp_group != nullptr)
      << "NPU model-side CP closure requires a non-null cp_group when the "
      << "plan is enabled (cp_size=" << plan.cp_size << ")";
  CHECK(cp_group->world_size() == plan.cp_size)
      << "NPU model-side CP closure: cp_group world_size ("
      << cp_group->world_size() << ") must match plan.cp_size (" << plan.cp_size
      << ")";
  CHECK(cp_group->rank() == plan.cp_rank)
      << "NPU model-side CP closure: cp_group rank (" << cp_group->rank()
      << ") must match plan.cp_rank (" << plan.cp_rank << ")";
  if (cp_group->world_size() <= 1) {
    return local_hidden;
  }
  torch::Tensor gathered =
      parallel_state::gather(local_hidden, cp_group, /*dim=*/0);
  return restore_from_gathered(gathered, plan);
}

}  // namespace npu_cp
}  // namespace xllm

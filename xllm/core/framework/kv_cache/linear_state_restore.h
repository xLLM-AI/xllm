/* Copyright 2025-2026 The xLLM Authors.

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

#include <cstdint>
#include <vector>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_types.h"

namespace xllm {

// Convert logical sequence KV cursors to the warm/cold state mask consumed by
// active linear-attention rows. Data-parallel expansion keeps each logical
// row contiguous in the execution batch.
LinearStateValidityMask build_linear_state_mask(
    const std::vector<int32_t>& cached_tokens,
    int64_t active_rows);

// Apply each op's linear-state preparation plan across every linear-attention
// layer in `kv_caches`, then record the outcome in `validity_mask`. The caller
// sizes the mask to the active batch and pre-fills it with the KV-cache
// default; this helper only overrides rows that require preparation:
//   - RESTORED: copy the checkpoint source into the live slot and mark warm.
//   - DIRECT_READ: keep the checkpoint in its source slot, skip the copy, and
//     mark warm so the forward reads the source and writes the live slot.
//   - COLD_START: clear the reused live slot and mark cold.
//   - continued request / no reused prefix: leave the default unchanged.
// Slot ownership, source resolution, and save promotion stay scheduler-side;
// this helper only executes the worker-side reset/restore/direct-read plan.
// Device copies and clears run on the current stream (today the worker's
// `prepare_stream_`); the caller inserts the stream-event barrier before model
// forward. Direct-read rows enqueue no state copy, but their checkpoint source
// must remain pinned until the forward finishes.
void restore_linear_state_slots(
    std::vector<KVCache>& kv_caches,
    const std::vector<LinearStateCacheOp>& cache_ops,
    LinearStateValidityMask& validity_mask);

}  // namespace xllm

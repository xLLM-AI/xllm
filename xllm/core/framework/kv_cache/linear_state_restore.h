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

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"

namespace xllm {

// Apply each op's restore plan in-place: copy `restore_src_slot_id` into
// `linear_state_id` across every linear-attention layer present in
// `kv_caches`, then record each op's outcome into `has_initial_state`. The
// caller sizes `has_initial_state` to the active batch and pre-fills it with
// the kv-cache default; this helper only overrides the entries it must:
//   - RESTORED:  a checkpoint was copied into the live slot -> set 1 (warm).
//   - COLD_START: a restore was requested but the checkpoint was unavailable
//     (miss / out-of-range / copy failure) -> set 0, so the forward does not
//     treat reused kv blocks as warm recurrent state.
//   - no restore requested (continued request / no prefix reused) -> left
//     untouched, so warm continued requests keep the kv-cache default.
// Slot ownership and LRU eviction live in the scheduler-side
// LinearStateBlockManager; this helper is purely the worker-side restore copy
// that cache dictates. Saves require no worker-side copy and are handled by
// promotion in the scheduler, so this API only covers restores. Copies are
// enqueued on whichever stream is current at the call site (today this is the
// worker's dedicated `prepare_stream_`, not the default model-forward stream);
// the caller is responsible for inserting a stream-event barrier before the
// model forward consumes the restored slots — see worker_impl.cpp where
// prepare_stream.record_event() is awaited prior to the forward.
void restore_linear_state_slots(
    std::vector<KVCache>& kv_caches,
    const std::vector<LinearStateCacheOp>& cache_ops,
    std::vector<int64_t>& has_initial_state);

}  // namespace xllm

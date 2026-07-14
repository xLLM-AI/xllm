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

#include "framework/kv_cache/linear_state_restore.h"

#include <glog/logging.h>

namespace xllm {

namespace {

// Copy conv + ssm recurrent state between two physical slots across all
// linear-attention layers. Pure-attention layers are skipped. Returns true
// if at least one layer was copied; false means the caller's
// `discover_num_slots > 0` invariant was violated (e.g. a future refactor
// decoupled cache layout from layer presence). Treat false as a soft
// failure at the call site — fall back to cold start, do not crash the
// service.
bool copy_slot_across_layers(std::vector<KVCache>& kv_caches,
                             int32_t dst_slot_id,
                             int32_t src_slot_id) {
  bool copied = false;
  for (const auto& kv_cache : kv_caches) {
    torch::Tensor conv_cache = kv_cache.get_conv_cache();
    torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
    if (!conv_cache.defined() || !ssm_cache.defined()) {
      continue;
    }
    const int64_t checkpoint_stride = ssm_cache.size(0) / conv_cache.size(0);
    conv_cache.select(0, dst_slot_id).copy_(conv_cache.select(0, src_slot_id));
    ssm_cache
        .narrow(0,
                static_cast<int64_t>(dst_slot_id) * checkpoint_stride,
                checkpoint_stride)
        .copy_(ssm_cache.narrow(
            0,
            static_cast<int64_t>(src_slot_id) * checkpoint_stride,
            checkpoint_stride));
    copied = true;
  }
  return copied;
}

// Discover the slot count from any linear-attention layer in `kv_caches`.
// Returns 0 when no linear-attention layer is present, which short-circuits
// restore. CHECKs slot-count consistency across layers.
int32_t discover_num_slots(const std::vector<KVCache>& kv_caches) {
  int32_t num_slots = 0;
  for (const auto& kv_cache : kv_caches) {
    torch::Tensor conv_cache = kv_cache.get_conv_cache();
    torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
    if (!conv_cache.defined() || !ssm_cache.defined()) {
      continue;
    }
    CHECK_GT(conv_cache.size(0), 0) << "conv cache must have positive slots.";
    CHECK_EQ(ssm_cache.size(0) % conv_cache.size(0), 0)
        << "ssm cache checkpoint layout mismatch, ssm_rows="
        << ssm_cache.size(0) << ", conv_rows=" << conv_cache.size(0);
    if (num_slots == 0) {
      num_slots = static_cast<int32_t>(conv_cache.size(0));
      continue;
    }
    CHECK_EQ(num_slots, static_cast<int32_t>(conv_cache.size(0)));
  }
  return num_slots;
}

}  // namespace

void restore_linear_state_slots(
    std::vector<KVCache>& kv_caches,
    const std::vector<LinearStateCacheOp>& cache_ops,
    std::vector<int64_t>& has_initial_state) {
  if (cache_ops.empty()) {
    return;
  }
  const int32_t num_slots = discover_num_slots(kv_caches);
  if (num_slots == 0) {
    return;
  }
  CHECK_EQ(cache_ops.size(), has_initial_state.size())
      << "has_initial_state must be sized to the cache_ops batch before "
      << "restore, cache_ops=" << cache_ops.size()
      << ", has_initial_state=" << has_initial_state.size();
  const auto slot_in_range = [num_slots](int32_t slot_id) {
    return slot_id >= 0 && slot_id < num_slots;
  };

  for (size_t i = 0; i < cache_ops.size(); ++i) {
    const LinearStateCacheOp& cache_op = cache_ops[i];
    const int32_t live_slot_id = cache_op.linear_state_id;
    const int32_t src_slot_id = cache_op.restore_src_slot_id;
    const bool restore_requested = cache_op.restore_requested;

    // Continued requests and cold starts have no checkpoint to restore; leave
    // has_initial_state at its kv-cache default.
    if (!restore_requested && src_slot_id < 0) {
      continue;
    }
    // Restore requested but unservable: scheduler could not resolve a
    // checkpoint, or slot ids are out of range. Force cold start so the
    // forward does not treat reused kv blocks as warm recurrent state.
    if (src_slot_id < 0 || !slot_in_range(live_slot_id) ||
        !slot_in_range(src_slot_id)) {
      has_initial_state[i] = 0;
      continue;
    }
    // Scheduler invariant: a resolved src slot implies a restore request. The
    // batch builder only mounts restore_src_slot_id when it flags a restore
    // (batch_input_builder.cpp).
    DCHECK(restore_requested);

    if (!copy_slot_across_layers(kv_caches, live_slot_id, src_slot_id)) {
      // discover_num_slots > 0 said at least one linear layer was copyable;
      // landing here means that invariant broke. Log every occurrence so the
      // regression is impossible to miss, and fall back to cold start so
      // the request still produces correct (uncached) output.
      LOG(ERROR) << "Linear state restore: discover_num_slots > 0 but no "
                    "layer was copied; falling back to cold start. "
                    "live_slot_id="
                 << live_slot_id << ", src_slot_id=" << src_slot_id;
      has_initial_state[i] = 0;
      continue;
    }
    has_initial_state[i] = 1;
    VLOG(1) << "Qwen3.5 linear state checkpoint restored; live_slot_id="
            << live_slot_id << ", src_slot_id=" << src_slot_id;
  }
  // The copies above are enqueued on whichever stream is current at the call
  // site (today the worker's dedicated prepare stream). The caller must
  // insert a cross-stream event barrier so the model forward — which runs on
  // a different stream — observes them as completed; no host sync is added
  // here.
}

}  // namespace xllm

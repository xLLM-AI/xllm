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

#include <glog/logging.h>

#include <cstdint>
#include <limits>

#include "scheduler/scheduler_policy.h"

namespace xllm {

// =============================================================================
// DecodeFirstPolicy::schedule
// =============================================================================

void DecodeFirstPolicy::schedule(
    SchedulerState& state,
    ScheduleBudget& budget,
    std::vector<std::shared_ptr<Request>>& finished) {
  // === Requeue phase ===
  // ChunkedPrefillScheduler path: continuations go to chunk_queue (front),
  // decode requests go to running_queue.
  for (auto it = state.running_requests.rbegin();
       it != state.running_requests.rend();
       ++it) {
    if (*it == nullptr) {
      continue;
    }
    auto req = *it;
    handle_running_requests(req, state);
    if (req->sequences()[0]->is_chunked_prefill_stage()) {
      state.chunk_queue.push(req, /*if_back=*/false);
    } else {
      state.decode_queue.push(req);
    }
  }
  reset_batch_state(state);

  // === Schedule phase: decode first, then prefill with remaining budget ===
  budget.latency_budget = options_.max_global_tpot_ms();

  // Step 1: schedule decode requests first (decode-maximal batching).
  schedule_decode_from_queue(&state.decode_queue, state, budget);

  // Step 2: schedule prefill requests (continuations from chunk_queue first,
  // then new prefill from waiting_queue).
  if (!budget_exhausted(budget)) {
    if (state.running_sequences.empty()) {
      budget.latency_budget = std::numeric_limits<int32_t>::max();
    }
    schedule_prefill_from_queue(&state.chunk_queue, state, budget, finished);
    schedule_prefill_from_queue(&state.prefill_queue, state, budget, finished);
  }

  // Step 3: redistribute remaining budget to prefill sequences.
  if (budget.remaining_token_budget > 0 &&
      budget.latency_budget > budget.estimate_latency) {
    std::vector<Sequence*> prefill_stage_sequences;
    for (size_t i = 0; i < state.running_sequences.size(); ++i) {
      if (state.running_sequences[i]->is_chunked_prefill_stage() ||
          state.running_sequences[i]->kv_state().kv_cache_tokens_num() == 0) {
        prefill_stage_sequences.emplace_back(state.running_sequences[i]);
      }
    }
    redistribute_remaining_budget(state, budget, prefill_stage_sequences);
  }
}

// =============================================================================
// DecodeFirstPolicy::redistribute_remaining_budget
// =============================================================================

void DecodeFirstPolicy::redistribute_remaining_budget(
    SchedulerState& state,
    ScheduleBudget& budget,
    std::vector<Sequence*>& prefill_seqs) {
  size_t prefill_seq_idx = 0;
  for (size_t i = 0; i < state.running_sequences.size() &&
                     prefill_seq_idx < prefill_seqs.size();
       ++i) {
    if (prefill_seqs[prefill_seq_idx] != state.running_sequences[i]) {
      continue;
    }
    ++prefill_seq_idx;
    Sequence* sequence = state.running_sequences[i];
    size_t& token_budget = state.running_sequences_budgets[i];

    // Return previously allocated tokens to the pool.
    budget.remaining_token_budget += token_budget;

    if (options_.enable_latency_aware_schedule()) {
      double origin_latency =
          state.profile_manager->predict_step_time(sequence, false);
      budget.estimate_latency -= origin_latency;

      double cur_latency = state.profile_manager->predict_step_time(
          budget.remaining_token_budget,
          sequence->kv_state().kv_cache_tokens_num(),
          false);
      if (budget.estimate_latency + cur_latency > budget.latency_budget) {
        break;
      }
      budget.estimate_latency += cur_latency;
    }

    size_t actual_tokens = 0;
    if (!allocate_for_prefill(sequence,
                              budget.remaining_token_budget,
                              &actual_tokens,
                              state,
                              /*skip_shared=*/true)) {
      break;
    }

    token_budget = actual_tokens;
    CHECK(budget.remaining_token_budget >= actual_tokens);
    budget.remaining_token_budget -= actual_tokens;

    if (budget.remaining_token_budget == 0) {
      break;
    }
  }
}

}  // namespace xllm

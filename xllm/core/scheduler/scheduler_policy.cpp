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

#include "scheduler/scheduler_policy.h"

#include <absl/time/clock.h>
#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <limits>

#include "common/metrics.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/kv_cache_store_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/scheduler/request_output_dispatcher.h"
#include "framework/batch/batch_factory.h"
#include "framework/model/model_args.h"
#include "framework/request/priority_comparator.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {

namespace {

inline size_t maybe_align_cp_chunk_tokens(size_t num_tokens,
                                          int32_t kv_split_size,
                                          int32_t logical_block_size,
                                          size_t remaining_in_seq) {
  if (kv_split_size <= 1) {
    return num_tokens;
  }
  if (logical_block_size <= 0 || num_tokens == 0) {
    return num_tokens;
  }
  if (num_tokens >= remaining_in_seq) {
    return num_tokens;
  }
  const size_t kv_term = static_cast<size_t>(logical_block_size);
  if (num_tokens < kv_term) {
    return num_tokens;
  }
  return (num_tokens / kv_term) * kv_term;
}

size_t prefill_target_tokens(Sequence* sequence,
                             size_t token_budget,
                             const SchedulerState& state) {
  const size_t cached_tokens = sequence->kv_cache_tokens_num();
  CHECK_LE(cached_tokens, sequence->num_tokens());
  const size_t target =
      cached_tokens +
      std::min(token_budget, sequence->num_tokens() - cached_tokens);
  if (!has_linear_attention_layers(state.model_args) ||
      !state.enable_prefix_cache || !sequence->is_prefill_stage()) {
    return target;
  }

  // Prefer checkpoint boundaries when the quota reaches one. Smaller chunks
  // remain valid under the linear-state cache's copy-on-write lifecycle.
  const size_t chunk_stride = static_cast<size_t>(
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill());
  CHECK_GT(chunk_stride, 0u);
  const size_t aligned = (target / chunk_stride) * chunk_stride;
  if (aligned > cached_tokens) {
    return aligned;
  }
  return target;
}

// Estimate extra blocks needed for a decode step with speculative tokens.
size_t estimate_decode_extra_blocks(Sequence* sequence,
                                    size_t updated_num_tokens,
                                    size_t block_size) {
  const size_t num_blocks = sequence->kv_state().num_blocks(BlockType::KV);
  const size_t num_blocks_needed =
      (updated_num_tokens + block_size - 1) / block_size;
  if (num_blocks_needed > num_blocks) {
    return num_blocks_needed - num_blocks;
  }
  if (sequence->check_beam_search() &&
      !sequence->kv_state().src_blocks().empty() &&
      sequence->kv_state().need_swap()) {
    return 1;
  }
  return 0;
}

size_t get_sequence_free_blocks_for_rank(KVCacheManager* kv_cache_manager,
                                         int32_t dp_rank) {
  const auto free_blocks = kv_cache_manager->num_free_blocks();
  if (free_blocks.empty()) {
    return 0;
  }
  if (dp_rank >= 0 && static_cast<size_t>(dp_rank) < free_blocks.size()) {
    return free_blocks[dp_rank];
  }
  return util::max(free_blocks);
}

class ScopedDpRankAssignment final {
 public:
  explicit ScopedDpRankAssignment(Sequence& sequence)
      : sequence_(sequence), original_rank_(sequence.dp_rank()) {}

  ~ScopedDpRankAssignment() {
    if (!committed_ && !sequence_.has_any_blocks()) {
      if (sequence_.dp_rank() != original_rank_) {
        sequence_.kv_state().set_prefix_cache_matched(/*matched=*/false);
        sequence_.host_kv_state().set_prefix_cache_matched(/*matched=*/false);
        sequence_.clear_host_cache_match();
      }
      sequence_.set_dp_rank(original_rank_);
    }
  }

  void commit() { committed_ = true; }

 private:
  DISALLOW_COPY_AND_ASSIGN(ScopedDpRankAssignment);
  Sequence& sequence_;
  int32_t original_rank_;
  bool committed_ = false;
};

}  // namespace

// =============================================================================
// Construction
// =============================================================================

SchedulerPolicy::SchedulerPolicy(const BatchMode& mode,
                                 const SchedulerOptions& options)
    : batch_mode_(mode), options_(options) {}

SchedulerPolicy::PrefillPlacement SchedulerPolicy::prepare_prefill_placement(
    Sequence* sequence,
    SchedulerState& state,
    const ScheduleBudget& budget) {
  const bool can_match_prefix =
      state.enable_prefix_cache ||
      state.kv_cache_manager->supports_host_cache_restore();
  const size_t global_tokens =
      compute_prefill_tokens(sequence, budget.remaining_token_budget, state);
  if (!can_match_prefix &&
      (global_tokens == 0 || global_tokens > budget.remaining_token_budget)) {
    return {.status = PrefillPlacementStatus::BUDGET_EXHAUSTED};
  }

  const int32_t assigned_rank = sequence->dp_rank();
  int32_t dp_rank = assigned_rank;
  std::vector<DpRankLoad> loads;
  std::vector<uint8_t> prefix_probe_ranks;
  const int32_t max_prefills =
      state.model_args.max_concurrent_prefills_per_dp();
  if (max_prefills > 0 || !budget.dp_group_token_caps.empty()) {
    loads = state.kv_cache_manager->dp_rank_loads();
    const size_t dp_size = loads.size();
    CHECK(budget.dp_group_token_caps.empty() ||
          budget.dp_group_token_caps.size() == dp_size);
    CHECK_EQ(budget.dp_group_token_used.size(),
             budget.dp_group_token_caps.size());
    if (!can_match_prefix && !batch_mode_.enable_chunked_prefill &&
        !budget.dp_group_token_caps.empty()) {
      // A full prefill that exceeds an unused rank quota cannot make progress
      // by waiting for another round, even if profiling raises the global
      // budget.
      size_t capacity = *std::max_element(budget.dp_group_token_caps.begin(),
                                          budget.dp_group_token_caps.end());
      if (dp_rank >= 0) {
        CHECK_LT(static_cast<size_t>(dp_rank), dp_size);
        capacity = budget.dp_group_token_caps[dp_rank];
      }
      if (global_tokens > capacity) {
        return {.status = PrefillPlacementStatus::BUDGET_EXHAUSTED};
      }
    }

    std::vector<size_t> per_dp_counts;
    if (max_prefills > 0) {
      per_dp_counts.resize(dp_size, /*value=*/0);
      std::vector<bool> counted_dp_ranks(dp_size, /*value=*/false);
      for (const auto& request : state.running_requests) {
        if (request == nullptr) {
          continue;
        }
        std::fill(counted_dp_ranks.begin(),
                  counted_dp_ranks.end(),
                  /*value=*/false);
        for (const auto& running_sequence : request->sequences()) {
          if (!running_sequence || !running_sequence->is_prefill_stage()) {
            continue;
          }
          const int32_t rank = running_sequence->dp_rank();
          if (rank >= 0 && static_cast<size_t>(rank) < dp_size &&
              !counted_dp_ranks[rank]) {
            ++per_dp_counts[rank];
            counted_dp_ranks[rank] = true;
          }
        }
      }
    }

    std::vector<uint8_t> eligible_ranks(dp_size, /*value=*/0);
    if (can_match_prefix) {
      prefix_probe_ranks.resize(dp_size, /*value=*/0);
    }
    for (size_t rank = 0; rank < dp_size; ++rank) {
      if (max_prefills > 0 &&
          per_dp_counts[rank] >= static_cast<size_t>(max_prefills)) {
        continue;
      }
      size_t available_tokens = budget.remaining_token_budget;
      if (!budget.dp_group_token_caps.empty()) {
        CHECK_LE(budget.dp_group_token_used[rank],
                 budget.dp_group_token_caps[rank]);
        available_tokens = std::min(available_tokens,
                                    budget.dp_group_token_caps[rank] -
                                        budget.dp_group_token_used[rank]);
      }
      if (can_match_prefix && available_tokens > 0) {
        prefix_probe_ranks[rank] = 1;
      }
      const size_t num_tokens =
          compute_prefill_tokens(sequence, available_tokens, state);
      if (num_tokens == 0 || num_tokens > available_tokens) {
        continue;
      }
      eligible_ranks[rank] = 1;
    }
    DpPlacementConstraints constraints{
        .assigned_rank = dp_rank,
        .eligible_ranks = eligible_ranks,
    };
    dp_rank = state.dp_rank_selector.select(loads, constraints);
    if (dp_rank < 0 && can_match_prefix) {
      // Matching a prefix can make a full prefill or final short chunk fit.
      // Probe only ranks that still have a request slot and some token quota.
      constraints.eligible_ranks = prefix_probe_ranks;
      dp_rank = state.dp_rank_selector.select(loads, constraints);
    }
    if (dp_rank < 0) {
      return {.status = PrefillPlacementStatus::DEFERRED};
    }
  } else if (dp_rank < 0) {
    loads = state.kv_cache_manager->dp_rank_loads();
    dp_rank = state.dp_rank_selector.select(loads);
    CHECK_GE(dp_rank, 0);
  }

  PrefillPlacementStatus blocked_status =
      PrefillPlacementStatus::BUDGET_EXHAUSTED;
  while (true) {
    sequence->set_dp_rank(dp_rank);
    allocate_shared_blocks_for(sequence, state);
    size_t available_tokens = budget.remaining_token_budget;
    if (!budget.dp_group_token_caps.empty()) {
      available_tokens = std::min(available_tokens,
                                  budget.dp_group_token_caps[dp_rank] -
                                      budget.dp_group_token_used[dp_rank]);
    }
    const size_t num_tokens =
        compute_prefill_tokens(sequence, available_tokens, state);
    if (num_tokens > 0 && num_tokens <= available_tokens) {
      return {PrefillPlacementStatus::READY, dp_rank, num_tokens};
    }
    const size_t global_needed =
        compute_prefill_tokens(sequence, budget.remaining_token_budget, state);
    if (global_needed > 0 && global_needed <= budget.remaining_token_budget &&
        (batch_mode_.enable_chunked_prefill ||
         budget.dp_group_token_caps.empty() ||
         global_needed <= budget.dp_group_token_caps[dp_rank])) {
      // Any rank that can fit next round keeps the request eligible to wait.
      blocked_status = PrefillPlacementStatus::DEFERRED;
    }
    if (assigned_rank >= 0 || !can_match_prefix) {
      return {.status = blocked_status};
    }

    // These are temporary prefix aliases, not an admitted request's cache.
    // Release them and their probe state before considering another rank.
    if (prefix_probe_ranks.empty()) {
      prefix_probe_ranks.resize(loads.size(), /*value=*/1);
    }
    prefix_probe_ranks[dp_rank] = 0;
    state.kv_cache_manager->deallocate(sequence);
    sequence->set_dp_rank(assigned_rank);
    dp_rank = state.dp_rank_selector.select(
        loads, {.eligible_ranks = prefix_probe_ranks});
    if (dp_rank < 0) {
      return {.status = blocked_status};
    }
  }
}

void SchedulerPolicy::adjust_latency_budget_and_reorder(
    RequestPriorityQueue* /*first_queue*/,
    RequestPriorityQueue* /*second_queue*/,
    double& /*latency_budget*/,
    bool /*for_prefill*/,
    const SchedulerState& /*state*/) {}

// =============================================================================
// Common phases
// =============================================================================

void SchedulerPolicy::drain_request_queue(
    SchedulerState& state,
    folly::MPMCQueue<std::shared_ptr<Request>>& request_queue) {
  std::vector<std::shared_ptr<Request>> incoming;
  std::shared_ptr<Request> request;
  while (request_queue.read(request)) {
    CHECK(request);
    incoming.emplace_back(std::move(request));
  }
  if (incoming.empty()) {
    return;
  }
  // MPMC drain order is not arrival order across producer threads. Restore
  // FCFS before the waiting queues see the batch.
  if (batch_mode_.priority_strategy == "fcfs") {
    std::stable_sort(incoming.begin(),
                     incoming.end(),
                     create_comparator("fcfs", /*is_reversed=*/true));
  }

  for (std::shared_ptr<Request>& queued : incoming) {
    if (queued->finished() || queued->cancelled()) {
      // collect_finished() owns terminal cleanup.
      state.running_requests.emplace_back(std::move(queued));
      continue;
    }
    if (!state.enable_prefix_cache &&
        !(state.options.enable_disagg_pd() &&
          state.options.instance_role().has_value() &&
          state.options.instance_role().value() != InstanceRole::DECODE)) {
      queued->expand_sequences(/*force=*/false);
    }

    if (queued->sequences()[0]->kv_state().kv_cache_tokens_num() == 0) {
      // New request goes to waiting queue (back = FIFO from MPMC).
      state.prefill_queue.push(std::move(queued), /*if_back=*/true);
      continue;
    }
    // Request from prefill instance in disagg PD mode -- already has KV.
    if (batch_mode_.priority_strategy == "fcfs") {
      // Staged until the requeue phase ends, so a new handoff stays behind
      // older waiters; the rest of the loop has no ordering guarantee.
      pending_decode_handoffs_.emplace_back(std::move(queued));
    } else {
      state.running_requests.emplace_back(std::move(queued));
    }
  }
}

bool SchedulerPolicy::collect_terminal(
    std::shared_ptr<Request>& request,
    SchedulerState& state,
    std::vector<std::shared_ptr<Request>>& finished) {
  if (request == nullptr) {
    return false;
  }
  request->update_connection_status();
  if (!request->finished() && !request->cancelled()) {
    return false;
  }
  if (request->cancelled() && state.release_failed_request) {
    state.release_failed_request(request);
  }
  clear_mtp_bootstrap(request.get(), state);
  state.kv_cache_manager->deallocate(request.get());
  finished.emplace_back(std::move(request));
  return true;
}

std::vector<std::shared_ptr<Request>> SchedulerPolicy::collect_finished(
    SchedulerState& state) {
  std::vector<std::shared_ptr<Request>> finished_requests;
  for (auto it = state.running_requests.rbegin();
       it != state.running_requests.rend();
       ++it) {
    collect_terminal(*it, state, finished_requests);
  }
  for (auto it = pending_decode_handoffs_.rbegin();
       it != pending_decode_handoffs_.rend();
       ++it) {
    collect_terminal(*it, state, finished_requests);
  }

  // FCFS decode waiters can terminate before any budget admits them; rotate
  // once, bounded by the current waiter count, and re-append live waiters to
  // keep their order. Other strategies lack order-preserving back insertion.
  if (batch_mode_.priority_strategy != "fcfs") {
    return finished_requests;
  }
  bool has_terminal_waiter = false;
  for (const auto& request : state.decode_queue) {
    request->update_connection_status();
    if (request->finished() || request->cancelled()) {
      has_terminal_waiter = true;
      break;
    }
  }
  if (!has_terminal_waiter) {
    return finished_requests;
  }
  const size_t num_decode_waiters = state.decode_queue.size();
  for (size_t i = 0; i < num_decode_waiters; ++i) {
    std::shared_ptr<Request> request = state.decode_queue.top();
    state.decode_queue.pop_top();
    if (request == nullptr ||
        collect_terminal(request, state, finished_requests)) {
      continue;
    }
    state.decode_queue.push(std::move(request), /*if_back=*/true);
  }
  return finished_requests;
}

void SchedulerPolicy::reset_batch_state(SchedulerState& state) {
  state.last_step_prefill = false;
  state.running_requests.clear();
  state.running_sequences.clear();
  state.running_sequences_budgets.clear();
  // New FCFS handoffs join the decode queue behind requeued requests.
  for (std::shared_ptr<Request>& request : pending_decode_handoffs_) {
    if (request == nullptr) {
      continue;
    }
    handle_running_requests(request, state);
    state.decode_queue.push(std::move(request), /*if_back=*/true);
  }
  pending_decode_handoffs_.clear();
}

// =============================================================================
// Requeue helper
// =============================================================================
// Prefill scheduling
// =============================================================================

void SchedulerPolicy::schedule_prefill_from_queue(
    RequestPriorityQueue* queue,
    SchedulerState& state,
    ScheduleBudget& budget,
    std::vector<std::shared_ptr<Request>>& finished,
    std::vector<size_t>& reserved_full_footprint) {
  if (queue == nullptr || queue->empty()) {
    return;
  }

  bool budget_exhausted = false;
  bool blocks_exhausted = false;
  // Reuse scratch capacity across requests within this scheduling round.
  std::vector<Sequence*> prefill_sequences;
  std::vector<size_t> prefill_sequences_budget;

  while (!queue->empty() && budget.remaining_seq_budget > 0 &&
         budget.remaining_token_budget > 0 &&
         budget.latency_budget > budget.estimate_latency) {
    // Memory-utilization short-circuit.
    if (!options_.enable_disagg_pd() &&
        state.kv_cache_manager->kv_cache_utilization() >=
            SchedulerConfig::get_instance()
                .prefill_scheduling_memory_usage_threshold()) {
      blocks_exhausted = true;
      break;
    }

    std::shared_ptr<Request> request(queue->top());
    if (collect_terminal(request, state, finished)) {
      queue->pop_top();
      continue;
    }

    const size_t num_sequences = request->sequences().size();
    if (!request->preempted()) {
      CHECK(num_sequences == 1 || num_sequences == request->best_of())
          << "Waiting request should have either 1 or best_of("
          << request->best_of() << ") sequences, got " << num_sequences;
    }

    // Commit all budget charges with the request. An unsuccessful attempt may
    // retain owned cache blocks, but consumes no compute budget this round.
    ScheduleBudget request_budget = budget;
    bool can_schedule = true;
    prefill_sequences.clear();
    prefill_sequences_budget.clear();
    prefill_sequences.reserve(request->sequences().size());
    prefill_sequences_budget.reserve(request->sequences().size());

    for (auto& prefill_sequence : request->sequences()) {
      if (prefill_sequence->finished()) {
        continue;
      }

      if (request_budget.remaining_seq_budget == 0) {
        can_schedule = false;
        budget_exhausted = true;
        break;
      }
      const bool fresh_prefill = prefill_sequence->kv_cache_tokens_num() == 0;
      ScopedDpRankAssignment assignment(*prefill_sequence);
      const PrefillPlacement placement = prepare_prefill_placement(
          prefill_sequence.get(), state, request_budget);
      if (placement.status == PrefillPlacementStatus::DEFERRED) {
        return;
      }
      if (placement.status == PrefillPlacementStatus::BUDGET_EXHAUSTED) {
        can_schedule = false;
        budget_exhausted = true;
        break;
      }
      const int32_t dp_rank = placement.dp_rank;
      const size_t num_tokens = placement.num_tokens;

      bool reserve_full_prompt = false;
      if (batch_mode_.enable_chunked_prefill && fresh_prefill) {
        const size_t block_size =
            static_cast<size_t>(state.kv_cache_manager->block_size());
        const size_t full_footprint =
            (prefill_sequence->num_tokens() + block_size - 1) / block_size;
        const size_t total_blocks =
            state.kv_cache_manager->admission_capacity(dp_rank);
        if (full_footprint > state.kv_cache_manager->num_blocks()) {
          can_schedule = false;
          blocks_exhausted = true;
          break;
        }
        if (reserved_full_footprint[dp_rank] + full_footprint > total_blocks) {
          if (!state.running_sequences.empty() || !state.decode_queue.empty()) {
            can_schedule = false;
            blocks_exhausted = true;
            break;
          }
          // An idle elastic cache's advisory capacity can lag a release. Let
          // allocation validate the full prompt before rejecting this request.
          reserve_full_prompt = true;
        }
      }

      // Latency check.
      double seq_estimate_latency = 0;
      if (options_.enable_latency_aware_schedule()) {
        size_t kv_cache_tokens_num =
            prefill_sequence->kv_state().kv_cache_tokens_num();
        seq_estimate_latency = state.profile_manager->predict_step_time(
            num_tokens + kv_cache_tokens_num, kv_cache_tokens_num, false);
        if (request_budget.estimate_latency + seq_estimate_latency >
            request_budget.latency_budget) {
          can_schedule = false;
          budget_exhausted = true;
          break;
        }
      }

      size_t actual_tokens = 0;
      // allocate_for_prefill internally handles cleanup on failure:
      // - Fresh requests (no prior blocks): prefix cache blocks are released
      //   and sequence is reset inside kv_cache_manager->allocate().
      // - In-flight chunked requests: existing blocks are preserved.
      // No external deallocate needed here.
      if (!allocate_for_prefill(prefill_sequence.get(),
                                num_tokens,
                                &actual_tokens,
                                state,
                                /*skip_shared=*/true,
                                reserve_full_prompt)) {
        can_schedule = false;
        blocks_exhausted = true;
        break;
      }

      CHECK_GT(actual_tokens, 0u);
      CHECK_LE(actual_tokens, num_tokens);
      assignment.commit();
      prefill_sequences_budget.emplace_back(actual_tokens);
      prefill_sequences.emplace_back(prefill_sequence.get());
      request_budget.remaining_token_budget -= actual_tokens;
      --request_budget.remaining_seq_budget;
      request_budget.estimate_latency += seq_estimate_latency;
      if (!request_budget.dp_group_token_used.empty()) {
        request_budget.dp_group_token_used[dp_rank] += actual_tokens;
      }
    }

    if (!can_schedule) {
      break;
    }

    if (prefill_sequences.empty()) {
      return;
    }
    budget = std::move(request_budget);
    queue->pop_top();
    state.running_requests.emplace_back(request);
    request->record_num_prefix_cache_tokens();
    state.running_sequences.insert(state.running_sequences.end(),
                                   prefill_sequences.begin(),
                                   prefill_sequences.end());
    state.running_sequences_budgets.insert(
        state.running_sequences_budgets.end(),
        prefill_sequences_budget.begin(),
        prefill_sequences_budget.end());
    cache_in_batch_prefix(prefill_sequences, prefill_sequences_budget, state);

    // Update reserved footprint for the admission gate.
    if (batch_mode_.enable_chunked_prefill) {
      const size_t block_size =
          static_cast<size_t>(state.kv_cache_manager->block_size());
      for (auto* seq : prefill_sequences) {
        const int32_t dp_rank = seq->dp_rank();
        CHECK(dp_rank >= 0 &&
              dp_rank < static_cast<int32_t>(reserved_full_footprint.size()))
            << "seq dp_rank=" << dp_rank << " out of range [0,"
            << reserved_full_footprint.size() << ")";
        reserved_full_footprint[dp_rank] +=
            (seq->num_tokens() + block_size - 1) / block_size;
      }
    }
  }

  // Handle unschedulable head request.
  handle_unschedulable_head(
      queue, state, finished, budget_exhausted, blocks_exhausted);
}

size_t SchedulerPolicy::compute_prefill_tokens(
    Sequence* seq,
    size_t remaining_budget,
    const SchedulerState& state) const {
  if (!batch_mode_.enable_chunked_prefill) {
    // Full prefill: compute all remaining tokens.
    return seq->num_need_compute_tokens();
  }

  // Chunked prefill: compute min(remaining, max_chunk, budget).
  const size_t max_tokens_per_chunk =
      options_.max_tokens_per_chunk_for_prefill();
  size_t num_tokens = seq->num_tokens();
  size_t assume_max = std::min(max_tokens_per_chunk, remaining_budget);
  num_tokens = std::min(assume_max, num_tokens);

  // CP-aware chunk alignment.
  const size_t kv_cache_tokens_num = seq->kv_cache_tokens_num();
  const size_t remaining_in_seq = seq->num_tokens() > kv_cache_tokens_num
                                      ? seq->num_tokens() - kv_cache_tokens_num
                                      : 0;
  const int32_t kv_split_for_align =
      ::xllm::ParallelConfig::get_instance().kv_split_size_effective();
  num_tokens = maybe_align_cp_chunk_tokens(num_tokens,
                                           kv_split_for_align,
                                           state.kv_cache_manager->block_size(),
                                           remaining_in_seq);

  return prefill_target_tokens(seq, num_tokens, state) - kv_cache_tokens_num;
}

bool SchedulerPolicy::allocate_for_prefill(Sequence* seq,
                                           size_t token_budget,
                                           size_t* actual_tokens,
                                           SchedulerState& state,
                                           bool skip_shared,
                                           bool reserve_full_prompt) {
  if (!batch_mode_.enable_chunked_prefill) {
    // Full prefill: match prefix cache first.
    if (!skip_shared) {
      allocate_shared_blocks_for(seq, state);
    }
    // Full prefill: allocate for full prompt.
    *actual_tokens = seq->num_need_compute_tokens();
    const size_t target = seq->kv_cache_tokens_num() + *actual_tokens;
    return state.kv_cache_manager->allocate(seq, target);
  }

  // Chunked prefill: allocate shared blocks first (skip during redistribution
  // to avoid matching a sequence's own in-batch-published blocks).
  if (!skip_shared) {
    allocate_shared_blocks_for(seq, state);
  }

  const size_t kv_cache_tokens_num = seq->kv_cache_tokens_num();
  const size_t max_handle_num_tokens =
      prefill_target_tokens(seq, token_budget, state);
  if (max_handle_num_tokens <= kv_cache_tokens_num) {
    *actual_tokens = 0;
    return false;
  }

  CHECK_GT(max_handle_num_tokens, kv_cache_tokens_num);
  *actual_tokens = max_handle_num_tokens - kv_cache_tokens_num;
  const size_t allocation_target =
      reserve_full_prompt ? seq->num_tokens() : max_handle_num_tokens;
  return state.kv_cache_manager->allocate(seq, allocation_target);
}

void SchedulerPolicy::allocate_shared_blocks_for(Sequence* seq,
                                                 SchedulerState& state) {
  if (seq->dp_rank() < 0) {
    seq->set_dp_rank(
        state.dp_rank_selector.select(state.kv_cache_manager->dp_rank_loads()));
    CHECK_GE(seq->dp_rank(), 0);
  }
  if (!seq->kv_state().has_any_blocks()) {
    state.kv_cache_manager->allocate_shared(seq);
    return;
  }
  // Refresh the shared match before sizing the next chunk when required by
  // the cache manager.
  if (state.kv_cache_manager->needs_shared_reprobe(seq)) {
    state.kv_cache_manager->allocate_shared(seq);
    return;
  }
  if (seq->is_chunked_prefill_stage()) {
    if (has_linear_attention_layers(state.model_args) &&
        state.enable_prefix_cache) {
      // Linear-state prefix cache can only resume at saved state checkpoints.
      // Re-match at every chunk boundary.
      state.kv_cache_manager->allocate_shared(seq);
      return;
    }
    const size_t max_tokens_per_chunk =
        std::max(options_.max_tokens_per_chunk_for_prefill(), 64);
    size_t total_chunked_size =
        (seq->num_tokens() + max_tokens_per_chunk - 1) / max_tokens_per_chunk;
    if (total_chunked_size <
        ::xllm::SchedulerConfig::get_instance().chunked_match_frequency()) {
      state.kv_cache_manager->allocate_shared(seq);
      return;
    }
    size_t prefix_cache_interval =
        (total_chunked_size +
         ::xllm::SchedulerConfig::get_instance().chunked_match_frequency() -
         1) /
        ::xllm::SchedulerConfig::get_instance().chunked_match_frequency();
    size_t cur_chunked_index =
        seq->kv_state().kv_cache_tokens_num() / max_tokens_per_chunk;
    if (cur_chunked_index % prefix_cache_interval == 0) {
      state.kv_cache_manager->allocate_shared(seq);
    }
  }
}

void SchedulerPolicy::schedule_decode_restore(SchedulerState& state,
                                              ScheduleBudget& budget) {
  if (state.decode_restore_waiting.empty() ||
      state.kv_cache_manager->has_pending_async_block_release() ||
      budget_exhausted(budget)) {
    return;
  }

  DecodeRestoreEntry& entry = state.decode_restore_waiting.front();
  const std::shared_ptr<Request>& request = entry.request;
  CHECK(request != nullptr);
  CHECK_EQ(request->sequences().size(), 1u);
  Sequence* sequence = request->sequences().front().get();
  CHECK(sequence != nullptr);

  const size_t num_tokens =
      compute_prefill_tokens(sequence, budget.remaining_token_budget, state);
  if (num_tokens == 0 || num_tokens > budget.remaining_token_budget ||
      budget.remaining_seq_budget == 0) {
    return;
  }

  size_t actual_tokens = 0;
  if (!allocate_for_prefill(sequence, num_tokens, &actual_tokens, state)) {
    return;
  }

  CHECK_LE(actual_tokens, budget.remaining_token_budget);
  state.running_requests.emplace_back(request);
  state.running_sequences.emplace_back(sequence);
  state.running_sequences_budgets.emplace_back(actual_tokens);
  budget.remaining_token_budget -= actual_tokens;
  --budget.remaining_seq_budget;
  cache_in_batch_prefix({sequence}, {actual_tokens}, state);
  state.decode_restore_waiting.pop_front();
}

// =============================================================================
// Decode scheduling
// =============================================================================

void SchedulerPolicy::schedule_decode_from_queue(RequestPriorityQueue* queue,
                                                 SchedulerState& state,
                                                 ScheduleBudget& budget) {
  if (queue == nullptr || queue->empty()) {
    return;
  }

  // Keep scratch storage local to the call, but avoid allocating per request.
  std::vector<Sequence*> candidate_sequences;
  std::vector<size_t> candidate_token_budgets;
  std::vector<Sequence*> active_sequences;
  while (!queue->empty() &&
         budget.remaining_token_budget >
             static_cast<size_t>(state.min_speculative_tokens_required) &&
         budget.latency_budget > budget.estimate_latency &&
         budget.remaining_seq_budget > 0) {
    std::shared_ptr<Request> request = queue->top();

    const size_t num_sequences = request->sequences().size();
    candidate_sequences.clear();
    candidate_token_budgets.clear();
    candidate_sequences.reserve(num_sequences);
    candidate_token_budgets.reserve(num_sequences);

    bool has_enough_budget = true;
    bool has_enough_blocks = true;
    size_t allocated_tokens = 0;
    size_t allocated_seqs = 0;
    double allocated_estimate_latency = 0;

    if (request->check_beam_search()) {
      // Beam search path.
      active_sequences.clear();
      active_sequences.reserve(num_sequences);
      for (auto& seq : request->sequences()) {
        if (!seq->finished()) {
          active_sequences.emplace_back(seq.get());
        }
      }
      if (active_sequences.empty()) {
        queue->pop_top();
        continue;
      }

      const size_t decode_step_tokens =
          state.min_speculative_tokens_required + 1;
      if (decode_step_tokens * active_sequences.size() >
              budget.remaining_token_budget ||
          active_sequences.size() > budget.remaining_seq_budget) {
        has_enough_budget = false;
      }

      if (has_enough_budget && options_.enable_latency_aware_schedule() &&
          !(options_.instance_role().has_value() &&
            options_.instance_role().value() == InstanceRole::PREFILL)) {
        for (auto* sequence : active_sequences) {
          const double seq_latency =
              state.profile_manager->predict_step_time(sequence, false);
          if (budget.estimate_latency + allocated_estimate_latency +
                  seq_latency >
              budget.latency_budget) {
            has_enough_budget = false;
            break;
          }
          allocated_estimate_latency += seq_latency;
        }
      }
      allocated_estimate_latency = 0.0;

      if (has_enough_budget) {
        const size_t block_size = state.kv_cache_manager->block_size();
        size_t needed_blocks = 0;
        for (auto* sequence : active_sequences) {
          const size_t updated_num_tokens =
              sequence->num_tokens() + state.min_speculative_tokens_required;
          needed_blocks += estimate_decode_extra_blocks(
              sequence, updated_num_tokens, block_size);
        }
        const int32_t dp_rank = active_sequences.front()->dp_rank();
        const size_t free_blocks =
            get_sequence_free_blocks_for_rank(state.kv_cache_manager, dp_rank);
        if (needed_blocks > free_blocks) {
          has_enough_blocks = false;
        }
      }

      if (has_enough_budget && has_enough_blocks) {
        bool allocate_failed = false;
        for (auto* sequence : active_sequences) {
          const size_t updated_num_tokens =
              sequence->num_tokens() + state.min_speculative_tokens_required;
          if (!state.kv_cache_manager->allocate(sequence, updated_num_tokens)) {
            allocate_failed = true;
            break;
          }
          if (sequence->if_cache_block_for_prefill()) {
            state.kv_cache_manager->cache(sequence);
          }
          candidate_sequences.emplace_back(sequence);
          candidate_token_budgets.emplace_back(decode_step_tokens);
          allocated_tokens += decode_step_tokens;
          allocated_seqs += 1;
        }

        if (allocate_failed) {
          LOG(ERROR) << "Beam strict scheduling allocation failed. "
                     << "request_id=" << request->request_id()
                     << ", beam=" << request->check_beam_search();
          state.kv_cache_manager->deallocate(request.get());
          queue->pop_top();
          request->set_preempted();
          state.prefill_queue.push(request);
          continue;
        }

        if (options_.enable_latency_aware_schedule() &&
            !(options_.instance_role().has_value() &&
              options_.instance_role().value() == InstanceRole::PREFILL)) {
          for (auto* sequence : candidate_sequences) {
            allocated_estimate_latency +=
                state.profile_manager->predict_step_time(sequence, false);
          }
        }
      }
    } else {
      // Non-beam-search decode path.
      for (auto& sequence : request->sequences()) {
        if (sequence->finished()) {
          continue;
        }
        const bool is_prefill = sequence->is_prefill_stage();
        const size_t remaining_tokens =
            budget.remaining_token_budget - allocated_tokens;
        size_t num_tokens = state.min_speculative_tokens_required + 1;
        if (is_prefill) {
          allocate_shared_blocks_for(sequence.get(), state);
          num_tokens = std::min(
              compute_prefill_tokens(sequence.get(), remaining_tokens, state),
              sequence->num_need_compute_tokens());
        }
        if (num_tokens == 0 || num_tokens > remaining_tokens ||
            allocated_seqs >= budget.remaining_seq_budget) {
          has_enough_budget = false;
          break;
        }
        double seq_estimate_latency = 0;
        if (options_.enable_latency_aware_schedule() &&
            !(options_.instance_role().has_value() &&
              options_.instance_role().value() == InstanceRole::PREFILL)) {
          seq_estimate_latency =
              is_prefill ? state.profile_manager->predict_step_time(
                               sequence->kv_cache_tokens_num() + num_tokens,
                               sequence->kv_cache_tokens_num(),
                               false)
                         : state.profile_manager->predict_step_time(
                               sequence.get(), false);
          if (budget.estimate_latency + allocated_estimate_latency +
                  seq_estimate_latency >
              budget.latency_budget) {
            has_enough_budget = false;
            break;
          }
        }
        const bool allocated =
            is_prefill ? allocate_for_prefill(sequence.get(),
                                              num_tokens,
                                              &num_tokens,
                                              state,
                                              /*skip_shared=*/true)
                       : state.kv_cache_manager->allocate(
                             sequence.get(),
                             sequence->num_tokens() +
                                 state.min_speculative_tokens_required);
        if (!allocated) {
          has_enough_blocks = false;
          break;
        }
        if (sequence->if_cache_block_for_prefill()) {
          state.kv_cache_manager->cache(sequence.get());
        }
        allocated_tokens += num_tokens;
        allocated_seqs += 1;
        allocated_estimate_latency += seq_estimate_latency;
        candidate_sequences.emplace_back(sequence.get());
        candidate_token_budgets.emplace_back(num_tokens);
      }
    }

    if (has_enough_budget && has_enough_blocks) {
      queue->pop_top();
      state.running_requests.emplace_back(request);
      state.running_sequences.insert(state.running_sequences.end(),
                                     candidate_sequences.begin(),
                                     candidate_sequences.end());
      state.running_sequences_budgets.insert(
          state.running_sequences_budgets.end(),
          candidate_token_budgets.begin(),
          candidate_token_budgets.end());
      budget.remaining_token_budget -= allocated_tokens;
      budget.remaining_seq_budget -= allocated_seqs;
      budget.estimate_latency += allocated_estimate_latency;
      continue;
    }

    // Budget exhausted: partial schedule or fail.
    if (!has_enough_budget) {
      handle_abnormal_decode_request(queue,
                                     state,
                                     budget,
                                     request,
                                     candidate_sequences,
                                     candidate_token_budgets,
                                     allocated_tokens,
                                     allocated_seqs,
                                     allocated_estimate_latency,
                                     /*budget_exhausted=*/true);
      break;
    }

    // Blocks exhausted: wait for an in-flight async release before selecting
    // another victim. The released blocks remain unavailable until the
    // transfer completes.
    if (state.kv_cache_manager->has_pending_async_block_release()) {
      return;
    }

    // First try preempting from chunk_queue (has KV blocks to free), then
    // from the decode queue's lowest priority.
    if (!has_enough_blocks && !state.chunk_queue.empty()) {
      std::shared_ptr<Request> request_to_preempt = state.chunk_queue.back();
      ++budget.num_preempted_requests;
      state.kv_cache_manager->deallocate(request_to_preempt.get());
      state.chunk_queue.pop_back();
      request_to_preempt->set_preempted();
      state.prefill_queue.push(request_to_preempt);
      if (state.kv_cache_manager->has_pending_async_block_release()) {
        return;
      }
      continue;
    }
    if (!has_enough_blocks && queue->size() > 1) {
      std::shared_ptr<Request> request_to_preempt = queue->back();
      if (request_to_preempt.get() != request.get()) {
        ++budget.num_preempted_requests;
        enqueue_decode_restore(request_to_preempt, state);
        queue->pop_back();
        if (state.kv_cache_manager->has_pending_async_block_release()) {
          return;
        }
        continue;
      }
    }

    // Cannot preempt or single request left.
    handle_abnormal_decode_request(queue,
                                   state,
                                   budget,
                                   request,
                                   candidate_sequences,
                                   candidate_token_budgets,
                                   allocated_tokens,
                                   allocated_seqs,
                                   allocated_estimate_latency,
                                   /*budget_exhausted=*/false);
    break;
  }
}

bool SchedulerPolicy::should_wait_for_decode_restore(
    const std::shared_ptr<Request>& request,
    const SchedulerState& state) const {
  if (!state.options.enable_disagg_pd() ||
      !state.options.instance_role().has_value() ||
      state.options.instance_role().value() != InstanceRole::DECODE ||
      !state.enable_prefix_cache || state.options.enable_schedule_overlap() ||
      !state.kv_cache_manager->supports_host_cache_restore() ||
      ::xllm::KVCacheStoreConfig::get_instance().host_blocks_factor() <= 1.0 ||
      request == nullptr || request->sequences().size() != 1 ||
      request->check_beam_search()) {
    return false;
  }

  return state.options.num_speculative_tokens() == 0 ||
         SpeculativeConfig::is_mtp_algorithm(
             SpeculativeConfig::get_instance().speculative_algorithm());
}

void SchedulerPolicy::enqueue_decode_restore(
    const std::shared_ptr<Request>& request,
    SchedulerState& state) {
  const bool should_wait = should_wait_for_decode_restore(request, state);
  if (should_wait) {
    clear_mtp_bootstrap(request.get(), state);
  }
  state.kv_cache_manager->deallocate(request.get());
  request->set_preempted();
  if (should_wait) {
    state.decode_restore_waiting.emplace_back(
        DecodeRestoreEntry{request, absl::Now()});
    return;
  }
  state.prefill_queue.push(request);
}

// =============================================================================
// Preemption and error handling
// =============================================================================

void SchedulerPolicy::handle_abnormal_decode_request(
    RequestPriorityQueue* queue,
    SchedulerState& state,
    ScheduleBudget& budget,
    std::shared_ptr<Request>& request,
    const std::vector<Sequence*>& candidate_sequences,
    const std::vector<size_t>& candidate_token_budgets,
    size_t allocated_tokens,
    size_t allocated_seqs,
    double allocated_estimate_latency,
    bool budget_exhausted) {
  if (!candidate_sequences.empty() && !request->check_beam_search()) {
    // Partial schedule: commit the sequences we already allocated.
    queue->pop_top();
    state.running_requests.emplace_back(request);
    state.running_sequences.insert(state.running_sequences.end(),
                                   candidate_sequences.begin(),
                                   candidate_sequences.end());
    state.running_sequences_budgets.insert(
        state.running_sequences_budgets.end(),
        candidate_token_budgets.begin(),
        candidate_token_budgets.end());
    budget.remaining_token_budget -= allocated_tokens;
    budget.remaining_seq_budget -= allocated_seqs;
    budget.estimate_latency += allocated_estimate_latency;
  } else if (candidate_sequences.empty() && state.running_sequences.empty()) {
    // Nothing scheduled at all -- request is unschedulable.
    queue->pop_top();
    clear_mtp_bootstrap(request.get(), state);
    state.kv_cache_manager->deallocate(request.get());
    if (budget_exhausted) {
      state.output_dispatcher->process_failed_request(
          request,
          {StatusCode::RESOURCE_EXHAUSTED,
           "No enough budget to schedule a single sequence."});
    } else {
      state.output_dispatcher->process_failed_request(
          request,
          {StatusCode::RESOURCE_EXHAUSTED,
           "No enough memory to schedule a single sequence."});
    }
  }
}

void SchedulerPolicy::handle_unschedulable_head(
    RequestPriorityQueue* queue,
    SchedulerState& state,
    std::vector<std::shared_ptr<Request>>& finished,
    bool budget_exhausted,
    bool blocks_exhausted) {
  if (blocks_exhausted &&
      state.kv_cache_manager->has_pending_async_block_release()) {
    return;
  }
  if (state.running_sequences.empty() && !queue->empty() &&
      state.decode_queue.empty()) {
    std::shared_ptr<Request> request(queue->top());
    queue->pop_top();
    if (state.release_failed_request) {
      state.release_failed_request(request);
    }
    clear_mtp_bootstrap(request.get(), state);
    state.kv_cache_manager->deallocate(request.get());
    if (blocks_exhausted) {
      LOG(ERROR) << "Request prompt is too long, no enough memory to schedule "
                    "a single sequence.";
      state.output_dispatcher->process_failed_request(
          request,
          {StatusCode::RESOURCE_EXHAUSTED,
           "No enough memory to schedule single sequence"});
    } else if (budget_exhausted) {
      LOG(ERROR) << "Request prompt is too long, no enough budget to schedule "
                    "a single sequence. Please set a larger budget.";
      state.output_dispatcher->process_failed_request(
          request,
          {StatusCode::RESOURCE_EXHAUSTED,
           "No enough budget to schedule single sequence."});
    } else {
      LOG(FATAL) << "Unexpected error: blocks and budget are enough but can "
                    "not schedule.";
    }
  }
}

// =============================================================================
// Metrics reporting
// =============================================================================

void SchedulerPolicy::report_metrics(const SchedulerState& state,
                                     double elapsed_seconds,
                                     size_t num_preempted_requests) {
  GAUGE_SET(num_running_requests, state.running_requests.size());
  GAUGE_SET(num_waiting_requests,
            state.prefill_queue.size() + state.decode_restore_waiting.size());
  GAUGE_SET(num_preempted_requests, num_preempted_requests);
  GAUGE_SET(num_running_sequences, state.running_sequences.size());
  GAUGE_SET(kv_cache_utilization_perc,
            state.kv_cache_manager->kv_cache_utilization());
  GAUGE_SET(num_blocks_in_prefix_cache,
            util::min(state.kv_cache_manager->num_blocks_in_prefix_cache()));
  GAUGE_SET(num_free_blocks,
            util::max(state.kv_cache_manager->num_free_blocks()));
  GAUGE_SET(num_used_blocks,
            util::min(state.kv_cache_manager->num_used_blocks()));
}

// =============================================================================
// Helpers
// =============================================================================

void SchedulerPolicy::handle_running_requests(
    const std::shared_ptr<Request>& request,
    SchedulerState& state) {
  if (request->finished() || request->cancelled()) {
    LOG(FATAL) << "Unknown error, finished/cancelled request should have been "
                  "handled before. request_id is "
               << request->request_id();
  }

  if (request->expand_sequences()) {
    state.kv_cache_manager->cache(request->sequences()[0].get());
  }

  for (auto& sequence : request->sequences()) {
    if (sequence->finished()) {
      state.kv_cache_manager->deallocate(sequence.get());
    }
  }
}

void SchedulerPolicy::cache_in_batch_prefix(
    const std::vector<Sequence*>& sequences,
    const std::vector<size_t>& token_budgets,
    SchedulerState& state) {
  if (!state.enable_prefix_cache || sequences.empty()) {
    return;
  }
  bool enable_in_batch =
      ::xllm::KVCacheConfig::get_instance().enable_in_batch_prefix_cache();
  if (!enable_in_batch) {
    return;
  }
  CHECK_EQ(sequences.size(), token_budgets.size());
  for (size_t i = 0; i < sequences.size(); ++i) {
    Sequence* sequence = sequences[i];
    if (sequence == nullptr || !sequence->is_prefill_stage()) {
      continue;
    }
    const size_t max_handle_num_tokens =
        sequence->kv_state().kv_cache_tokens_num() + token_budgets[i];
    state.kv_cache_manager->cache(sequence, max_handle_num_tokens);
  }
}

void SchedulerPolicy::clear_mtp_bootstrap(Request* request,
                                          const SchedulerState& state) {
  if (!state.options.enable_disagg_pd() ||
      state.options.num_speculative_tokens() <= 0 || request == nullptr ||
      request->sequences().empty()) {
    return;
  }
  Sequence* sequence = request->sequences()[0].get();
  if (sequence == nullptr) {
    return;
  }
  sequence->clear_mtp_bootstrap_embedding();
}

// =============================================================================
// Factory
// =============================================================================

BatchMode create_batch_mode(const SchedulerOptions& options) {
  BatchMode mode;
  mode.priority_strategy = options.priority_strategy();
  mode.enable_chunked_prefill = options.enable_chunked_prefill();
  mode.enable_mix_batch =
      ::xllm::SchedulerConfig::get_instance().enable_mix_batch();

  // multi_slo_and_prio requires chunked prefill.
  if (mode.priority_strategy == "multi_slo_and_prio") {
    mode.enable_chunked_prefill = true;
  }

  // CP/KV-sharded decode/MTP: prefill cannot mix with decode in one batch.
  if (options.cp_size() > 1 ||
      ParallelConfig::get_instance().kv_split_size_effective() > 1 ||
      options.num_speculative_tokens() > 0) {
    mode.enable_mix_batch = false;
  }

  // No chunked prefill: prefill occupies the full batch exclusively.
  if (!mode.enable_chunked_prefill) {
    mode.enable_mix_batch = false;
  }

  // PD PREFILL instance: always use exclusive batch (PrefillFirstPolicy).
  // Prefill instances never have local decode requests, so mix batch is
  // meaningless and PrefillFirstPolicy gives cleaner chunk_queue priority.
  if (options.enable_disagg_pd() && options.instance_role().has_value() &&
      options.instance_role().value() == InstanceRole::PREFILL) {
    mode.enable_mix_batch = false;
  }

  return mode;
}

std::unique_ptr<SchedulerPolicy> create_scheduler_policy(
    const BatchMode& mode,
    const SchedulerOptions& options) {
  if (mode.enable_mix_batch && mode.priority_strategy == "multi_slo_and_prio") {
    return std::make_unique<UnifiedPolicy>(mode, options);
  } else if (mode.enable_mix_batch) {
    return std::make_unique<DecodeFirstPolicy>(mode, options);
  } else {
    return std::make_unique<PrefillFirstPolicy>(mode, options);
  }
}

}  // namespace xllm

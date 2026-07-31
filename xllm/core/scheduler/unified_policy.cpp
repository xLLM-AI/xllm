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

#include <algorithm>
#include <cstdint>
#include <limits>

#include "async_response_processor.h"
#include "common/metrics.h"
#include "core/framework/config/kv_cache_store_config.h"
#include "core/framework/config/scheduler_config.h"
#include "framework/request/priority_comparator.h"
#include "scheduler/scheduler_policy.h"

namespace xllm {

// =============================================================================
// UnifiedPolicy::drain_request_queue
// =============================================================================

void UnifiedPolicy::drain_request_queue(
    SchedulerState& state,
    folly::MPMCQueue<std::shared_ptr<Request>>& request_queue) {
  std::shared_ptr<Request> request;
  while (request_queue.read(request)) {
    CHECK(request);
    if (!state.enable_prefix_cache) {
      request->expand_sequences(/*force=*/false);
    }
    state.unified_queue.push_back(request);
  }
}

// =============================================================================
// UnifiedPolicy::schedule
// =============================================================================

void UnifiedPolicy::schedule(SchedulerState& state,
                             ScheduleBudget& budget,
                             std::vector<std::shared_ptr<Request>>& finished) {
  // === Requeue phase ===
  // All running requests go back into unified_queue.
  for (auto it = state.running_requests.rbegin();
       it != state.running_requests.rend();
       ++it) {
    if (*it == nullptr) {
      continue;
    }
    handle_running_requests(*it, state);
    state.unified_queue.push_back(*it);
  }
  reset_batch_state(state);

  // Allocate prefix cache ahead for all requests.
  for (auto& request : state.unified_queue) {
    auto& sequence = request->sequences()[0];
    allocate_shared_blocks_for(sequence.get(), state);
  }

  // === Sort + schedule ===
  get_latency_budget_and_request_order(
      state.unified_queue, budget.latency_budget, state);
  schedule_from_unified_queue(state.unified_queue, state, budget, finished);
  // Remaining unscheduled requests stay in unified_queue for next step.
}

// =============================================================================
// UnifiedPolicy::schedule_from_unified_queue
// =============================================================================

void UnifiedPolicy::schedule_from_unified_queue(
    std::list<std::shared_ptr<Request>>& unified,
    SchedulerState& state,
    ScheduleBudget& budget,
    std::vector<std::shared_ptr<Request>>& finished) {
  if (unified.empty()) {
    return;
  }

  size_t remaining_copy_units_budget =
      (options_.enable_latency_aware_schedule() &&
       ::xllm::KVCacheStoreConfig::get_instance()
           .enable_control_h2d_block_num())
          ? get_max_copy_block_num(unified, budget, state)
          : std::numeric_limits<int32_t>::max();

  std::vector<std::shared_ptr<Request>> preempted_request_vec;
  bool is_preempt_iterator_valid = true;
  auto preempt_iterator = std::prev(unified.end());

  while (!unified.empty() &&
         budget.remaining_token_budget >
             static_cast<size_t>(state.min_speculative_tokens_required) &&
         budget.latency_budget > budget.estimate_latency &&
         budget.remaining_seq_budget > 0) {
    std::shared_ptr<Request> request(unified.front());
    if (preempt_iterator == unified.begin()) {
      is_preempt_iterator_valid = false;
    }

    if (request->finished() || request->cancelled()) {
      clear_mtp_bootstrap(request.get(), state);
      state.kv_cache_manager->deallocate(request.get());
      finished.emplace_back(request);
      unified.pop_front();
      continue;
    }

    const size_t num_sequences = request->sequences().size();
    CHECK(num_sequences == 1) << "MixScheduler currently only supports one "
                                 "sequence per request.";

    std::vector<Sequence*> candidate_sequences;
    std::vector<size_t> candidate_token_budgets;
    candidate_sequences.reserve(num_sequences);
    candidate_token_budgets.reserve(num_sequences);

    bool budget_exhausted = false;
    bool blocks_exhausted = false;
    size_t allocated_tokens = 0;
    size_t allocated_seqs = 0;
    double allocated_estimate_latency = 0;
    size_t allocated_copy_units = 0;

    for (auto& sequence : request->sequences()) {
      if (sequence->finished()) {
        continue;
      }

      const size_t unallocated_copy_units =
          remaining_copy_units_budget > allocated_copy_units
              ? remaining_copy_units_budget - allocated_copy_units
              : 0;
      const bool host_cache_enabled =
          state.kv_cache_manager->supports_host_cache_restore();
      const size_t full_copy_units = sequence->host_cache_copy_units();
      HostCacheRestorePoint selected_restore{
          /*restore_target_tokens=*/sequence->kv_cache_tokens_num(),
          /*copy_units=*/full_copy_units};
      if (unallocated_copy_units < full_copy_units) {
        selected_restore = state.kv_cache_manager->select_host_cache_restore(
            sequence.get(), unallocated_copy_units);
      }

      const size_t current_step_copy_units = selected_restore.copy_units;
      const size_t kv_cache_tokens_num = selected_restore.restore_target_tokens;

      size_t num_tokens = sequence->num_tokens();
      size_t assume_max_tokens =
          budget.remaining_token_budget - allocated_tokens;

      // Latency-aware scheduling.
      if (options_.enable_latency_aware_schedule()) {
        if (sequence->is_prefill_stage()) {
          assume_max_tokens =
              get_max_chunk(sequence.get(),
                            num_tokens,
                            kv_cache_tokens_num,
                            static_cast<int32_t>(budget.latency_budget -
                                                 budget.estimate_latency),
                            state);
          if (assume_max_tokens == kv_cache_tokens_num) {
            budget_exhausted = true;
            break;
          }
          if (assume_max_tokens != num_tokens &&
              (assume_max_tokens - kv_cache_tokens_num) <= 50) {
            budget_exhausted = true;
            break;
          }
          allocated_estimate_latency = state.profile_manager->predict_step_time(
              assume_max_tokens,
              kv_cache_tokens_num,
              /*if_need_add_constant_term=*/false);
          assume_max_tokens -= kv_cache_tokens_num;
        } else {
          assume_max_tokens = 1;
          allocated_estimate_latency = state.profile_manager->predict_step_time(
              num_tokens,
              kv_cache_tokens_num,
              /*if_need_add_constant_term=*/false);
          if (budget.estimate_latency + allocated_estimate_latency >
              budget.latency_budget) {
            budget_exhausted = true;
            break;
          }
        }
      } else {
        // Token-budget based scheduling.
        size_t num_tokens_to_handle;
        if (sequence->is_prefill_stage()) {
          size_t remaining = num_tokens - kv_cache_tokens_num;
          size_t max_chunk =
              static_cast<size_t>(options_.max_tokens_per_chunk_for_prefill());
          num_tokens_to_handle =
              std::min({assume_max_tokens, max_chunk, remaining});
        } else {
          num_tokens_to_handle = 1 + state.min_speculative_tokens_required;
        }
        if (allocated_seqs + 1 > budget.remaining_seq_budget ||
            allocated_tokens + num_tokens_to_handle >
                budget.remaining_token_budget) {
          budget_exhausted = true;
          break;
        }
      }

      // Allocate blocks after committing the scheduler-selected Host restore
      // boundary. The allocator only executes the resulting restore plan.
      size_t max_handle_num_tokens =
          std::min(kv_cache_tokens_num + assume_max_tokens, num_tokens);
      if (sequence->is_prefill_stage()) {
        size_t max_chunk =
            static_cast<size_t>(options_.max_tokens_per_chunk_for_prefill());
        max_handle_num_tokens =
            std::min(max_handle_num_tokens, kv_cache_tokens_num + max_chunk);
      }
      if (options_.num_speculative_tokens() > 0 &&
          !sequence->is_chunked_prefill_stage() && kv_cache_tokens_num > 0) {
        max_handle_num_tokens += state.min_speculative_tokens_required;
      }
      CHECK_GT(max_handle_num_tokens, kv_cache_tokens_num);
      size_t current_step_handle_tokens =
          max_handle_num_tokens - kv_cache_tokens_num;

      bool alloc_success = false;
      if (host_cache_enabled) {
        state.kv_cache_manager->trim_host_cache(sequence.get(),
                                                selected_restore);
        alloc_success = state.kv_cache_manager->allocate(sequence.get(),
                                                         max_handle_num_tokens);
      } else {
        alloc_success = state.kv_cache_manager->allocate(sequence.get(),
                                                         max_handle_num_tokens);
      }
      if (!alloc_success) {
        blocks_exhausted = true;
        break;
      }

      allocated_tokens += current_step_handle_tokens;
      allocated_seqs += 1;
      allocated_copy_units += current_step_copy_units;
      candidate_sequences.emplace_back(sequence.get());
      candidate_token_budgets.emplace_back(current_step_handle_tokens);
    }

    if (!blocks_exhausted && !budget_exhausted) {
      unified.pop_front();
      request->record_num_prefix_cache_tokens();
      state.running_requests.emplace_back(request);
      state.running_sequences.insert(state.running_sequences.end(),
                                     candidate_sequences.begin(),
                                     candidate_sequences.end());
      state.running_sequences_budgets.insert(
          state.running_sequences_budgets.end(),
          candidate_token_budgets.begin(),
          candidate_token_budgets.end());
      cache_in_batch_prefix(
          candidate_sequences, candidate_token_budgets, state);
      budget.remaining_token_budget -= allocated_tokens;
      budget.remaining_seq_budget -= allocated_seqs;
      remaining_copy_units_budget -= allocated_copy_units;
      budget.estimate_latency += allocated_estimate_latency;
      continue;
    }

    if (budget_exhausted) {
      if (candidate_sequences.empty() && state.running_sequences.empty()) {
        LOG(ERROR) << "Request prompt = "
                   << request->sequences()[0]->num_tokens()
                   << " is too long, please set a larger "
                      "max_tokens value via --max_tokens_per_batch.";
        unified.pop_front();
        clear_mtp_bootstrap(request.get(), state);
        state.kv_cache_manager->deallocate(request.get());
        state.response_processor->process_failed_request(
            request,
            {StatusCode::RESOURCE_EXHAUSTED,
             "No enough resource to schedule a single sequence"});
      }
      break;
    }

    // Memory exhausted -- preempt lowest priority request.
    bool find_preempt = false;
    while (is_preempt_iterator_valid && preempt_iterator != unified.begin()) {
      std::shared_ptr<Request> request_to_preempt = *preempt_iterator;
      if (request_to_preempt.get() != request.get()) {
        if (request_to_preempt->sequences()[0]
                ->kv_state()
                .kv_cache_tokens_num() != 0) {
          ++budget.num_preempted_requests;
          clear_mtp_bootstrap(request_to_preempt.get(), state);
          state.kv_cache_manager->deallocate(request_to_preempt.get());
          auto prev = preempt_iterator;
          preempt_iterator--;
          unified.erase(prev);
          request_to_preempt->set_preempted();
          preempted_request_vec.push_back(request_to_preempt);
          find_preempt = true;
          break;
        } else {
          preempt_iterator--;
        }
      } else {
        LOG(FATAL) << "Unexpected error: preempting the candidate itself.";
      }
    }
    if (find_preempt) {
      continue;
    }

    // No enough memory to preempt.
    if (candidate_sequences.empty() && state.running_sequences.empty()) {
      LOG(ERROR) << "Request prompt is too long, no enough memory to schedule "
                 << "a single sequence.";
      unified.pop_front();
      clear_mtp_bootstrap(request.get(), state);
      state.kv_cache_manager->deallocate(request.get());
      state.response_processor->process_failed_request(
          request,
          {StatusCode::RESOURCE_EXHAUSTED,
           "No enough resource to schedule a single sequence"});
    }
    break;
  }

  // Push preempted requests back to unified queue.
  while (!preempted_request_vec.empty()) {
    unified.push_back(preempted_request_vec.back());
    preempted_request_vec.pop_back();
  }
}

// =============================================================================
// UnifiedPolicy::get_latency_budget_and_request_order
// =============================================================================

void UnifiedPolicy::get_latency_budget_and_request_order(
    std::list<std::shared_ptr<Request>>& queue,
    double& latency_budget,
    const SchedulerState& state) {
  if (queue.empty()) {
    return;
  }

  // Update request metrics.
  for (auto& request : queue) {
    auto& sequence = request->sequences()[0];
    const HostCacheRestorePoint full_restore =
        state.kv_cache_manager->select_host_cache_restore(
            sequence.get(), std::numeric_limits<size_t>::max());
    const size_t candidate_prefix = std::max(
        sequence->kv_cache_tokens_num(), full_restore.restore_target_tokens);
    const size_t usable_prefix =
        sequence->num_tokens() == 0
            ? 0
            : std::min(candidate_prefix, sequence->num_tokens() - 1);
    sequence->set_estimated_latency(state.profile_manager->predict_step_time(
        sequence->num_tokens(),
        usable_prefix,
        /*if_need_add_constant_term=*/false));
    request->set_elapsed_time_ms();
    request->set_deadline_ms();
    request->set_starved(false);
  }

  double constant_overhead = state.profile_manager->get_constant_overhead();
  double total_exec_time = 0.0;
  int32_t min_remaining_time = std::numeric_limits<int32_t>::max();
  int32_t min_tpot = std::numeric_limits<int32_t>::max();

  for (auto it = queue.cbegin(); it != queue.cend(); ++it) {
    const auto request = *it;
    const auto& sequence = request->sequences()[0];
    auto remaining_time = request->get_remaining_time();
    total_exec_time += sequence->estimated_latency();
    if (request->tpot_slo_ms() < min_tpot) {
      min_tpot = static_cast<int32_t>(request->tpot_slo_ms());
    }
    if (remaining_time < sequence->estimated_latency() + constant_overhead) {
      continue;
    }
    if (remaining_time < min_remaining_time) {
      min_remaining_time = static_cast<int32_t>(remaining_time);
    }
  }

  int32_t latency_budget_threshold = static_cast<int32_t>(0.65 * min_tpot);
  latency_budget = std::max(min_remaining_time, latency_budget_threshold);

  const double lambda =
      ::xllm::SchedulerConfig::get_instance().aggressive_coeff();
  const double denominator = std::max(latency_budget - constant_overhead, 1e-6);
  double load_judge_func = total_exec_time * latency_budget / denominator;

  for (auto& request : queue) {
    auto& sequence = request->sequences()[0];

    if (::xllm::SchedulerConfig::get_instance().enable_starve_prevent()) {
      const int32_t starve_unit_time = sequence->is_prefill_stage()
                                           ? -request->ttft_slo_ms()
                                           : -request->tpot_slo_ms();
      const int32_t starve_time_threshold = static_cast<int32_t>(
          ::xllm::SchedulerConfig::get_instance().starve_threshold() *
          starve_unit_time);
      if (request->get_remaining_time() < starve_time_threshold) {
        request->set_starved(true);
      }
    }

    if (request->get_remaining_time() < lambda * load_judge_func) {
      request->set_urgency(Urgency::URGENT);
    } else {
      request->set_urgency(Urgency::NORMAL);
    }
  }

  // Sort using multi_slo_and_prio comparator.
  auto priority_strategy = options_.priority_strategy();
  if (priority_strategy == "fcfs") {
    priority_strategy = "multi_slo_and_prio";
  }
  queue.sort(create_comparator(priority_strategy, true));
}

// =============================================================================
// UnifiedPolicy::get_max_copy_block_num
// =============================================================================

size_t UnifiedPolicy::get_max_copy_block_num(
    std::list<std::shared_ptr<Request>>& queue,
    ScheduleBudget& budget,
    const SchedulerState& state) {
  double min_total_exec_time = state.profile_manager->get_constant_overhead();
  size_t full_copy_units = 0;
  std::vector<size_t> per_request_copy_units;
  std::vector<std::shared_ptr<Request>> requests_with_host_restore;

  for (auto& request : queue) {
    auto& sequence = request->sequences()[0];
    const HostCacheRestorePoint full_restore =
        state.kv_cache_manager->select_host_cache_restore(
            sequence.get(), std::numeric_limits<size_t>::max());
    const size_t candidate_prefix = std::max(
        sequence->kv_cache_tokens_num(), full_restore.restore_target_tokens);
    const size_t usable_prefix =
        sequence->num_tokens() == 0
            ? 0
            : std::min(candidate_prefix, sequence->num_tokens() - 1);
    min_total_exec_time += state.profile_manager->predict_step_time(
        sequence->num_tokens(),
        usable_prefix,
        /*if_need_add_constant_term=*/false);

    full_copy_units += full_restore.copy_units;
    if (full_restore.copy_units > 0) {
      per_request_copy_units.push_back(full_restore.copy_units);
      requests_with_host_restore.push_back(request);
    }
  }

  size_t max_copy_units = std::numeric_limits<int32_t>::max();
  if (min_total_exec_time >= budget.latency_budget) {
    max_copy_units =
        state.profile_manager->get_max_copy_block_num(budget.latency_budget);
  } else if (full_copy_units > 0) {
    const double full_h2d_transfer_time =
        state.profile_manager->predict_copy_blocks_time(full_copy_units);
    if (full_h2d_transfer_time > min_total_exec_time) {
      max_copy_units = get_needed_copy_block_num(requests_with_host_restore,
                                                 per_request_copy_units,
                                                 full_h2d_transfer_time,
                                                 min_total_exec_time,
                                                 full_copy_units,
                                                 state);
    }
  }
  return max_copy_units;
}

// =============================================================================
// UnifiedPolicy::get_needed_copy_block_num
// =============================================================================

size_t UnifiedPolicy::get_needed_copy_block_num(
    const std::vector<std::shared_ptr<Request>>& requests,
    const std::vector<size_t>& per_request_copy_units,
    double full_h2d_transfer_time,
    double full_restore_exec_time,
    size_t full_copy_units,
    const SchedulerState& state) {
  if (requests.empty()) {
    CHECK(per_request_copy_units.empty());
    return 0;
  }
  CHECK_EQ(requests.size(), per_request_copy_units.size());

  size_t needed_copy_units = full_copy_units;
  double total_exec_time = full_restore_exec_time;
  double h2d_transfer_time = full_h2d_transfer_time;
  CHECK_GT(h2d_transfer_time, total_exec_time);

  size_t index = requests.size() - 1;
  for (auto it = requests.rbegin(); it != requests.rend(); ++it, --index) {
    Sequence* sequence = (*it)->sequences()[0].get();
    const HostCacheRestorePoint full_restore =
        state.kv_cache_manager->select_host_cache_restore(
            sequence, std::numeric_limits<size_t>::max());
    CHECK_EQ(full_restore.copy_units, per_request_copy_units[index]);

    const size_t full_prefix =
        sequence->num_tokens() == 0
            ? 0
            : std::min(full_restore.restore_target_tokens,
                       sequence->num_tokens() - 1);
    const double full_exec_time = state.profile_manager->predict_step_time(
        sequence->num_tokens(),
        full_prefix,
        /*if_need_add_constant_term=*/false);
    needed_copy_units -= full_restore.copy_units;
    total_exec_time -= full_exec_time;
    h2d_transfer_time -= state.profile_manager->predict_copy_blocks_time(
        full_restore.copy_units, /*if_need_add_constant_term=*/false);

    HostCacheRestorePoint selected_restore =
        state.kv_cache_manager->select_host_cache_restore(sequence,
                                                          /*max_copy_units=*/0);
    double selected_exec_time = 0;
    double selected_total_exec_time = 0;
    double selected_total_h2d_time = 0;
    double selected_latency = std::numeric_limits<double>::max();
    auto evaluate_restore = [&](const HostCacheRestorePoint& restore) {
      if (restore.copy_units > full_restore.copy_units ||
          sequence->num_tokens() == 0) {
        return;
      }
      const size_t prefix =
          std::min(restore.restore_target_tokens, sequence->num_tokens() - 1);
      const double exec_time = state.profile_manager->predict_step_time(
          sequence->num_tokens(),
          prefix,
          /*if_need_add_constant_term=*/false);
      const double candidate_h2d_time =
          h2d_transfer_time + state.profile_manager->predict_copy_blocks_time(
                                  restore.copy_units,
                                  /*if_need_add_constant_term=*/false);
      const double candidate_exec_time = total_exec_time + exec_time;
      const double latency = std::max(candidate_h2d_time, candidate_exec_time);
      if (latency < selected_latency ||
          (latency == selected_latency &&
           restore.restore_target_tokens >
               selected_restore.restore_target_tokens)) {
        selected_restore = restore;
        selected_exec_time = exec_time;
        selected_total_exec_time = candidate_exec_time;
        selected_total_h2d_time = candidate_h2d_time;
        selected_latency = latency;
      }
    };

    evaluate_restore(selected_restore);
    for (size_t copy_units = 1; copy_units <= full_restore.copy_units;
         ++copy_units) {
      evaluate_restore(state.kv_cache_manager->select_host_cache_restore(
          sequence, copy_units));
    }

    needed_copy_units += selected_restore.copy_units;
    if (selected_restore.copy_units > 0 ||
        selected_total_h2d_time <= selected_total_exec_time) {
      return needed_copy_units;
    }

    total_exec_time += selected_exec_time;
  }
  return needed_copy_units;
}

// =============================================================================
// UnifiedPolicy::get_max_chunk
// =============================================================================

int32_t UnifiedPolicy::get_max_chunk(Sequence* sequence,
                                     size_t num_tokens,
                                     size_t kv_cache_tokens_num,
                                     int32_t latency_budget,
                                     const SchedulerState& state) {
  if (num_tokens <= kv_cache_tokens_num) {
    return kv_cache_tokens_num;
  }
  if (state.profile_manager->predict_step_time(
          num_tokens,
          kv_cache_tokens_num,
          /*if_need_add_constant_term=*/false) <= latency_budget) {
    return num_tokens;
  }
  if (latency_budget <= 0) {
    return kv_cache_tokens_num;
  }

  // Binary search for the maximum chunk that fits within latency budget.
  int32_t left = kv_cache_tokens_num + 1;
  int32_t right = num_tokens + 1;
  while (left < right) {
    int32_t mid = left + (right - left) / 2;
    const double predict_time = state.profile_manager->predict_step_time(
        mid,
        kv_cache_tokens_num,
        /*if_need_add_constant_term=*/false);
    if (predict_time <= latency_budget) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left - 1;
}

}  // namespace xllm

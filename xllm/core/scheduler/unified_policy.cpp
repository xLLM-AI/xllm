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

  size_t remaining_copy_blocks_budget =
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
    size_t allocated_copy_blocks = 0;

    for (auto& sequence : request->sequences()) {
      if (sequence->finished()) {
        continue;
      }

      // H2D swap support.
      int32_t block_size = state.kv_cache_manager->block_size();
      size_t host_blocks_num =
          sequence->host_kv_state().kv_cache_tokens_num() / block_size;
      size_t device_blocks_num =
          sequence->kv_state().kv_cache_tokens_num() / block_size;
      size_t cur_step_copy_blocks = host_blocks_num > device_blocks_num
                                        ? host_blocks_num - device_blocks_num
                                        : 0;
      cur_step_copy_blocks =
          std::min(cur_step_copy_blocks,
                   remaining_copy_blocks_budget - allocated_copy_blocks);
      size_t kv_cache_tokens_num =
          cur_step_copy_blocks == 0
              ? sequence->kv_state().kv_cache_tokens_num()
              : (sequence->kv_state().kv_cache_tokens_num() / block_size +
                 cur_step_copy_blocks) *
                    block_size;

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
              assume_max_tokens, kv_cache_tokens_num, false);
          assume_max_tokens -= kv_cache_tokens_num;
        } else {
          assume_max_tokens = 1;
          allocated_estimate_latency = state.profile_manager->predict_step_time(
              num_tokens, kv_cache_tokens_num, false);
          if (budget.estimate_latency + allocated_estimate_latency >
              budget.latency_budget) {
            budget_exhausted = true;
            break;
          }
        }
      } else {
        // Token-budget based scheduling.
        size_t num_tokens_to_handle =
            sequence->is_prefill_stage()
                ? std::min(assume_max_tokens, num_tokens - kv_cache_tokens_num)
                : 1 + state.min_speculative_tokens_required;
        if (allocated_seqs + 1 > budget.remaining_seq_budget ||
            allocated_tokens + num_tokens_to_handle >
                budget.remaining_token_budget) {
          budget_exhausted = true;
          break;
        }
      }

      // Allocate blocks (MixScheduler version with copy blocks support).
      size_t max_handle_num_tokens =
          std::min(kv_cache_tokens_num + assume_max_tokens, num_tokens);
      if (options_.num_speculative_tokens() > 0 &&
          !sequence->is_chunked_prefill_stage() && kv_cache_tokens_num > 0) {
        max_handle_num_tokens += state.min_speculative_tokens_required;
      }
      CHECK_GT(max_handle_num_tokens, kv_cache_tokens_num);
      size_t current_step_handle_tokens =
          max_handle_num_tokens - kv_cache_tokens_num;

      bool alloc_success = false;
      if (::xllm::KVCacheStoreConfig::get_instance().host_blocks_factor() >
          1.0) {
        alloc_success = state.kv_cache_manager->allocate(
            sequence.get(), max_handle_num_tokens, cur_step_copy_blocks);
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
      allocated_copy_blocks += cur_step_copy_blocks;
      candidate_sequences.emplace_back(sequence.get());
      candidate_token_budgets.emplace_back(current_step_handle_tokens);
    }

    if (!blocks_exhausted && !budget_exhausted) {
      unified.pop_front();
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
      remaining_copy_blocks_budget -= allocated_copy_blocks;
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
    sequence->set_estimated_latency(
        state.profile_manager->predict_step_time(sequence.get(), false));
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
  size_t max_h2d_block_num = 0;
  int32_t block_size = state.kv_cache_manager->block_size();
  std::vector<size_t> req_copy_block_num_vec;
  std::vector<std::shared_ptr<Request>> req_vec;

  for (auto& request : queue) {
    auto& sequence = request->sequences()[0];
    min_total_exec_time +=
        state.profile_manager->predict_step_time(sequence.get(), false);

    size_t host_blocks_num =
        sequence->host_kv_state().kv_cache_tokens_num() / block_size;
    size_t device_blocks_num =
        sequence->kv_state().kv_cache_tokens_num() / block_size;
    size_t cur_step_copy_blocks = host_blocks_num > device_blocks_num
                                      ? host_blocks_num - device_blocks_num
                                      : 0;
    max_h2d_block_num += cur_step_copy_blocks;
    if (cur_step_copy_blocks > 0) {
      req_copy_block_num_vec.push_back(cur_step_copy_blocks);
      req_vec.push_back(request);
    }
  }

  size_t max_copy_block_num = std::numeric_limits<int32_t>::max();
  if (min_total_exec_time >= budget.latency_budget) {
    max_copy_block_num =
        state.profile_manager->get_max_copy_block_num(budget.latency_budget);
  } else {
    double max_h2d_transfer_time =
        state.profile_manager->predict_copy_blocks_time(max_h2d_block_num);
    if (max_h2d_transfer_time > min_total_exec_time) {
      max_copy_block_num = get_needed_copy_block_num(req_vec,
                                                     req_copy_block_num_vec,
                                                     max_h2d_transfer_time,
                                                     min_total_exec_time,
                                                     max_h2d_block_num,
                                                     state);
    }
  }
  return max_copy_block_num;
}

// =============================================================================
// UnifiedPolicy::get_needed_copy_block_num
// =============================================================================

size_t UnifiedPolicy::get_needed_copy_block_num(
    std::vector<std::shared_ptr<Request>>& req_vec,
    std::vector<size_t>& req_copy_block_num_vec,
    double max_h2d_transfer_time,
    double min_total_exec_time,
    size_t max_h2d_block_num,
    const SchedulerState& state) {
  int32_t block_size = state.kv_cache_manager->block_size();
  size_t total_needed_copy_blocks = max_h2d_block_num;
  double total_exec_time = min_total_exec_time;
  double h2d_transfer_time = max_h2d_transfer_time;
  CHECK_GT(h2d_transfer_time, total_exec_time);

  size_t index = req_vec.size() - 1;
  for (auto it = req_vec.rbegin(); it != req_vec.rend(); ++it, --index) {
    auto request = *it;
    auto& sequence = request->sequences()[0];
    total_needed_copy_blocks -= req_copy_block_num_vec[index];
    total_exec_time -=
        state.profile_manager->predict_step_time(sequence.get(), false);
    double cur_seq_max_exec_time = state.profile_manager->predict_step_time(
        sequence->num_tokens(),
        sequence->kv_state().kv_cache_tokens_num(),
        false);
    total_exec_time += cur_seq_max_exec_time;
    h2d_transfer_time -= state.profile_manager->predict_copy_blocks_time(
        req_copy_block_num_vec[index], false);

    if (h2d_transfer_time < total_exec_time) {
      double base_total_exec_time = total_exec_time - cur_seq_max_exec_time;
      size_t left = 0;
      size_t right = req_copy_block_num_vec[index] + 1;
      double min_latency = std::numeric_limits<double>::max();
      while (left < right) {
        size_t mid = left + (right - left) / 2;
        double cur_seq_h2d_time =
            state.profile_manager->predict_copy_blocks_time(mid, false);
        size_t kv_cache_tokens_num =
            mid == 0
                ? sequence->kv_state().kv_cache_tokens_num()
                : (sequence->kv_state().kv_cache_tokens_num() / block_size +
                   mid) *
                      block_size;
        double cur_seq_exec_time = state.profile_manager->predict_step_time(
            sequence->num_tokens(), kv_cache_tokens_num, false);
        if (h2d_transfer_time + cur_seq_h2d_time <
            base_total_exec_time + cur_seq_exec_time) {
          left = mid + 1;
          min_latency = std::max(h2d_transfer_time + cur_seq_h2d_time,
                                 base_total_exec_time + cur_seq_exec_time);
        } else {
          right = mid;
        }
      }
      size_t needed_copy_blocks = left - 1;
      if (left <= req_copy_block_num_vec[index]) {
        double cur_seq_h2d_time =
            state.profile_manager->predict_copy_blocks_time(left, false);
        size_t kv_cache_tokens_num =
            left == 0
                ? sequence->kv_state().kv_cache_tokens_num()
                : (sequence->kv_state().kv_cache_tokens_num() / block_size +
                   left) *
                      block_size;
        double cur_seq_exec_time = state.profile_manager->predict_step_time(
            sequence->num_tokens(), kv_cache_tokens_num, false);
        double candidate_latency =
            std::max(h2d_transfer_time + cur_seq_h2d_time,
                     base_total_exec_time + cur_seq_exec_time);
        if (min_latency > candidate_latency) {
          needed_copy_blocks = left;
        }
      }

      total_needed_copy_blocks += needed_copy_blocks;
      break;
    }
  }
  return total_needed_copy_blocks;
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
          num_tokens, kv_cache_tokens_num, false) <= latency_budget) {
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
    auto predict_time = state.profile_manager->predict_step_time(
        mid, kv_cache_tokens_num, false);
    if (predict_time <= latency_budget) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left - 1;
}

}  // namespace xllm

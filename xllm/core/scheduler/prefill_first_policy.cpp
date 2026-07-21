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

#include "core/framework/config/scheduler_config.h"
#include "framework/request/priority_comparator.h"
#include "scheduler/scheduler_policy.h"

namespace xllm {

// =============================================================================
// PrefillFirstPolicy::schedule
// =============================================================================

void PrefillFirstPolicy::schedule(
    SchedulerState& state,
    ScheduleBudget& budget,
    std::vector<std::shared_ptr<Request>>& finished) {
  // === Requeue phase ===
  if (batch_mode_.priority_strategy == "fcfs") {
    // ContinuousScheduler / PrefillOnlyScheduler with FCFS:
    // Respect last_step_prefill for insertion order.
    if (state.last_step_prefill) {
      // Forward iterate: push_back for decode maintains FCFS order.
      // Continuations go to chunk_queue front (reverse order preserved by
      // collecting then pushing).
      for (auto it = state.running_requests.cbegin();
           it != state.running_requests.cend();
           ++it) {
        if (*it == nullptr) {
          continue;
        }
        auto req = *it;
        handle_running_requests(req, state);
        if (batch_mode_.enable_chunked_prefill &&
            req->sequences()[0]->is_chunked_prefill_stage()) {
          state.chunk_queue.push(req, /*if_back=*/false);
        } else {
          state.decode_queue.push(req, /*if_back=*/true);
        }
      }
    } else {
      // last step was decode -- all requests here are decode stage
      for (auto it = state.running_requests.crbegin();
           it != state.running_requests.crend();
           ++it) {
        if (*it == nullptr) {
          continue;
        }
        auto req = *it;
        handle_running_requests(req, state);
        state.decode_queue.push(req, /*if_back=*/false);
      }
    }
  } else {
    // Priority / deadline / multi_slo_and_prio in exclusive mode.
    for (auto it = state.running_requests.cbegin();
         it != state.running_requests.cend();
         ++it) {
      if (*it == nullptr) {
        continue;
      }
      auto req = *it;
      handle_running_requests(req, state);
      if (batch_mode_.enable_chunked_prefill &&
          req->sequences()[0]->is_chunked_prefill_stage()) {
        state.chunk_queue.push(req, /*if_back=*/false);
      } else {
        state.decode_queue.push(req);
      }
    }
  }
  reset_batch_state(state);

  // === Schedule phase ===
  budget.latency_budget = options_.max_global_ttft_ms();

  // Compute urgency + reorder for both prefill queues together.
  adjust_latency_budget_and_reorder(&state.chunk_queue,
                                    &state.prefill_queue,
                                    budget.latency_budget,
                                    /*for_prefill=*/true,
                                    state);

  // Schedule chunked prefill continuations first (they already have partial
  // KV).
  schedule_prefill_from_queue(&state.chunk_queue, state, budget, finished);
  // Then new prefill requests.
  schedule_prefill_from_queue(&state.prefill_queue, state, budget, finished);

  if (!state.running_sequences.empty()) {
    state.last_step_prefill = true;
  }

  // If no prefill sequences were scheduled, try decode.
  if (state.running_sequences.empty()) {
    budget.latency_budget = options_.max_global_tpot_ms();
    adjust_latency_budget_and_reorder(&state.decode_queue,
                                      /*second_queue=*/nullptr,
                                      budget.latency_budget,
                                      /*for_prefill=*/false,
                                      state);
    schedule_decode_from_queue(&state.decode_queue, state, budget);
  }
}

// =============================================================================
// PrefillFirstPolicy::adjust_latency_budget_and_reorder
// =============================================================================

void PrefillFirstPolicy::adjust_latency_budget_and_reorder(
    RequestPriorityQueue* first_queue,
    RequestPriorityQueue* second_queue,
    double& latency_budget,
    bool for_prefill,
    const SchedulerState& state) {
  if (batch_mode_.priority_strategy != "multi_slo_and_prio") {
    return;
  }
  bool both_empty = (first_queue == nullptr || first_queue->empty()) &&
                    (second_queue == nullptr || second_queue->empty());
  if (both_empty) {
    return;
  }
  CHECK(state.profile_manager != nullptr);

  const double constant_overhead =
      state.profile_manager->get_constant_overhead();
  double total_exec_time = 0.0;
  int32_t min_remaining_time = std::numeric_limits<int32_t>::max();
  int32_t min_tpot = std::numeric_limits<int32_t>::max();

  // Lambda to process one queue's requests for metrics.
  auto process_queue = [&](RequestPriorityQueue* queue) {
    if (queue == nullptr || queue->empty()) {
      return;
    }
    for (auto it = queue->begin(); it != queue->end(); ++it) {
      auto request = *it;
      auto& sequence = request->sequences()[0];
      sequence->set_estimated_latency(
          state.profile_manager->predict_step_time(sequence.get(), false));
      request->set_elapsed_time_ms();
      request->set_deadline_ms();
      request->set_starved(false);

      const int32_t remaining_time = request->get_remaining_time();
      total_exec_time += sequence->estimated_latency();
      if (request->tpot_slo_ms() < min_tpot) {
        min_tpot = request->tpot_slo_ms();
      }
      if (remaining_time < sequence->estimated_latency() + constant_overhead) {
        continue;
      }
      if (remaining_time < min_remaining_time) {
        min_remaining_time = remaining_time;
      }
    }
  };

  process_queue(first_queue);
  process_queue(second_queue);

  if (!for_prefill) {
    int32_t latency_budget_threshold = static_cast<int32_t>(0.65 * min_tpot);
    latency_budget = std::max(min_remaining_time, latency_budget_threshold);
  }

  const double lambda = SchedulerConfig::get_instance().aggressive_coeff();
  double load_judge_func = 0.0;
  if (for_prefill) {
    load_judge_func = total_exec_time + constant_overhead;
  } else {
    const double denominator =
        std::max(latency_budget - constant_overhead, 1e-6);
    load_judge_func = total_exec_time * latency_budget / denominator;
  }

  // Lambda to set urgency on one queue's requests.
  auto set_urgency = [&](RequestPriorityQueue* queue) {
    if (queue == nullptr || queue->empty()) {
      return;
    }
    for (auto it = queue->begin(); it != queue->end(); ++it) {
      auto request = *it;
      auto& sequence = request->sequences()[0];

      if (SchedulerConfig::get_instance().enable_starve_prevent()) {
        const int32_t starve_unit_time = sequence->is_prefill_stage()
                                             ? -request->ttft_slo_ms()
                                             : -request->tpot_slo_ms();
        const int32_t starve_time_threshold = static_cast<int32_t>(
            SchedulerConfig::get_instance().starve_threshold() *
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
  };

  set_urgency(first_queue);
  set_urgency(second_queue);

  // Sort each queue independently with the same comparator.
  auto comparator = create_comparator("multi_slo_and_prio", true);
  if (first_queue != nullptr && !first_queue->empty()) {
    CHECK(first_queue->supports_sort())
        << "multi_slo_and_prio requires sortable request queue.";
    first_queue->sort(comparator);
  }
  if (second_queue != nullptr && !second_queue->empty()) {
    CHECK(second_queue->supports_sort())
        << "multi_slo_and_prio requires sortable request queue.";
    second_queue->sort(comparator);
  }

  // Adjust latency budget for prefill based on top request across both queues.
  if (for_prefill) {
    constexpr int32_t kSmallPositiveTimeMs = 2;
    if (min_remaining_time > constant_overhead + kSmallPositiveTimeMs) {
      latency_budget = min_remaining_time;
    } else {
      // Find the top request across both queues.
      std::shared_ptr<Request> top_request;
      if (first_queue != nullptr && !first_queue->empty()) {
        top_request = first_queue->top();
      }
      if (second_queue != nullptr && !second_queue->empty()) {
        if (top_request == nullptr ||
            comparator(second_queue->top(), top_request)) {
          top_request = second_queue->top();
        }
      }
      if (top_request != nullptr) {
        const int32_t top_remaining_time = top_request->get_remaining_time();
        if (top_remaining_time > constant_overhead + kSmallPositiveTimeMs) {
          latency_budget = top_remaining_time;
        } else {
          const auto& top_sequence = top_request->sequences()[0];
          latency_budget = top_sequence->estimated_latency() +
                           constant_overhead + kSmallPositiveTimeMs;
        }
      }
    }
  }
}

}  // namespace xllm

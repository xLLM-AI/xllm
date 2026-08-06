/* Copyright 2026 The xLLM Authors.

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

#include <cstdint>
#include <vector>

#include "common/metrics.h"
#include "core/framework/config/scheduler_config.h"
#include "framework/request/priority_comparator.h"
#include "scheduler/scheduler_policy.h"

namespace xllm {
void ShortRequestFirstPolicy::schedule(
    SchedulerState& state,
    ScheduleBudget& budget,
    std::vector<std::shared_ptr<Request>>& finished) {
  const SchedulerConfig& scheduler_config = SchedulerConfig::get_instance();
  const int32_t threshold = scheduler_config.short_request_first_threshold();
  std::vector<std::shared_ptr<Request>> waiting_requests;
  waiting_requests.reserve(state.prefill_queue.size());
  for (auto it = state.prefill_queue.begin(); it != state.prefill_queue.end();
       ++it) {
    waiting_requests.emplace_back(*it);
  }
  state.prefill_queue.sort(ShortRequestFirstComparator(
      threshold,
      select_short_request_first_promoted(
          waiting_requests,
          threshold,
          scheduler_config.short_request_first_long_max_wait_ms(),
          absl::Now())));
  PrefillFirstPolicy::schedule(state, budget, finished);
}

void ShortRequestFirstPolicy::report_metrics(const SchedulerState& state,
                                             double elapsed_seconds,
                                             size_t num_preempted_requests) {
  PrefillFirstPolicy::report_metrics(
      state, elapsed_seconds, num_preempted_requests);

  const SchedulerConfig& scheduler_config = SchedulerConfig::get_instance();
  const int32_t threshold = scheduler_config.short_request_first_threshold();
  size_t short_waiting = 0;
  size_t long_waiting = 0;
  for (auto it = state.prefill_queue.begin(); it != state.prefill_queue.end();
       ++it) {
    const ShortRequestFirstRequestClass request_class =
        classify_short_request_first(**it, threshold);
    if (request_class == ShortRequestFirstRequestClass::SHORT) {
      ++short_waiting;
    } else {
      ++long_waiting;
    }
  }
  GAUGE_SET(num_short_request_first_short_waiting, short_waiting);
  GAUGE_SET(num_short_request_first_long_waiting, long_waiting);
}

}  // namespace xllm

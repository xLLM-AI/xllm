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

#include <algorithm>
#include <cstdint>

#include "core/framework/config/scheduler_config.h"
#include "glog/logging.h"
#include "scheduler/scheduler_policy.h"

namespace xllm {
namespace {

// Sort rank for ShortRequestFirst ordering; lower ranks are scheduled first.
int32_t short_request_first_rank(ShortRequestFirstRequestClass request_class,
                                 const std::shared_ptr<Request>& request,
                                 const std::shared_ptr<Request>& promoted) {
  if (request_class == ShortRequestFirstRequestClass::IMMEDIATE) {
    return 0;
  }
  if (request_class == ShortRequestFirstRequestClass::LONG &&
      promoted != nullptr && request.get() == promoted.get()) {
    return 1;
  }
  if (request_class == ShortRequestFirstRequestClass::SHORT) {
    return 2;
  }
  return 3;
}

}  // namespace

ShortRequestFirstRequestClass classify_short_request_first(Request& request,
                                                           int32_t threshold) {
  CHECK(!request.sequences().empty());
  Sequence* sequence = request.sequences()[0].get();
  CHECK(sequence != nullptr);

  // Chunked prefill continuations that hold KV blocks are routed to the
  // chunk queue and scheduled ahead of the prefill queue, so the prefill
  // queue only contains fresh requests and preempted requests (whose KV has
  // already been deallocated). IMMEDIATE therefore means "preempted".
  if (request.preempted()) {
    return ShortRequestFirstRequestClass::IMMEDIATE;
  }
  if (sequence->num_prompt_tokens() <= static_cast<size_t>(threshold)) {
    return ShortRequestFirstRequestClass::SHORT;
  }
  return ShortRequestFirstRequestClass::LONG;
}

void sort_short_request_first_queue(RequestPriorityQueue& queue,
                                    int32_t threshold,
                                    double long_max_wait_ms,
                                    absl::Time now) {
  if (queue.empty()) {
    return;
  }
  CHECK(queue.supports_sort())
      << "ShortRequestFirst requires a sortable request queue.";

  // Identify the LONG request with the oldest creation time; it is the only
  // candidate for aging promotion ahead of waiting SHORT requests.
  std::shared_ptr<Request> oldest_long;
  bool has_short = false;
  bool has_long = false;
  for (auto it = queue.begin(); it != queue.end(); ++it) {
    const ShortRequestFirstRequestClass request_class =
        classify_short_request_first(**it, threshold);
    if (request_class == ShortRequestFirstRequestClass::SHORT) {
      has_short = true;
    } else if (request_class == ShortRequestFirstRequestClass::LONG) {
      has_long = true;
      if (oldest_long == nullptr ||
          (*it)->created_time() < oldest_long->created_time()) {
        oldest_long = *it;
      }
    }
  }

  std::shared_ptr<Request> promoted;
  if (long_max_wait_ms > 0.0 && has_short && has_long &&
      oldest_long != nullptr &&
      now - oldest_long->created_time() >=
          absl::Milliseconds(long_max_wait_ms)) {
    promoted = oldest_long;
  }

  queue.sort([threshold, promoted](const std::shared_ptr<Request>& a,
                                   const std::shared_ptr<Request>& b) {
    const ShortRequestFirstRequestClass class_a =
        classify_short_request_first(*a, threshold);
    const ShortRequestFirstRequestClass class_b =
        classify_short_request_first(*b, threshold);
    const int32_t rank_a = short_request_first_rank(class_a, a, promoted);
    const int32_t rank_b = short_request_first_rank(class_b, b, promoted);
    if (rank_a != rank_b) {
      return rank_a < rank_b;
    }
    return a->created_time() < b->created_time();
  });
}

void ShortRequestFirstPolicy::schedule(
    SchedulerState& state,
    ScheduleBudget& budget,
    std::vector<std::shared_ptr<Request>>& finished) {
  const SchedulerConfig& scheduler_config = SchedulerConfig::get_instance();
  sort_short_request_first_queue(
      state.prefill_queue,
      scheduler_config.short_request_first_threshold(),
      scheduler_config.short_request_first_long_max_wait_ms());
  PrefillFirstPolicy::schedule(state, budget, finished);
}

}  // namespace xllm

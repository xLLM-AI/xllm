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

#include "rate_limiter.h"

#include <gflags/gflags.h>

#include "common/global_flags.h"
#include "common/metrics.h"
#include "core/framework/config/service_config.h"

namespace xllm {

bool RateLimiter::is_limited() {
  const int32_t max =
      ::xllm::ServiceConfig::get_instance().max_concurrent_requests();
  int32_t expected = num_concurrent_requests_.load(std::memory_order_relaxed);
  while (true) {
    // Check if sleeping.
    if (expected == kSleeping) {
      return true;
    }
    // Check rate limit.
    if (max > 0 && expected >= max) {
      COUNTER_INC(server_request_total_limit);
      return true;
    }
    // Atomic check+increment. On CAS failure, `expected` is refreshed and we
    // retry (re-checking the sleep/limit conditions above with the new value).
    if (num_concurrent_requests_.compare_exchange_weak(
            expected,
            expected + 1,
            std::memory_order_acq_rel,
            std::memory_order_relaxed)) {
      GAUGE_SET(num_concurrent_requests, expected + 1);
      return false;
    }
  }
}

void RateLimiter::decrease_one_request() {
  num_concurrent_requests_.fetch_sub(1, std::memory_order_relaxed);
  GAUGE_SET(num_concurrent_requests,
            num_concurrent_requests_.load(std::memory_order_relaxed));
}

bool RateLimiter::try_set_sleeping() {
  int32_t expected = 0;
  // CAS: only succeed if current value is 0.
  return num_concurrent_requests_.compare_exchange_strong(
      expected, kSleeping, std::memory_order_acq_rel);
}

bool RateLimiter::try_wakeup() {
  int32_t expected = kSleeping;
  // CAS: only succeed if current value is kSleeping.
  return num_concurrent_requests_.compare_exchange_strong(
      expected, 0, std::memory_order_acq_rel);
}

}  // namespace xllm

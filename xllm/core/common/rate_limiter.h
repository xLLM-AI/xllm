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

#include <atomic>

namespace xllm {

class RateLimiter final {
 public:
  // Special value indicating sleep state.
  static constexpr int32_t kSleeping = INT32_MIN;

  RateLimiter() = default;

  ~RateLimiter() = default;

  // Returns true if request is rate-limited or sleeping.
  // If not limited and not sleeping, increments the counter.
  bool is_limited();

  void decrease_one_request();

  int32_t get_num_concurrent_requests() const {
    return num_concurrent_requests_.load(std::memory_order_relaxed);
  }

  // CAS: only succeeds if num_concurrent_requests == 0.
  // Sets to kSleeping on success. Returns true on success.
  bool try_set_sleeping();

  // CAS: only succeeds if num_concurrent_requests == kSleeping.
  // Sets to 0 on success. Returns true on success.
  bool try_wakeup();

  bool is_sleeping() const {
    return num_concurrent_requests_.load(std::memory_order_relaxed) ==
           kSleeping;
  }

 private:
  std::atomic<int32_t> num_concurrent_requests_{0};
};

}  // namespace xllm

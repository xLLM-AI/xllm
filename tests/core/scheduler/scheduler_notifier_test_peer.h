/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <mutex>
#include <thread>

#include "core/scheduler/scheduler_notifier.h"

namespace xllm {

class SchedulerNotifierTestPeer final {
 public:
  static bool wait_until_armed(
      const SchedulerNotifier& notifier,
      std::chrono::steady_clock::duration timeout = std::chrono::seconds(1)) {
    return wait_until(
        [&notifier] {
          std::lock_guard<std::mutex> lock(notifier.mutex_);
          return is_armed(notifier);
        },
        timeout);
  }

  static bool is_armed(const SchedulerNotifier& notifier) {
    return (notifier.state_.load(std::memory_order_acquire) &
            SchedulerNotifier::kWaitArmed) != 0;
  }

  static std::unique_lock<std::mutex> lock_wait_mutex(
      const SchedulerNotifier& notifier) {
    return std::unique_lock<std::mutex>(notifier.mutex_);
  }

  static bool wait_until_generation_changes(
      const SchedulerNotifier& notifier,
      uint64_t observed_generation,
      std::chrono::steady_clock::duration timeout = std::chrono::seconds(1)) {
    return wait_until(
        [&notifier, observed_generation] {
          return notifier.state_.load(std::memory_order_acquire) /
                     SchedulerNotifier::kGenerationIncrement !=
                 observed_generation;
        },
        timeout);
  }

  static bool wait_until_stopped(
      const SchedulerNotifier& notifier,
      std::chrono::steady_clock::duration timeout = std::chrono::seconds(1)) {
    return wait_until(
        [&notifier] {
          return (notifier.state_.load(std::memory_order_acquire) &
                  SchedulerNotifier::kStopRequested) != 0;
        },
        timeout);
  }

  // Set up the counter boundary before starting any producers or waiter.
  static void prepare_generation_wraparound(SchedulerNotifier& notifier) {
    constexpr uint64_t kLastGenerationState =
        std::numeric_limits<uint64_t>::max() -
        (SchedulerNotifier::kGenerationIncrement - 1);
    notifier.state_.store(kLastGenerationState, std::memory_order_release);
  }

 private:
  template <typename Predicate>
  static bool wait_until(Predicate&& predicate,
                         std::chrono::steady_clock::duration timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
      if (predicate()) {
        return true;
      }
      std::this_thread::yield();
    }
    return predicate();
  }
};

}  // namespace xllm

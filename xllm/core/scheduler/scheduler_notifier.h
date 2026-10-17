/* Copyright 2026 The xLLM Authors.

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

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace xllm {

// Coordinates one scheduler waiter with multiple event producers. Producers
// publish work before notify(); the scheduler takes a snapshot before scanning
// for work, then waits with that generation if the scan finds nothing ready.
class SchedulerNotifier final {
 public:
  struct Snapshot {
    uint64_t generation;
    bool stop_requested;
  };

  Snapshot snapshot() const {
    const uint64_t state = state_.load(std::memory_order_acquire);
    return {state / kGenerationIncrement, (state & kStopRequested) != 0};
  }

  // Records every event without taking the mutex while the scheduler is busy.
  // Concurrent events share one notification for an armed waiter. The return
  // value reports whether notify_one() was called, not whether work exists.
  bool notify() {
    uint64_t state = state_.load(std::memory_order_acquire);
    while (!state_.compare_exchange_weak(
        state,
        (state + kGenerationIncrement) & ~kWaitArmed,
        std::memory_order_acq_rel,
        std::memory_order_acquire)) {
    }
    if ((state & kWaitArmed) == 0) {
      return false;
    }
    {
      // Wait until the waiter has either observed the event or atomically
      // unlocked the mutex and entered the CV wait before sending the signal.
      std::lock_guard<std::mutex> lock(mutex_);
    }
    condition_.notify_one();
    return true;
  }

  void request_stop() {
    uint64_t state = state_.load(std::memory_order_acquire);
    while (!state_.compare_exchange_weak(
        state,
        ((state + kGenerationIncrement) | kStopRequested) & ~kWaitArmed,
        std::memory_order_acq_rel,
        std::memory_order_acquire)) {
    }
    {
      // Use the same CV handshake as notify(), even if already disarmed.
      std::lock_guard<std::mutex> lock(mutex_);
    }
    condition_.notify_all();
  }

  // Returns whether stop was requested. An event received since the snapshot
  // skips the mutex and waiting. Otherwise the atomic registration and the CV
  // handshake prevent lost wakeups while the scheduler transitions into
  // waiting.
  bool wait_for(uint64_t observed_generation,
                std::chrono::nanoseconds timeout) {
    uint64_t state = state_.load(std::memory_order_acquire);
    if (state / kGenerationIncrement != observed_generation ||
        (state & kStopRequested) != 0) {
      return (state & kStopRequested) != 0;
    }

    std::unique_lock<std::mutex> lock(mutex_);
    if (!state_.compare_exchange_strong(state,
                                        state | kWaitArmed,
                                        std::memory_order_acq_rel,
                                        std::memory_order_acquire)) {
      return (state & kStopRequested) != 0;
    }
    condition_.wait_for(lock, timeout, [this, observed_generation] {
      const auto current = snapshot();
      return current.generation != observed_generation ||
             current.stop_requested;
    });
    state = state_.fetch_and(~kWaitArmed, std::memory_order_acq_rel);
    return (state & kStopRequested) != 0;
  }

 private:
  friend class SchedulerNotifierTestPeer;

  // Keep registration, stop and the 62-bit generation in one atomic state so
  // publishing an event and claiming its notification cannot race with arming.
  static constexpr uint64_t kWaitArmed = 1;
  static constexpr uint64_t kStopRequested = 2;
  static constexpr uint64_t kGenerationIncrement = 4;

  std::atomic<uint64_t> state_{0};
  mutable std::mutex mutex_;
  std::condition_variable condition_;
};

}  // namespace xllm

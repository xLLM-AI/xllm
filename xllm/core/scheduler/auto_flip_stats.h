/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <absl/time/clock.h>
#include <absl/time/time.h>

#include <atomic>
#include <deque>
#include <mutex>

namespace xllm {

// AutoFlipStats maintains a rolling window of recently arrived requests, and
// exposes the aggregate signals that AutoFlipController feeds to the CP<->DP
// mode-switch decision:
//   * long_ratio: fraction of requests in the window whose prompt_tokens
//     >= long_prompt_threshold.
//   * total_in_window: sample count in the window (guards against
//     making decisions on too few samples).
//
// Callers push a sample via `record_request(prompt_tokens)` from every
// `add_request` code path. Read-side callers (AutoFlipController tick)
// query `snapshot(long_prompt_threshold, window_ms)` to prune old samples
// and fold the aggregate.
//
// Thread safety: internal mutex; both add and query paths are cheap
// (deque push_back + prune-at-front). Not intended to be a hot-loop
// scheduling structure; the request add rate is O(QPS), the tick rate is
// O(1/s).
class AutoFlipStats {
 public:
  AutoFlipStats() = default;

  // Called from ContinuousScheduler::add_request. `prompt_tokens` is the
  // number of prompt tokens for the incoming request's first sequence.
  void record_request(int64_t prompt_tokens) {
    const absl::Time now = absl::Now();
    std::lock_guard<std::mutex> lock(mu_);
    samples_.push_back({now, prompt_tokens});
  }

  struct Snapshot {
    double long_ratio;         // 0..1
    uint64_t total_in_window;  // number of samples in the window
    uint64_t long_in_window;   // number of long-prompt samples
  };

  // Prune samples older than `window_ms` and compute the aggregate.
  Snapshot snapshot(int64_t long_prompt_threshold, int64_t window_ms) {
    const absl::Time now = absl::Now();
    const absl::Time cutoff = now - absl::Milliseconds(window_ms);

    std::lock_guard<std::mutex> lock(mu_);
    // Prune samples older than cutoff. Deque is FIFO on arrival time so
    // this is amortized O(1) per record.
    while (!samples_.empty() && samples_.front().arrival_time < cutoff) {
      samples_.pop_front();
    }

    Snapshot s{};
    s.total_in_window = samples_.size();
    s.long_in_window = 0;
    for (const auto& sample : samples_) {
      if (sample.prompt_tokens >= long_prompt_threshold) {
        ++s.long_in_window;
      }
    }
    s.long_ratio =
        (s.total_in_window > 0)
            ? static_cast<double>(s.long_in_window) / s.total_in_window
            : 0.0;
    return s;
  }

 private:
  struct Sample {
    absl::Time arrival_time;
    int64_t prompt_tokens;
  };
  mutable std::mutex mu_;
  std::deque<Sample> samples_;
};

}  // namespace xllm

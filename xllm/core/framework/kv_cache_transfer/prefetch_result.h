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

#include <glog/logging.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace xllm {

// Collects block-granular Mooncake prefetch results from every TP worker.
// A block is publishable to the Host PrefixCache only when every worker reports
// a hit for the corresponding transfer info.
class PrefetchResult final {
 public:
  PrefetchResult(size_t worker_count,
                 size_t block_count,
                 size_t batch_size = 1,
                 int64_t stream_idle_timeout_ms = -1)
      : worker_results_(worker_count,
                        std::vector<uint8_t>(block_count, /*value=*/0)),
        worker_completed_(worker_count, /*value=*/0),
        remaining_workers_(worker_count),
        block_count_(block_count),
        batch_size_(batch_size),
        stream_idle_timeout_ms_(stream_idle_timeout_ms) {
    CHECK_GT(worker_count, 0u);
    CHECK_GT(batch_size, 0u);
    CHECK(stream_idle_timeout_ms == -1 || stream_idle_timeout_ms > 0);
  }

  size_t worker_count() const { return worker_results_.size(); }
  size_t block_count() const { return block_count_; }
  size_t batch_size() const { return batch_size_; }
  int64_t stream_idle_timeout_ms() const { return stream_idle_timeout_ms_; }

  bool request_stop() {
    return !stop_requested_.exchange(true, std::memory_order_acq_rel);
  }

  bool stop_requested() const {
    return stop_requested_.load(std::memory_order_acquire);
  }

  bool set_batch_result(size_t worker_index,
                        size_t offset,
                        const std::vector<uint8_t>& hits) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (worker_index >= worker_results_.size() || offset > block_count_ ||
        hits.size() > block_count_ - offset ||
        worker_completed_[worker_index] != 0) {
      return false;
    }

    std::vector<uint8_t>& worker_result = worker_results_[worker_index];
    std::transform(hits.begin(),
                   hits.end(),
                   worker_result.begin() + static_cast<std::ptrdiff_t>(offset),
                   [](uint8_t hit) { return hit == 0 ? 0 : 1; });
    return true;
  }

  void mark_worker_completed(size_t worker_index, bool worker_ok) {
    std::lock_guard<std::mutex> lock(mutex_);
    CHECK_LT(worker_index, worker_results_.size());
    if (worker_completed_[worker_index] != 0) {
      return;
    }
    if (!worker_ok) {
      std::fill(worker_results_[worker_index].begin(),
                worker_results_[worker_index].end(),
                /*value=*/0);
    }
    worker_completed_[worker_index] = 1;
    const size_t previous =
        remaining_workers_.fetch_sub(1, std::memory_order_acq_rel);
    CHECK_GT(previous, 0u);
  }

  bool completed() const {
    return remaining_workers_.load(std::memory_order_acquire) == 0;
  }

  std::vector<uint8_t> merged_hits() const {
    CHECK(completed()) << "Cannot merge an unfinished prefetch result.";
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<uint8_t> merged(block_count_, /*value=*/1);
    for (const std::vector<uint8_t>& worker_result : worker_results_) {
      for (size_t i = 0; i < block_count_; ++i) {
        merged[i] =
            static_cast<uint8_t>(merged[i] != 0 && worker_result[i] != 0);
      }
    }
    return merged;
  }

 private:
  mutable std::mutex mutex_;
  std::vector<std::vector<uint8_t>> worker_results_;
  std::vector<uint8_t> worker_completed_;
  std::atomic<size_t> remaining_workers_;
  size_t block_count_ = 0;
  size_t batch_size_ = 1;
  int64_t stream_idle_timeout_ms_ = -1;
  std::atomic<bool> stop_requested_{false};
};

}  // namespace xllm

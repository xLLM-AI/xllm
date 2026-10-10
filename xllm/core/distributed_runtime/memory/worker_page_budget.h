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

#include <cstddef>
#include <cstdint>
#include <vector>

namespace xllm {

// Tracks page capacity independently for each worker. Callers provide their
// own synchronization when the ledger is shared between threads.
class WorkerPageBudget final {
 public:
  WorkerPageBudget() = default;
  explicit WorkerPageBudget(size_t total_pages, int32_t worker_count);

  void reset(size_t total_pages, int32_t worker_count);

  size_t total_pages() const { return total_pages_; }
  int32_t worker_count() const {
    return static_cast<int32_t>(used_pages_.size());
  }

  bool try_reserve(int32_t start_worker,
                   int32_t end_worker,
                   size_t pages_per_worker);
  bool try_reserve(const std::vector<size_t>& pages_per_worker);

  void release(int32_t start_worker,
               int32_t end_worker,
               size_t pages_per_worker);
  void release(const std::vector<size_t>& pages_per_worker);

  size_t free_pages(int32_t worker) const;
  size_t min_free_pages(int32_t start_worker, int32_t end_worker) const;
  std::vector<size_t> free_pages_snapshot() const;
  const std::vector<size_t>& used_pages() const { return used_pages_; }

 private:
  bool is_valid_range(int32_t start_worker, int32_t end_worker) const;

  size_t total_pages_ = 0;
  std::vector<size_t> used_pages_;
};

}  // namespace xllm

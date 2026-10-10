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

#include "core/distributed_runtime/memory/worker_page_budget.h"

#include <glog/logging.h>

#include <algorithm>

namespace xllm {

WorkerPageBudget::WorkerPageBudget(size_t total_pages, int32_t worker_count) {
  reset(total_pages, worker_count);
}

void WorkerPageBudget::reset(size_t total_pages, int32_t worker_count) {
  if (worker_count < 0) {
    total_pages_ = 0;
    used_pages_.clear();
    return;
  }
  total_pages_ = total_pages;
  used_pages_.assign(static_cast<size_t>(worker_count), 0);
}

bool WorkerPageBudget::is_valid_range(int32_t start_worker,
                                      int32_t end_worker) const {
  return start_worker >= 0 && end_worker >= start_worker &&
         end_worker <= worker_count();
}

bool WorkerPageBudget::try_reserve(int32_t start_worker,
                                   int32_t end_worker,
                                   size_t pages_per_worker) {
  if (!is_valid_range(start_worker, end_worker)) {
    return false;
  }
  for (int32_t worker = start_worker; worker < end_worker; ++worker) {
    if (free_pages(worker) < pages_per_worker) {
      return false;
    }
  }
  for (int32_t worker = start_worker; worker < end_worker; ++worker) {
    used_pages_[static_cast<size_t>(worker)] += pages_per_worker;
  }
  return true;
}

bool WorkerPageBudget::try_reserve(
    const std::vector<size_t>& pages_per_worker) {
  if (pages_per_worker.size() != used_pages_.size()) {
    return false;
  }
  for (size_t worker = 0; worker < pages_per_worker.size(); ++worker) {
    if (free_pages(static_cast<int32_t>(worker)) < pages_per_worker[worker]) {
      return false;
    }
  }
  for (size_t worker = 0; worker < pages_per_worker.size(); ++worker) {
    used_pages_[worker] += pages_per_worker[worker];
  }
  return true;
}

void WorkerPageBudget::release(int32_t start_worker,
                               int32_t end_worker,
                               size_t pages_per_worker) {
  if (start_worker < 0 || end_worker <= start_worker) {
    return;
  }
  const int32_t first_worker =
      std::min(start_worker, static_cast<int32_t>(used_pages_.size()));
  const int32_t last_worker =
      std::min(end_worker, static_cast<int32_t>(used_pages_.size()));
  for (int32_t worker = first_worker; worker < last_worker; ++worker) {
    size_t& used_pages = used_pages_[static_cast<size_t>(worker)];
    if (pages_per_worker > used_pages) {
      LOG(WARNING) << "Worker " << worker
                   << " pages underflow during release: used=" << used_pages
                   << ", requested=" << pages_per_worker;
    }
    used_pages =
        pages_per_worker >= used_pages ? 0 : used_pages - pages_per_worker;
  }
}

void WorkerPageBudget::release(const std::vector<size_t>& pages_per_worker) {
  if (pages_per_worker.size() != used_pages_.size()) {
    LOG(WARNING) << "Ignoring worker page release with "
                 << pages_per_worker.size() << " entries for "
                 << used_pages_.size() << " workers";
  }
  const size_t count = std::min(pages_per_worker.size(), used_pages_.size());
  for (size_t worker = 0; worker < count; ++worker) {
    if (pages_per_worker[worker] > used_pages_[worker]) {
      LOG(WARNING) << "Worker " << worker
                   << " pages underflow during release: used="
                   << used_pages_[worker]
                   << ", requested=" << pages_per_worker[worker];
    }
    used_pages_[worker] = pages_per_worker[worker] >= used_pages_[worker]
                              ? 0
                              : used_pages_[worker] - pages_per_worker[worker];
  }
}

size_t WorkerPageBudget::free_pages(int32_t worker) const {
  if (worker < 0 || worker >= worker_count()) {
    return 0;
  }
  const size_t used_pages = used_pages_[static_cast<size_t>(worker)];
  return used_pages <= total_pages_ ? total_pages_ - used_pages : 0;
}

size_t WorkerPageBudget::min_free_pages(int32_t start_worker,
                                        int32_t end_worker) const {
  CHECK(is_valid_range(start_worker, end_worker));
  if (start_worker == end_worker) {
    return total_pages_;
  }
  size_t min_free = total_pages_;
  for (int32_t worker = start_worker; worker < end_worker; ++worker) {
    min_free = std::min(min_free, free_pages(worker));
  }
  return min_free;
}

std::vector<size_t> WorkerPageBudget::free_pages_snapshot() const {
  std::vector<size_t> result;
  result.reserve(used_pages_.size());
  for (int32_t worker = 0; worker < worker_count(); ++worker) {
    result.push_back(free_pages(worker));
  }
  return result;
}

}  // namespace xllm

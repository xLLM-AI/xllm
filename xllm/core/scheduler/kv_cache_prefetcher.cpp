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

#include "core/scheduler/kv_cache_prefetcher.h"

#include <glog/logging.h>

#include <utility>

#include "core/framework/block/kv_cache_manager.h"
#include "core/framework/request/request.h"
#include "core/scheduler/scheduler_notifier.h"

namespace xllm {

KVCachePrefetcher::KVCachePrefetcher(
    KVCacheManager& manager,
    std::mutex& admission_mutex,
    std::shared_ptr<SchedulerNotifier> notifier,
    std::function<bool(std::shared_ptr<Request>)> enqueue_ready_request)
    : manager_(manager),
      admission_mutex_(admission_mutex),
      notifier_(std::move(notifier)),
      enqueue_ready_request_(std::move(enqueue_ready_request)) {
  manager_.set_prefetch_completion_callback(
      [notifier = notifier_]() { notifier->notify(); });
}

KVCachePrefetcher::~KVCachePrefetcher() {
  CHECK_EQ(pending_requests_.load(std::memory_order_acquire), 0u)
      << "KVCachePrefetcher destroyed with pending prefetch callbacks";
}

void KVCachePrefetcher::add_request(std::shared_ptr<Request> request) {
  pending_requests_.fetch_add(1, std::memory_order_relaxed);
  VLOG(1) << "[Mooncake][AdmissionPending] request=" << request->request_id();
  admissions_.emplace_back(std::move(request));
  notifier_->notify();
}

size_t KVCachePrefetcher::num_pending_requests() const {
  return pending_requests_.load(std::memory_order_relaxed);
}

void KVCachePrefetcher::drain() {
  manager_.drain_prefetch_completions();
  drain_admissions();
  manager_.drain_prefetch_completions();
  drain_completed_prefetches();
}

void KVCachePrefetcher::shutdown() {
  manager_.set_prefetch_completion_callback({});
  {
    std::lock_guard<std::mutex> lock(admission_mutex_);
    for (const std::shared_ptr<Request>& request : admissions_) {
      request->set_cancel();
    }
    const size_t unissued = admissions_.size();
    admissions_.clear();
    const size_t previous =
        pending_requests_.fetch_sub(unissued, std::memory_order_acq_rel);
    CHECK_GE(previous, unissued);
  }
  manager_.drain_prefetch_completions();
  drain_completed_prefetches();
  CHECK_EQ(pending_requests_.load(std::memory_order_acquire), 0u)
      << "KVCachePrefetcher shut down with pending prefetch callbacks";
}

void KVCachePrefetcher::drain_admissions() {
  std::deque<std::shared_ptr<Request>> requests;
  {
    std::lock_guard<std::mutex> lock(admission_mutex_);
    requests.swap(admissions_);
  }

  for (std::shared_ptr<Request>& request : requests) {
    if (request->finished() || request->cancelled()) {
      const size_t previous =
          pending_requests_.fetch_sub(1, std::memory_order_relaxed);
      CHECK_GT(previous, 0u);
      continue;
    }

    manager_.prefetch_from_storage(
        std::move(request), [this](std::shared_ptr<Request> completed) {
          std::lock_guard<std::mutex> lock(admission_mutex_);
          completed_prefetches_.emplace_back(std::move(completed));
        });
  }
}

void KVCachePrefetcher::drain_completed_prefetches() {
  std::deque<std::shared_ptr<Request>> completed;
  {
    std::lock_guard<std::mutex> lock(admission_mutex_);
    completed.swap(completed_prefetches_);
  }

  for (std::shared_ptr<Request>& request : completed) {
    const bool cancelled = request->finished() || request->cancelled();
    {
      // Admission observes the queue size and reservation count together.
      std::lock_guard<std::mutex> lock(admission_mutex_);
      if (!cancelled) {
        CHECK(enqueue_ready_request_(request))
            << "Reserved request queue slot disappeared before prefetch "
               "completed.";
      }
      const size_t previous =
          pending_requests_.fetch_sub(1, std::memory_order_relaxed);
      CHECK_GT(previous, 0u);
    }
    VLOG(1) << (cancelled ? "[Mooncake][AdmissionCancelled] request="
                          : "[Mooncake][AdmissionReady] request=")
            << request->request_id();
  }
}

}  // namespace xllm

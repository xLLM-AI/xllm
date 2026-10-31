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
#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>

namespace xllm {

class KVCacheManager;
class Request;
class SchedulerNotifier;

// Prefetches KV cache from storage before requests enter the ready queue,
// reserving their queue slots. Created only for cache managers that support
// storage prefetch.
class KVCachePrefetcher final {
 public:
  KVCachePrefetcher(
      KVCacheManager& manager,
      std::mutex& admission_mutex,
      std::shared_ptr<SchedulerNotifier> notifier,
      std::function<bool(std::shared_ptr<Request>)> enqueue_ready_request);
  ~KVCachePrefetcher();

  // Caller holds admission_mutex and has checked ready-queue capacity together
  // with num_pending_requests().
  void add_request(std::shared_ptr<Request> request);

  size_t num_pending_requests() const;

  // Called on the scheduler thread to issue and complete storage prefetches.
  void drain();

  // Must run before the scheduler's ready queue and admission mutex are torn
  // down. Cancels unissued requests and drains callbacks already completed.
  void shutdown();

 private:
  KVCacheManager& manager_;
  std::mutex& admission_mutex_;
  std::shared_ptr<SchedulerNotifier> notifier_;
  std::function<bool(std::shared_ptr<Request>)> enqueue_ready_request_;
  std::atomic<size_t> pending_requests_{0};
  std::deque<std::shared_ptr<Request>> admissions_;
  std::deque<std::shared_ptr<Request>> completed_prefetches_;
};

}  // namespace xllm

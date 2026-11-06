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

#include "core/kv_cache/transfer/kv_transfer_completion.h"

#include <folly/futures/HeapTimekeeper.h>
#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <utility>

namespace xllm {
namespace {

constexpr std::chrono::seconds kKVTransferWaitTimeout{60};
constexpr std::chrono::seconds kKVTransferDrainTimeout{1};

}  // namespace

KVTransferCompletion::KVTransferCompletion()
    : KVTransferCompletion(kKVTransferWaitTimeout) {}

KVTransferCompletion::KVTransferCompletion(
    std::chrono::milliseconds wait_timeout)
    : wait_timeout_(wait_timeout) {
  CHECK_GT(wait_timeout_.count(), 0) << "wait timeout must be positive";
}

KVTransferCompletion::~KVTransferCompletion() {
  CHECK(futures_.empty())
      << "pending KV transfers must finish before source blocks are released";
  CHECK(tracker_ == nullptr || !tracker_->has_pending())
      << "pending KV transfer callbacks must finish before source blocks are "
         "released";
}

void KVTransferCompletion::add(
    folly::SemiFuture<std::vector<KVTransferTaskResult>> future,
    std::vector<std::string> fallback_request_ids) {
  if (tracker_ == nullptr) {
    tracker_ = std::make_unique<KVTransferTracker>();
  }
  std::shared_ptr<KVTransferTracker::Completion> transfer_completion =
      tracker_->track();
  future = std::move(future).deferEnsure(
      [transfer_completion = std::move(transfer_completion)]() mutable {
        transfer_completion.reset();
      });
  futures_.push_back({std::move(future), std::move(fallback_request_ids)});
}

std::unordered_set<std::string> KVTransferCompletion::wait() {
  if (waited_) {
    return failed_request_ids_;
  }
  if (futures_.empty()) {
    waited_ = true;
    return {};
  }

  std::vector<folly::SemiFuture<std::vector<KVTransferTaskResult>>> futures;
  futures.reserve(futures_.size());
  for (PendingTransfer& pending : futures_) {
    futures.emplace_back(std::move(pending.future));
  }
  static folly::HeapTimekeeper timekeeper;
  folly::SemiFuture<std::vector<folly::Try<std::vector<KVTransferTaskResult>>>>
      completion = folly::collectAll(std::move(futures))
                       .within(wait_timeout_,
                               static_cast<folly::Timekeeper*>(&timekeeper));
  folly::Try<std::vector<folly::Try<std::vector<KVTransferTaskResult>>>>
      completion_result = std::move(completion).getTry();
  const bool timed_out = !completion_result.hasValue();
  if (!timed_out) {
    const std::vector<folly::Try<std::vector<KVTransferTaskResult>>>& results =
        completion_result.value();
    for (size_t index = 0; index < results.size(); ++index) {
      const folly::Try<std::vector<KVTransferTaskResult>>& result =
          results[index];
      if (!result.hasValue()) {
        LOG(ERROR) << "KV cache transfer future failed: "
                   << result.exception().what();
        failed_request_ids_.insert(futures_[index].fallback_request_ids.begin(),
                                   futures_[index].fallback_request_ids.end());
        continue;
      }
      for (const KVTransferTaskResult& task : result.value()) {
        if (task.error_code != KVTransferErrorCode::NONE) {
          failed_request_ids_.insert(task.request_ids.begin(),
                                     task.request_ids.end());
        }
      }
    }
  }
  if (timed_out) {
    for (const PendingTransfer& pending : futures_) {
      failed_request_ids_.insert(pending.fallback_request_ids.begin(),
                                 pending.fallback_request_ids.end());
    }
    // A timeout only changes the request result. The transfer continuation may
    // still be reading source KV blocks, so drain it before releasing futures.
    // Keep a short configured timeout from turning normal async cleanup into an
    // immediate fatal path, while still bounding a permanently stuck transfer.
    const std::chrono::milliseconds drain_timeout =
        std::max(wait_timeout_,
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     kKVTransferDrainTimeout));
    if (!tracker_->wait_for(drain_timeout)) {
      LOG(FATAL) << "KV cache push did not finish after timeout";
    }
  } else {
    tracker_->wait();
  }
  futures_.clear();
  waited_ = true;
  return failed_request_ids_;
}

class KVTransferTracker::State final {
 public:
  void start() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++pending_transfers_;
  }

  void finish() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      CHECK_GT(pending_transfers_, 0u);
      --pending_transfers_;
    }
    completion_cv_.notify_all();
  }

  bool has_pending() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pending_transfers_ > 0;
  }

  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    completion_cv_.wait(lock, [this]() { return pending_transfers_ == 0; });
  }

  bool wait_for(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return completion_cv_.wait_for(
        lock, timeout, [this]() { return pending_transfers_ == 0; });
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable completion_cv_;
  size_t pending_transfers_ = 0;
};

KVTransferTracker::Completion::Completion(std::shared_ptr<State> state)
    : state_(std::move(state)) {
  CHECK(state_ != nullptr);
  state_->start();
}

KVTransferTracker::Completion::~Completion() { state_->finish(); }

KVTransferTracker::KVTransferTracker() : state_(std::make_shared<State>()) {}

KVTransferTracker::~KVTransferTracker() { wait(); }

std::shared_ptr<KVTransferTracker::Completion> KVTransferTracker::track() {
  return std::shared_ptr<Completion>(new Completion(state_));
}

bool KVTransferTracker::has_pending() const { return state_->has_pending(); }

void KVTransferTracker::wait() { state_->wait(); }

bool KVTransferTracker::wait_for(std::chrono::milliseconds timeout) {
  CHECK_GT(timeout.count(), 0) << "wait timeout must be positive";
  return state_->wait_for(timeout);
}

}  // namespace xllm

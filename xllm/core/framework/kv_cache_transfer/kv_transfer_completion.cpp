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

#include "core/framework/kv_cache_transfer/kv_transfer_completion.h"

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
}

void KVTransferCompletion::add(folly::SemiFuture<bool> future) {
  futures_.emplace_back(std::move(future));
}

bool KVTransferCompletion::wait() {
  if (futures_.empty()) {
    return true;
  }

  std::vector<folly::Try<bool>> results =
      folly::collectAll(futures_).get(wait_timeout_);
  futures_.clear();
  return std::all_of(
      results.begin(), results.end(), [](const folly::Try<bool>& result) {
        return result.hasValue() && result.value();
      });
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

}  // namespace xllm

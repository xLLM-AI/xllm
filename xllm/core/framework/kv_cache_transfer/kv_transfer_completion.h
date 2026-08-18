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

#include <folly/futures/Future.h>

#include <chrono>
#include <memory>
#include <vector>

namespace xllm {

// Owns asynchronous KV transfers until every transfer reaches a terminal
// state. Source KV blocks must not be released while this object is pending.
class KVTransferCompletion final {
 public:
  KVTransferCompletion();
  explicit KVTransferCompletion(std::chrono::milliseconds wait_timeout);
  ~KVTransferCompletion();

  KVTransferCompletion(const KVTransferCompletion&) = delete;
  KVTransferCompletion& operator=(const KVTransferCompletion&) = delete;
  KVTransferCompletion(KVTransferCompletion&&) = delete;
  KVTransferCompletion& operator=(KVTransferCompletion&&) = delete;

  void add(folly::SemiFuture<bool> future);

  // Waits until all owned transfers finish. Returns false when any transfer
  // reports failure or completes with an exception.
  bool wait();

 private:
  std::chrono::milliseconds wait_timeout_;
  std::vector<folly::SemiFuture<bool>> futures_;
};

// Tracks callbacks that retain KV block managers or blocks. A Completion
// token keeps one callback pending; releasing the last token unblocks wait().
// Destruction waits so owners can declare this as their last member and make
// callback lifetime a construction invariant instead of custom teardown code.
class KVTransferTracker final {
 private:
  class State;

 public:
  class Completion final {
   public:
    ~Completion();

    Completion(const Completion&) = delete;
    Completion& operator=(const Completion&) = delete;
    Completion(Completion&&) = delete;
    Completion& operator=(Completion&&) = delete;

   private:
    friend class KVTransferTracker;

    explicit Completion(std::shared_ptr<State> state);

    std::shared_ptr<State> state_;
  };

  KVTransferTracker();
  ~KVTransferTracker();

  KVTransferTracker(const KVTransferTracker&) = delete;
  KVTransferTracker& operator=(const KVTransferTracker&) = delete;
  KVTransferTracker(KVTransferTracker&&) = delete;
  KVTransferTracker& operator=(KVTransferTracker&&) = delete;

  std::shared_ptr<Completion> track();
  bool has_pending() const;
  void wait();

 private:
  std::shared_ptr<State> state_;
};

}  // namespace xllm

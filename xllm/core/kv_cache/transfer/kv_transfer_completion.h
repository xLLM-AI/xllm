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

#include <folly/futures/Future.h>

#include <array>
#include <chrono>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/kv_cache/transfer/kv_transfer_types.h"

namespace xllm {

class Device;
class ProcessGroup;
struct TransferKVInfo;

std::vector<std::string> reduce_failed_request_ids(
    const std::unordered_set<std::string>& local_failed_request_ids,
    const std::vector<std::string>& canonical_request_ids,
    const std::array<ProcessGroup*, 2>& groups,
    const Device& device);

class KVTransferTracker;

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

  void add(folly::SemiFuture<std::vector<KVTransferTaskResult>> future,
           std::vector<std::string> fallback_request_ids);

  // Waits until all owned transfers finish and returns the failed request IDs.
  std::unordered_set<std::string> wait();

 private:
  std::chrono::milliseconds wait_timeout_;
  std::unique_ptr<KVTransferTracker> tracker_;
  struct PendingTransfer {
    folly::SemiFuture<std::vector<KVTransferTaskResult>> future;
    std::vector<std::string> fallback_request_ids;
  };
  std::vector<PendingTransfer> futures_;
  std::unordered_set<std::string> failed_request_ids_;
  bool waited_ = false;
};

// Waits for KV push completion and reduces failed request IDs across the
// replicated TP/CP groups when PUSH transfer is enabled.
std::vector<std::string> finalize_kv_push_failures(
    KVTransferCompletion& kv_transfers,
    const std::vector<TransferKVInfo>& transfer_kv_infos,
    bool enable_kv_push,
    const std::array<ProcessGroup*, 2>& groups,
    const Device& device);

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
  bool wait_for(std::chrono::milliseconds timeout);

 private:
  std::shared_ptr<State> state_;
};

}  // namespace xllm

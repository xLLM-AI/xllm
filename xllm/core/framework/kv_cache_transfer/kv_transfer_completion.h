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

}  // namespace xllm

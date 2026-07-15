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

}  // namespace xllm

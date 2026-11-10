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

#include <folly/executors/InlineExecutor.h>
#include <glog/logging.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <utility>

#include "core/framework/parallel_state/process_group.h"
#include "core/platform/device.h"

namespace xllm {
namespace {

constexpr std::chrono::seconds kKVTransferWaitTimeout{60};

}  // namespace

KVTransferCompletion::KVTransferCompletion(
    folly::SemiFuture<std::vector<std::string>> future,
    std::vector<std::string> request_ids)
    : future_(std::move(future).via(&folly::InlineExecutor::instance())),
      request_ids_(std::move(request_ids)) {}

KVTransferCompletion::~KVTransferCompletion() {
  CHECK(!future_.valid())
      << "pending KV transfers must finish before source blocks are released";
}

std::vector<std::string> KVTransferCompletion::wait(
    const std::array<ProcessGroup*, 2>& groups,
    const Device& device) {
  if (!future_.valid()) {
    return {};
  }
  auto future = std::move(future_);
  // Timed SemiFuture waits detach deferred callbacks; retain an eager future.
  const bool timed_out = !future.wait(kKVTransferWaitTimeout).isReady();
  if (timed_out) {
    CHECK(future.wait(kKVTransferWaitTimeout).isReady())
        << "KV cache push did not finish after timeout";
  }
  auto result = std::move(future).getTry();
  if (result.hasException()) {
    LOG(ERROR) << "KV cache transfer future failed: "
               << result.exception().what();
  }
  std::vector<std::string> canonical_request_ids = std::move(request_ids_);
  if (canonical_request_ids.empty()) {
    return {};
  }
  std::sort(canonical_request_ids.begin(), canonical_request_ids.end());
  const bool all_failed = timed_out || !result.hasValue();
  auto& local_failed_request_ids =
      all_failed ? canonical_request_ids : result.value();
  if (!all_failed) {
    std::sort(local_failed_request_ids.begin(), local_failed_request_ids.end());
  }
  const auto locally_failed = [&](const std::string& request_id) {
    return all_failed || std::binary_search(local_failed_request_ids.begin(),
                                            local_failed_request_ids.end(),
                                            request_id);
  };

  // Only TP/CP replicas share a request set; DP groups carry unrelated batches.
  std::array<ProcessGroup*, 2> active_groups{};
  size_t num_groups = 0;
  for (ProcessGroup* group : groups) {
    if (group != nullptr && group->world_size() > 1 &&
        std::find(active_groups.begin(),
                  active_groups.begin() + num_groups,
                  group) == active_groups.begin() + num_groups) {
      active_groups[num_groups++] = group;
    }
  }
  if (num_groups == 0) {
    std::vector<std::string> failed_request_ids;
    for (const std::string& request_id : canonical_request_ids) {
      if (locally_failed(request_id)) {
        failed_request_ids.emplace_back(request_id);
      }
    }
    return failed_request_ids;
  }

  torch::Tensor bitmap_host = torch::zeros(
      {static_cast<int64_t>(canonical_request_ids.size())}, torch::kInt32);
  auto* bitmap_data = bitmap_host.data_ptr<int32_t>();
  for (size_t index = 0; index < canonical_request_ids.size(); ++index) {
    bitmap_data[index] = locally_failed(canonical_request_ids[index]);
  }
  torch::Tensor bitmap = bitmap_host.to(device);
  for (size_t group_index = 0; group_index < num_groups; ++group_index) {
    active_groups[group_index]->allreduce(bitmap);
  }
  bitmap_host = bitmap.to(torch::kCPU);
  bitmap_data = bitmap_host.data_ptr<int32_t>();
  std::vector<std::string> failed_request_ids;
  for (size_t index = 0; index < canonical_request_ids.size(); ++index) {
    if (bitmap_data[index] != 0) {
      failed_request_ids.emplace_back(canonical_request_ids[index]);
    }
  }
  return failed_request_ids;
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

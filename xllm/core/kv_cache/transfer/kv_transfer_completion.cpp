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
#include <xxhash.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstring>
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

  const size_t kPayloadMetadataSize = 2;
  const size_t payload_size =
      kPayloadMetadataSize + canonical_request_ids.size();
  torch::Tensor payload_host = torch::zeros(
      {static_cast<int64_t>(payload_size)},
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  int64_t* payload_data = payload_host.data_ptr<int64_t>();
  payload_data[0] = static_cast<int64_t>(canonical_request_ids.size());
  uint64_t request_ids_hash = 0;
  for (const auto& request_id : canonical_request_ids) {
    request_ids_hash = XXH3_64bits_withSeed(
        request_id.data(), request_id.size(), request_ids_hash);
  }
  std::memcpy(&payload_data[1], &request_ids_hash, sizeof(request_ids_hash));
  for (size_t index = 0; index < canonical_request_ids.size(); ++index) {
    if (locally_failed(canonical_request_ids[index])) {
      payload_data[kPayloadMetadataSize + index] = 1;
    }
  }
  // Orthogonal TP/CP groups carry the cumulative bitmap forward so a failure
  // found by an earlier group also reaches ranks that only share the later
  // group. Metadata stays unchanged, so the existing payload is reused.
  torch::Tensor payload = payload_host.to(device);
  for (size_t group_index = 0; group_index < num_groups; ++group_index) {
    ProcessGroup* group = active_groups[group_index];
    torch::Tensor gathered =
        group->allgather_base_sync(payload).to(torch::kCPU);
    CHECK_EQ(gathered.numel(),
             static_cast<int64_t>(group->world_size() * payload.numel()))
        << "unexpected KV transfer request payload shape";
    const int64_t* gathered_data = gathered.data_ptr<int64_t>();
    for (int32_t rank = 0; rank < group->world_size(); ++rank) {
      const int64_t rank_offset = rank * payload.numel();
      CHECK_EQ(gathered_data[rank_offset], payload_data[0])
          << "KV transfer request count differs across reduction ranks";
      CHECK_EQ(gathered_data[rank_offset + 1], payload_data[1])
          << "KV transfer request IDs differ across reduction ranks";
      for (size_t index = 0; index < canonical_request_ids.size(); ++index) {
        if (gathered_data[rank_offset + kPayloadMetadataSize + index] > 0) {
          payload_data[kPayloadMetadataSize + index] = 1;
        }
      }
    }
    if (group_index + 1 < num_groups) {
      payload.copy_(payload_host);
    }
  }
  std::vector<std::string> failed_request_ids;
  for (size_t index = 0; index < canonical_request_ids.size(); ++index) {
    if (payload_data[kPayloadMetadataSize + index] > 0) {
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

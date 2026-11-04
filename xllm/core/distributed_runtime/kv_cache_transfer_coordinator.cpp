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

#include "core/distributed_runtime/kv_cache_transfer_coordinator.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstddef>
#include <utility>

#include "core/distributed_runtime/distributed_worker_manager.h"
#include "core/distributed_runtime/kv_transfer_topology.h"

namespace xllm {

KVCacheTransferCoordinator::KVCacheTransferCoordinator(
    Options options,
    std::shared_ptr<DistributedWorkerManager> distributed_worker_manager)
    : options_(std::move(options)),
      distributed_worker_manager_(std::move(distributed_worker_manager)) {
  CHECK_GT(options_.dp_size, 0);
  CHECK(distributed_worker_manager_ != nullptr);
}

bool KVCacheTransferCoordinator::pull_kv_blocks(
    int32_t src_dp_size,
    int32_t src_dp_rank,
    const std::vector<uint64_t>& src_cluster_ids,
    const std::vector<std::string>& src_addrs,
    int32_t dst_dp_rank,
    const std::vector<KVTransferMapping>& mappings) {
  if (src_addrs.size() != src_cluster_ids.size()) {
    LOG(ERROR) << "Source cache endpoint counts do not match.";
    return false;
  }
  const auto& worker_clients =
      distributed_worker_manager_->get_worker_clients();
  const auto routes =
      KVTransferTopology::get_pull_worker_routes(src_cluster_ids.size(),
                                                 src_dp_size,
                                                 src_dp_rank,
                                                 worker_clients.size(),
                                                 options_.dp_size,
                                                 dst_dp_rank);
  if (!routes.has_value()) {
    LOG(ERROR) << "Invalid or heterogeneous topology for KV cache PULL.";
    return false;
  }

  std::vector<bool> results;
  results.reserve(routes->size());
  // Complete every selected worker call before reporting a failed PULL.
  for (const KVWorkerRoute& route : *routes) {
    results.emplace_back(worker_clients[route.dst_rank]->pull_kv_blocks(
        src_cluster_ids[route.src_rank], src_addrs[route.src_rank], mappings));
  }
  return std::all_of(
      results.begin(), results.end(), [](bool result) { return result; });
}

std::vector<folly::SemiFuture<uint32_t>>
KVCacheTransferCoordinator::transfer_kv_blocks(
    uint32_t dp_rank,
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  const auto& worker_clients =
      distributed_worker_manager_->get_worker_clients();
  const auto workers = KVTransferTopology::get_dp_worker_range(
      worker_clients.size(), options_.dp_size, static_cast<int32_t>(dp_rank));
  CHECK(workers.has_value()) << "Invalid DP topology for KV cache transfer.";

  std::vector<folly::SemiFuture<uint32_t>> futures;
  futures.reserve(workers->count);
  for (size_t local_rank = 0; local_rank < workers->count; ++local_rank) {
    futures.emplace_back(
        worker_clients[workers->begin + local_rank]->transfer_kv_blocks(
            block_transfer_info));
  }
  return futures;
}

std::vector<folly::SemiFuture<uint32_t>>
KVCacheTransferCoordinator::transfer_kv_blocks(
    uint32_t dp_rank,
    uint64_t batch_id,
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  const auto& worker_clients =
      distributed_worker_manager_->get_worker_clients();
  const auto workers = KVTransferTopology::get_dp_worker_range(
      worker_clients.size(), options_.dp_size, static_cast<int32_t>(dp_rank));
  CHECK(workers.has_value()) << "Invalid DP topology for KV cache transfer.";
  std::vector<folly::SemiFuture<uint32_t>> futures;
  futures.reserve(workers->count);
  for (size_t local_rank = 0; local_rank < workers->count; ++local_rank) {
    futures.emplace_back(
        worker_clients[workers->begin + local_rank]->transfer_kv_blocks(
            batch_id, block_transfer_info));
  }
  return futures;
}

void KVCacheTransferCoordinator::prefetch_from_storage(
    uint32_t dp_rank,
    std::shared_ptr<const StoragePrefetchRequest> request,
    PrefetchResult::StopPredicate stop_requested,
    PrefetchResult::DoneCallback done) {
  CHECK(request != nullptr);
  CHECK(request->valid());
  const auto& worker_clients =
      distributed_worker_manager_->get_worker_clients();
  const auto workers = KVTransferTopology::get_dp_worker_range(
      worker_clients.size(), options_.dp_size, static_cast<int32_t>(dp_rank));
  CHECK(workers.has_value()) << "Invalid DP topology for storage prefetch.";
  const int64_t timeout_ms =
      options_.prefetch_timeout_ms == 0
          ? -1
          : static_cast<int64_t>(options_.prefetch_timeout_ms);
  auto result =
      std::make_shared<PrefetchResult>(workers->count,
                                       request->batch_end_unit_offsets,
                                       timeout_ms,
                                       std::move(stop_requested),
                                       std::move(done));
  for (size_t local_rank = 0; local_rank < workers->count; ++local_rank) {
    worker_clients[workers->begin + local_rank]->prefetch_from_storage(
        request, result, local_rank);
  }
}

}  // namespace xllm

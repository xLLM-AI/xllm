/* Copyright 2025-2026 The xLLM Authors.

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
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "core/common/macros.h"
#include "core/framework/speculative/speculative_profile_registry.h"
#include "core/runtime/options.h"
#include "core/runtime/worker_client.h"

namespace xllm {

class WorkerServer;
class ThreadPool;
class DistributedWorkerManagerTest;
class ModelMemoryControllerTest;
class KVCacheTransferCoordinatorTest;
class DisaggPDSchedulerTestPeer;
class SchedulerMetricsTestPeer;

// Owns worker servers, cluster rendezvous, and worker clients. Multiple engines
// can share the manager to use the same distributed workers.
class DistributedWorkerManager final {
 public:
  explicit DistributedWorkerManager(const runtime::Options& options);
  ~DistributedWorkerManager();

  // Only the leader node creates clients for all global ranks.
  const std::vector<std::shared_ptr<WorkerClient>>& get_worker_clients() const {
    return worker_clients_;
  }

  // Broadcasts the predictor to every worker in global rank order.
  bool set_speculative_validate_time_predictor(
      const SpeculativeProfileRegistry::ValidateTimePredictor& predictor);

  // Appends the transport endpoints in global worker rank order.
  void get_cache_info(std::vector<uint64_t>& cluster_ids,
                      std::vector<std::string>& addrs,
                      std::vector<uint16_t>& ports) const;

  // Queries every worker before waiting, then returns bytes in worker rank
  // order.
  std::vector<int64_t> get_active_activation_memory() const;

  bool link_cluster(const std::vector<uint64_t>& cluster_ids,
                    const std::vector<std::string>& addrs,
                    const std::vector<uint16_t>& ports,
                    int32_t src_dp_size,
                    int32_t src_kv_split_size = 1);
  bool unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                      const std::vector<std::string>& addrs,
                      const std::vector<uint16_t>& ports,
                      int32_t src_dp_size,
                      int32_t src_kv_split_size = 1);

  // Each worker links to the remote weight-transfer address at its global rank.
  bool link_p2p(const std::vector<std::string>& remote_addrs);
  bool unlink_p2p(const std::vector<std::string>& remote_addrs);

 private:
  friend class DistributedWorkerManagerTest;
  friend class ModelMemoryControllerTest;
  friend class KVCacheTransferCoordinatorTest;
  friend class DisaggPDSchedulerTestPeer;
  friend class SchedulerMetricsTestPeer;

  DISALLOW_COPY_AND_ASSIGN(DistributedWorkerManager);

  explicit DistributedWorkerManager(
      std::vector<std::shared_ptr<WorkerClient>> worker_clients);

  void start_worker_servers(const runtime::Options& options,
                            const std::string& master_node_addr);
  void connect_worker_clients(const runtime::Options& options,
                              const std::string& master_node_addr);
  void ensure_link_threadpool();
  void wait_for_worker_servers() const;
  void start_health_checks();

  std::string collective_server_name_;
  std::vector<std::shared_ptr<WorkerClient>> worker_clients_;
  // Shared engines must not interleave cluster and weight-transfer operations.
  std::mutex link_mutex_;
  std::unique_ptr<ThreadPool> link_threadpool_;
  bool runtime_resources_started_ = false;
  // Worker threads borrow these flags; keep them alive until servers stop.
  std::vector<std::atomic<bool>> worker_ready_;
  std::vector<std::unique_ptr<WorkerServer>> worker_servers_;
};

}  // namespace xllm

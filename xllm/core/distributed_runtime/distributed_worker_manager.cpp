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

#include "core/distributed_runtime/distributed_worker_manager.h"

#include <folly/futures/Future.h>
#include <glog/logging.h>

#include <chrono>
#include <cstdint>
#include <thread>
#include <unordered_set>
#include <utility>

#include "core/common/health_check_manager.h"
#include "core/distributed_runtime/collective_service.h"
#include "core/distributed_runtime/comm_channel.h"
#include "core/distributed_runtime/remote_worker.h"
#include "core/distributed_runtime/shm_channel.h"
#include "core/distributed_runtime/worker_server.h"
#include "core/framework/config/service_config.h"
#include "core/framework/parallel_state/parallel_args.h"
#if defined(USE_CUDA) || defined(USE_MLU) || defined(USE_DCU)
#include "core/platform/numa_utils.h"
#endif
#include "core/util/net.h"
#include "core/util/threadpool.h"
#include "server/xllm_server_registry.h"

namespace xllm {

DistributedWorkerManager::DistributedWorkerManager(
    const runtime::Options& options)
    : collective_server_name_("CollectiveServer" +
                              std::to_string(options.server_idx())),
      worker_ready_(options.devices().size()) {
  const std::string master_node_addr = options.master_node_addr().value_or("");
  CHECK(!master_node_addr.empty()) << "master_node_addr is empty.";
  CHECK(!options.devices().empty()) << "At least one device is required";
  CHECK_GE(options.nnodes(), 1) << "At least one node is required";
  CHECK_GE(options.node_rank(), 0) << "Node rank must >= 0.";
  CHECK_LT(options.node_rank(), options.nnodes())
      << "Node rank must be less than the number of nodes.";
  CHECK_GT(options.dp_size(), 0) << "Data parallel size must be positive.";
  const int32_t world_size =
      static_cast<int32_t>(options.devices().size()) * options.nnodes();
  CHECK_EQ(world_size % options.dp_size(), 0)
      << "Global world size must be divisible by dp size.";

  for (auto& ready : worker_ready_) {
    ready.store(false, std::memory_order_relaxed);
  }

  start_worker_servers(options, master_node_addr);
  if (options.node_rank() == 0) {
    connect_worker_clients(options, master_node_addr);
    start_health_checks();
  }
  wait_for_worker_servers();
  runtime_resources_started_ = true;
}

DistributedWorkerManager::DistributedWorkerManager(
    std::vector<std::shared_ptr<WorkerClient>> worker_clients)
    : worker_clients_(std::move(worker_clients)) {}

DistributedWorkerManager::~DistributedWorkerManager() {
  std::lock_guard<std::mutex> lock(link_mutex_);
  // Drain connection work while the worker clients and servers are still alive.
  link_threadpool_.reset();
  if (!runtime_resources_started_) {
    return;
  }
  HealthCheckManager::instance().stop_health_check_thread();

  XllmServer* collective_server =
      ServerRegistry::get_instance().get_server(collective_server_name_);
  if (collective_server != nullptr) {
    collective_server->stop();
    ServerRegistry::get_instance().unregister_server(collective_server_name_);
  }

  for (const auto& server : worker_servers_) {
    server->stop();
  }
}

void DistributedWorkerManager::get_cache_info(
    std::vector<uint64_t>& cluster_ids,
    std::vector<std::string>& addrs,
    std::vector<uint16_t>& ports) const {
  cluster_ids.reserve(cluster_ids.size() + worker_clients_.size());
  addrs.reserve(addrs.size() + worker_clients_.size());
  ports.reserve(ports.size() + worker_clients_.size());
  for (const auto& worker : worker_clients_) {
    uint64_t cluster_id = 0;
    std::string addr;
    uint16_t port = 0;
    worker->get_cache_info(cluster_id, addr, port);
    cluster_ids.emplace_back(cluster_id);
    addrs.emplace_back(std::move(addr));
    ports.emplace_back(port);
  }
}

std::vector<int64_t> DistributedWorkerManager::get_active_activation_memory()
    const {
  std::vector<folly::SemiFuture<int64_t>> futures;
  futures.reserve(worker_clients_.size());
  for (const auto& worker : worker_clients_) {
    futures.emplace_back(worker->get_active_activation_memory_async());
  }

  auto results = folly::collectAll(futures).get();
  std::vector<int64_t> active_activation_memories;
  active_activation_memories.reserve(worker_clients_.size());
  for (const auto& result : results) {
    active_activation_memories.emplace_back(result.value());
  }
  return active_activation_memories;
}

void DistributedWorkerManager::ensure_link_threadpool() {
  if (link_threadpool_ != nullptr) {
    return;
  }
  link_threadpool_ = std::make_unique<ThreadPool>(
      /*num_threads=*/worker_clients_.size(),
      /*cpu_binding=*/false,
      /*pool_name=*/"DistributedWorkerManager.link");
}

bool DistributedWorkerManager::link_cluster(
    const std::vector<uint64_t>& cluster_ids,
    const std::vector<std::string>& addrs,
    const std::vector<uint16_t>& ports,
    int32_t src_dp_size,
    int32_t src_kv_split_size) {
  std::lock_guard<std::mutex> lock(link_mutex_);
  if (worker_clients_.empty()) {
    LOG(ERROR) << "Only the leader node can link clusters.";
    return false;
  }
  const int32_t src_world_size = static_cast<int32_t>(cluster_ids.size());
  if (src_dp_size <= 0 || src_kv_split_size <= 0 || src_world_size <= 0 ||
      src_world_size % src_dp_size != 0 ||
      (src_world_size / src_dp_size) % src_kv_split_size != 0 ||
      addrs.size() != cluster_ids.size() ||
      ports.size() != cluster_ids.size()) {
    LOG(ERROR) << "Invalid source topology for cache layout negotiation.";
    return false;
  }
  ensure_link_threadpool();

  // Every D worker negotiates with all P workers. Logical shard intersection
  // determines which edges carry bytes; modulo TP routing cannot represent
  // non-integer TP changes or KV-head replication.
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(worker_clients_.size());
  for (size_t worker_rank = 0; worker_rank < worker_clients_.size();
       ++worker_rank) {
    folly::Promise<bool> promise;
    auto future = promise.getSemiFuture();
    link_threadpool_->schedule([this,
                                promise = std::move(promise),
                                worker_rank,
                                &cluster_ids,
                                &addrs,
                                &ports]() mutable {
      promise.setValue(worker_clients_[worker_rank]->link_cluster(
          cluster_ids, addrs, ports));
    });
    futures.emplace_back(std::move(future));
  }

  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      LOG(ERROR) << "Link cluster failed.";
      return false;
    }
  }
  return true;
}

bool DistributedWorkerManager::unlink_cluster(
    const std::vector<uint64_t>& cluster_ids,
    const std::vector<std::string>& addrs,
    const std::vector<uint16_t>& ports,
    int32_t src_dp_size,
    int32_t src_kv_split_size) {
  std::lock_guard<std::mutex> lock(link_mutex_);
  if (worker_clients_.empty()) {
    LOG(ERROR) << "Only the leader node can unlink clusters.";
    return false;
  }
  const int32_t src_world_size = static_cast<int32_t>(cluster_ids.size());
  if (src_dp_size <= 0 || src_kv_split_size <= 0 || src_world_size <= 0 ||
      src_world_size % src_dp_size != 0 ||
      (src_world_size / src_dp_size) % src_kv_split_size != 0 ||
      addrs.size() != cluster_ids.size() ||
      ports.size() != cluster_ids.size()) {
    LOG(ERROR) << "Invalid source topology for cache unlink.";
    return false;
  }
  ensure_link_threadpool();

  // Symmetric to link_cluster: close every negotiated source edge.
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(worker_clients_.size());
  for (size_t worker_rank = 0; worker_rank < worker_clients_.size();
       ++worker_rank) {
    folly::Promise<bool> promise;
    auto future = promise.getSemiFuture();
    link_threadpool_->schedule([this,
                                promise = std::move(promise),
                                worker_rank,
                                &cluster_ids,
                                &addrs,
                                &ports]() mutable {
      promise.setValue(worker_clients_[worker_rank]->unlink_cluster(
          cluster_ids, addrs, ports));
    });
    futures.emplace_back(std::move(future));
  }

  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      LOG(ERROR) << "Unlink cluster failed.";
      return false;
    }
  }
  return true;
}

bool DistributedWorkerManager::link_p2p(
    const std::vector<std::string>& remote_addrs) {
  std::lock_guard<std::mutex> lock(link_mutex_);
  if (worker_clients_.empty()) {
    LOG(ERROR) << "Only the leader node can link P2P addresses.";
    return false;
  }
  if (remote_addrs.size() != worker_clients_.size()) {
    LOG(ERROR) << "remote_addrs size " << remote_addrs.size()
               << " != worker_clients_num " << worker_clients_.size();
    return false;
  }
  ensure_link_threadpool();

  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(worker_clients_.size());
  for (size_t worker_rank = 0; worker_rank < worker_clients_.size();
       ++worker_rank) {
    folly::Promise<bool> promise;
    auto future = promise.getSemiFuture();
    link_threadpool_->schedule([this,
                                promise = std::move(promise),
                                worker_rank,
                                &remote_addrs]() mutable {
      promise.setValue(
          worker_clients_[worker_rank]->link_p2p(remote_addrs[worker_rank]));
    });
    futures.emplace_back(std::move(future));
  }

  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      LOG(ERROR) << "Link P2P failed.";
      return false;
    }
  }
  return true;
}

bool DistributedWorkerManager::unlink_p2p(
    const std::vector<std::string>& remote_addrs) {
  std::lock_guard<std::mutex> lock(link_mutex_);
  if (worker_clients_.empty()) {
    LOG(ERROR) << "Only the leader node can unlink P2P addresses.";
    return false;
  }
  if (remote_addrs.size() != worker_clients_.size()) {
    LOG(ERROR) << "remote_addrs size " << remote_addrs.size()
               << " != worker_clients_num " << worker_clients_.size();
    return false;
  }
  ensure_link_threadpool();

  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(worker_clients_.size());
  for (size_t worker_rank = 0; worker_rank < worker_clients_.size();
       ++worker_rank) {
    folly::Promise<bool> promise;
    auto future = promise.getSemiFuture();
    link_threadpool_->schedule([this,
                                promise = std::move(promise),
                                worker_rank,
                                &remote_addrs]() mutable {
      promise.setValue(
          worker_clients_[worker_rank]->unlink_p2p(remote_addrs[worker_rank]));
    });
    futures.emplace_back(std::move(future));
  }

  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      LOG(ERROR) << "Unlink P2P failed.";
      return false;
    }
  }
  return true;
}

namespace {
std::unique_ptr<CommChannel> create_channel(const std::string& worker_addr,
                                            int32_t rank,
                                            int32_t dp_local_tp_size,
                                            const runtime::Options& options) {
  std::unique_ptr<CommChannel> channel;

  if (net::extract_ip(options.master_node_addr().value_or("")) ==
          net::extract_ip(worker_addr) &&
      options.enable_shm()) {
    const int32_t dp_group = rank / dp_local_tp_size;
    const bool is_driver = rank % dp_local_tp_size == 0;
    channel = std::make_unique<ShmChannel>(dp_group, rank, is_driver, options);
  } else {
    channel = std::make_unique<CommChannel>();
  }

  channel->init_brpc(worker_addr);

  return channel;
}

#if defined(USE_CUDA) || defined(USE_MLU) || defined(USE_DCU)
void setup_numa_affinity_and_isolation(
    const runtime::Options& options,
    std::vector<int32_t>& device_numa_nodes,
    std::vector<bool>& force_spawn_for_numa_isolation) {
  const auto& devices = options.devices();

  device_numa_nodes.assign(devices.size(), -1);
  force_spawn_for_numa_isolation.assign(devices.size(), false);

  std::unordered_set<int32_t> unique_numa_nodes;
  for (size_t i = 0; i < devices.size(); ++i) {
    device_numa_nodes[i] = numa::get_device_numa_node(devices[i].index());
    if (device_numa_nodes[i] >= 0) {
      unique_numa_nodes.insert(device_numa_nodes[i]);
    }

    LOG(INFO) << "NUMA mapping: local rank " << i << ", device "
              << devices[i].index() << " -> NUMA node " << device_numa_nodes[i];
  }

  int32_t engine_numa_node = -1;
  for (const int32_t numa_node : device_numa_nodes) {
    if (numa_node >= 0) {
      engine_numa_node = numa_node;
      break;
    }
  }

  if (engine_numa_node >= 0) {
    if (numa::bind_process_to_numa_node(engine_numa_node) != 0) {
      LOG(WARNING) << "Failed to pin engine process to NUMA node "
                   << engine_numa_node
                   << ", fallback to per-worker affinity only";
    }

    if (unique_numa_nodes.size() > 1) {
      for (size_t i = 0; i < devices.size(); ++i) {
        force_spawn_for_numa_isolation[i] =
            (device_numa_nodes[i] >= 0 &&
             device_numa_nodes[i] != engine_numa_node);
      }
      LOG(INFO) << "Detected multi-NUMA local devices. Workers outside NUMA "
                << engine_numa_node
                << " will be spawned as isolated processes to avoid engine "
                   "process cross-NUMA spanning";
    }
  }
}
#endif

}  // namespace

void DistributedWorkerManager::start_worker_servers(
    const runtime::Options& options,
    const std::string& master_node_addr) {
  const auto& devices = options.devices();

#if defined(USE_CUDA) || defined(USE_MLU) || defined(USE_DCU)
  std::vector<int32_t> device_numa_nodes;
  std::vector<bool> force_spawn_for_numa_isolation;
  setup_numa_affinity_and_isolation(
      options, device_numa_nodes, force_spawn_for_numa_isolation);
#endif

  // Each node uses the same device count; global ranks are node-major.
  const int32_t each_node_ranks = static_cast<int32_t>(devices.size());
  const int32_t world_size = each_node_ranks * options.nnodes();
  const int32_t base_rank = options.node_rank() * each_node_ranks;
  const int32_t dp_size = options.dp_size();
  const int32_t cp_size = options.cp_size();
  const int32_t ep_size = options.ep_size();
  /* TODO(CP): support smem  + CP */
  const int32_t dp_local_tp_size = world_size / dp_size;

  const std::string& model_backend = options.backend();
  if (model_backend == "dit") {
    const int32_t tp_size = options.tp_size();
    const int32_t sp_size = options.sp_size();
    const int32_t cfg_size = options.cfg_size();
    const int32_t vae_size = options.vae_size();
    const int32_t text_encoder_tp_size = options.text_encoder_tp_size();
    LOG(INFO) << "Multi-node serving world_size = " << world_size
              << ", each_node_ranks = " << each_node_ranks
              << ", current node rank = " << options.node_rank()
              << ", nnodes = " << options.nnodes() << ", dp_size = " << dp_size
              << ", tp_size = " << tp_size << ", sp_size = " << sp_size
              << ", cfg_size = " << cfg_size << ", vae_size = " << vae_size
              << ", text_encoder_tp_size = " << text_encoder_tp_size;
  } else {
    LOG(INFO) << "Multi-node serving world_size = " << world_size
              << ", each_node_ranks = " << each_node_ranks
              << ", current node rank = " << options.node_rank()
              << ", nnodes = " << options.nnodes() << ", dp_size = " << dp_size
              << ", cp_size = " << cp_size << ", ep_size = " << ep_size
              << ", tp_size = " << dp_local_tp_size;
  }

  runtime::Options worker_server_options = options;
  worker_server_options.world_size(world_size);
  WorkerType worker_type("LLM");
  if (model_backend == "llm") {
    if (options.task_type() == "generate") {
      worker_type = WorkerType::LLM;
    } else if (options.task_type() == "embed") {
      worker_type = WorkerType::ELM;
    } else {
      LOG(FATAL) << "Unsupported " << options.task_type()
                 << " for llm model backend";
    }
  } else if (model_backend == "vlm") {
    if (options.task_type() == "generate") {
      worker_type = WorkerType::VLM;
    } else if (options.task_type() == "embed") {
      worker_type = WorkerType::EVLM;
    } else if (options.task_type() == "mm_embed") {
      worker_type = WorkerType::MMEVLM;
    } else {
      LOG(FATAL) << "Unsupported " << options.task_type()
                 << " for vlm model backend";
    }
  } else if (model_backend == "rec") {
    worker_type = WorkerType::REC;
  } else if (model_backend == "dit") {
    worker_type = WorkerType::DIT;
  } else {
    LOG(FATAL) << "Unsupported " << model_backend << " in multi-node.";
  }
  // Launch every local server before waiting for cluster registration.
  worker_servers_.reserve(devices.size());
  for (int32_t i = 0; i < each_node_ranks; ++i) {
    const int32_t rank = i + base_rank;
    worker_server_options.server_idx(rank);

#if defined(USE_CUDA) || defined(USE_MLU) || defined(USE_DCU)
    const bool use_spawn_worker =
        (options.enable_offline_inference() && i > 0) ||
        force_spawn_for_numa_isolation[i];
    if (force_spawn_for_numa_isolation[i]) {
      LOG(INFO) << "Force spawn worker for local rank " << i << " (device "
                << devices[i].index() << ", NUMA " << device_numa_nodes[i]
                << ") to keep each process within a single NUMA region";
    }
#else
    const bool use_spawn_worker = options.enable_offline_inference() && i > 0;
#endif
    ParallelArgs parallel_args(
        rank, world_size, dp_size, cp_size, nullptr, ep_size);

    worker_servers_.emplace_back(
        std::make_unique<WorkerServer>(i,
                                       master_node_addr,
                                       worker_ready_[i],
                                       parallel_args,
                                       devices[i],
                                       worker_server_options,
                                       worker_type,
                                       use_spawn_worker));
  }
}

void DistributedWorkerManager::connect_worker_clients(
    const runtime::Options& options,
    const std::string& master_node_addr) {
  const auto& devices = options.devices();
  const int32_t each_node_ranks = static_cast<int32_t>(devices.size());
  const int32_t world_size = each_node_ranks * options.nnodes();
  const int32_t dp_local_tp_size = world_size / options.dp_size();
  auto collective_service = std::make_shared<CollectiveService>(world_size);
  XllmServer* collective_server =
      ServerRegistry::get_instance().register_server(collective_server_name_);
  CHECK(collective_server->start(
      collective_service, master_node_addr, collective_server_name_))
      << "Failed to start collective server on address: " << master_node_addr;

  const auto worker_addrs_map = collective_service->wait();
  worker_clients_.reserve(world_size);
  for (int32_t rank = 0; rank < world_size; ++rank) {
    const auto it = worker_addrs_map.find(rank);
    CHECK(it != worker_addrs_map.end())
        << "Not all workers connected to engine server. Missing rank " << rank;
    // TODO(CP): support shared memory with CP.
    auto channel = create_channel(it->second, rank, dp_local_tp_size, options);
    worker_clients_.emplace_back(std::make_shared<RemoteWorker>(
        rank, it->second, devices[rank % each_node_ranks], std::move(channel)));
  }
}

void DistributedWorkerManager::start_health_checks() {
  for (const auto& worker_client : worker_clients_) {
    auto* remote_worker = dynamic_cast<RemoteWorker*>(worker_client.get());
    if (remote_worker == nullptr) {
      continue;
    }
    const int32_t rank = remote_worker->global_rank();
    HealthCheckManager::instance().register_health_check(
        rank, [remote_worker]() { return remote_worker->check_health(); });
  }
  HealthCheckManager::instance().start_health_check_thread(
      ServiceConfig::get_instance().health_check_interval_ms());
  LOG(INFO) << "Started cluster health check thread";
}

void DistributedWorkerManager::wait_for_worker_servers() const {
  for (const auto& ready : worker_ready_) {
    while (!ready.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
}

}  // namespace xllm

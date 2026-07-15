/* Copyright 2025-2026 The xLLM Authors.

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

#include "distributed_runtime/dist_manager.h"

#include <glog/logging.h>

#include "comm_channel.h"
#include "common/health_check_manager.h"
#include "core/framework/config/service_config.h"
#include "distributed_runtime/collective_service.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "framework/parallel_state/process_group.h"
#if defined(USE_CUDA) || defined(USE_MLU) || defined(USE_DCU)
#include "platform/numa_utils.h"
#endif
#include "remote_worker.h"
#include "runtime/forward_shared_memory_manager.h"
#include "runtime/llm_worker_impl.h"
#include "server/xllm_server_registry.h"
#include "shm_channel.h"
#include "util/net.h"
namespace xllm {

DistManager::DistManager(const runtime::Options& options)
    : server_name_("CollectiveServer") {
  auto master_node_addr = options.master_node_addr().value_or("");
  if (!master_node_addr.empty()) {
    server_name_.append(std::to_string(options.server_idx()));
    setup_multi_node_workers(options, master_node_addr);
  } else {
    LOG(FATAL) << "master_node_addr is empty.";
  }
}

DistManager::~DistManager() {
  // Stop health check
  HealthCheckManager::instance().stop_health_check_thread();

  XllmServer* collective_server =
      ServerRegistry::get_instance().get_server(server_name_);
  if (collective_server != nullptr) {
    collective_server->stop();

    ServerRegistry::get_instance().unregister_server(server_name_);
  }

  for (size_t i = 0; i < servers_.size(); ++i) {
    servers_[i]->stop();
  }
}

namespace {
std::unique_ptr<CommChannel> create_channel(const std::string& worker_addrs,
                                            int r,
                                            int dp_local_tp_size,
                                            const runtime::Options& options) {
  std::unique_ptr<CommChannel> channel;

  if (net::extract_ip(options.master_node_addr().value_or("")) ==
          net::extract_ip(worker_addrs) &&
      options.enable_shm()) {
    // create shared memory manager for local rank
    bool is_driver = false;
    int dp_group = r / dp_local_tp_size;
    if (r % dp_local_tp_size == 0) {
      is_driver = true;
    }
    channel = std::make_unique<ShmChannel>(dp_group, r, is_driver, options);
  } else {
    channel = std::make_unique<CommChannel>();
  }

  channel->init_brpc(worker_addrs);

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

  std::set<int32_t> unique_numa_nodes;
  for (size_t i = 0; i < devices.size(); ++i) {
    device_numa_nodes[i] = numa::get_device_numa_node(devices[i].index());
    if (device_numa_nodes[i] >= 0) {
      unique_numa_nodes.insert(device_numa_nodes[i]);
    }

    LOG(INFO) << "NUMA mapping: local rank " << i << ", device "
              << devices[i].index() << " -> NUMA node " << device_numa_nodes[i];
  }

  int32_t engine_numa_node = -1;
  for (auto numa_node : device_numa_nodes) {
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

void DistManager::setup_multi_node_workers(
    const runtime::Options& options,
    const std::string& master_node_addr) {
  const auto& devices = options.devices();

#if defined(USE_CUDA) || defined(USE_MLU) || defined(USE_DCU)
  std::vector<int32_t> device_numa_nodes;
  std::vector<bool> force_spawn_for_numa_isolation;
  setup_numa_affinity_and_isolation(
      options, device_numa_nodes, force_spawn_for_numa_isolation);
#endif

  // Process/Thread Worker Mode, we use it in multi-nodes serving.

  // Here, we assume that all node use same index devices. That is, if we set
  // device='1,2,3,4' and nnodes=2, then both machine nodes will use the
  // devices '1,2,3,4'. Therefore, the total world size is 2 * 4 = 8. This
  // means that each of the two nodes will utilize four devices (specifically
  // devices 1, 2, 3, and 4), resulting in a total of 8 devices being used
  // across the entire distributed setup.

  // To maintain interface consistency, we have implemented a new WorkerImpl
  // class. In this class, we create processes, initialize NCCL ProcessGroup,
  // set up GRPC servers, and so on.

  std::vector<std::atomic<bool>> dones(1);
  dones[0].store(false, std::memory_order_relaxed);

  CHECK_GE(options.nnodes(), 1) << "At least one node is required";
  CHECK_GE(options.node_rank(), 0) << "Node rank must >= 0.";
  // Multi-process serving: each process runs exactly one worker. `devices` is
  // the visible-device pool (all cards each process can see, required so ATB
  // can build the cross-card HCCL comm group); this process binds the single
  // card selected by its node_rank. World size is therefore the node count.
  const int32_t each_node_ranks = 1;
  const int32_t world_size = options.nnodes();
  const int32_t base_rank = options.node_rank();
  const int32_t local_device_index = options.node_rank();
  CHECK_LT(local_device_index, static_cast<int32_t>(devices.size()))
      << "node_rank " << local_device_index << " exceeds the number of visible "
      << "devices " << devices.size()
      << ". Ensure *_VISIBLE_DEVICES exposes all cards used across processes.";
  const torch::Device local_device = devices[local_device_index];
  const int32_t dp_size = options.dp_size();
  const int32_t cp_size = options.cp_size();
  const int32_t ep_size = options.ep_size();
  /* TODO(CP): support smem  + CP */
  const int32_t dp_local_tp_size = world_size / dp_size;

  const auto& model_backend = options.backend();
  if (model_backend == "dit") {
    const int32_t tp_size = options.tp_size();
    const int32_t sp_size = options.sp_size();
    const int32_t cfg_size = options.cfg_size();
    const int32_t vae_size = options.vae_size();
    LOG(INFO) << "Multi-node serving world_size = " << world_size
              << ", each_node_ranks = " << each_node_ranks
              << ", current node rank = " << options.node_rank()
              << ", nnodes = " << options.nnodes() << ", dp_size = " << dp_size
              << ", tp_size = " << tp_size << ", sp_size = " << sp_size
              << ", cfg_size = " << cfg_size << ", vae_size = " << vae_size;
  } else {
    LOG(INFO) << "Multi-node serving world_size = " << world_size
              << ", each_node_ranks = " << each_node_ranks
              << ", current node rank = " << options.node_rank()
              << ", nnodes = " << options.nnodes() << ", dp_size = " << dp_size
              << ", cp_size = " << cp_size << ", ep_size = " << ep_size
              << ", tp_size = " << dp_local_tp_size;
  }

  CHECK_EQ((world_size % dp_size), 0)
      << "Global world size must be divisible by dp size in multi-node "
         "serving mode.";

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
  // create the single local worker for this process
  {
    const int32_t i = 0;
    const int32_t rank = base_rank;
    worker_server_options.server_idx(rank);

#if defined(USE_CUDA) || defined(USE_MLU) || defined(USE_DCU)
    bool use_spawn_worker = force_spawn_for_numa_isolation[local_device_index];
    if (use_spawn_worker) {
      LOG(INFO) << "Force spawn worker for node_rank " << local_device_index
                << " (device " << local_device.index() << ", NUMA "
                << device_numa_nodes[local_device_index]
                << ") to keep each process within a single NUMA region";
    }
#else
    bool use_spawn_worker = false;
#endif
    ParallelArgs parallel_args(
        rank, world_size, dp_size, cp_size, nullptr, ep_size);

    servers_.emplace_back(std::make_unique<WorkerServer>(i,
                                                         master_node_addr,
                                                         // done,
                                                         dones[0],
                                                         parallel_args,
                                                         local_device,
                                                         worker_server_options,
                                                         worker_type,
                                                         use_spawn_worker));
  }

  // Master node need to wait all workers done
  if (options.node_rank() == 0) {
    // if dp_size equals 1, use global process group directly
    // if dp_size equals world_size, distributed communication is not required
    auto dp_local_process_group_num =
        (dp_size > 1 && dp_size < world_size) ? dp_size : 0;

    // create collective server to sync all workers.
    std::shared_ptr<CollectiveService> collective_service =
        std::make_shared<CollectiveService>(
            dp_local_process_group_num, world_size, devices[0].index());
    XllmServer* collective_server =
        ServerRegistry::get_instance().register_server(server_name_);
    if (!collective_server->start(
            collective_service, master_node_addr, server_name_)) {
      LOG(ERROR) << "failed to start collective server on address: "
                 << master_node_addr;
      return;
    }

    auto worker_addrs_map = collective_service->wait();

    // check if all workers connected
    // and then create worker clients
    CHECK_LE(static_cast<size_t>(world_size), devices.size())
        << "world_size " << world_size << " exceeds the visible-device pool "
        << devices.size() << "; each global rank maps to one visible card.";
    for (size_t r = 0; r < world_size; ++r) {
      if (worker_addrs_map.find(r) == worker_addrs_map.end()) {
        LOG(FATAL) << "Not all worker connect to engine server. Miss rank is "
                   << r;
        return;
      }
      /* TODO(CP): support smem  + CP */
      auto channel =
          create_channel(worker_addrs_map[r], r, dp_local_tp_size, options);
      // Global rank r runs on the process whose node_rank == r, which binds the
      // card devices[r] from the shared visible-device pool.
      worker_clients_.emplace_back(std::make_unique<RemoteWorker>(
          r, worker_addrs_map[r], devices[r], std::move(channel)));
    }

    // Register health check for each worker and start background health check
    for (auto& worker_client : worker_clients_) {
      auto* remote_worker = dynamic_cast<RemoteWorker*>(worker_client.get());
      if (remote_worker) {
        int rank = remote_worker->global_rank();
        HealthCheckManager::instance().register_health_check(
            rank, [remote_worker]() { return remote_worker->check_health(); });
      }
    }
    // Start background health check thread with 3(magic num) second interval
    HealthCheckManager::instance().start_health_check_thread(
        ::xllm::ServiceConfig::get_instance().health_check_interval_ms());

    LOG(INFO) << "Started cluster health check thread";
  }

  for (int idx = 0; idx < dones.size(); ++idx) {
    while (!dones[idx].load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
}
}  // namespace xllm

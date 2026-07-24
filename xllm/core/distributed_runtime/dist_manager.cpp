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
#include "runtime/worker.h"
#include "runtime/worker_client.h"
#include "server/xllm_server_registry.h"
#include "shm_channel.h"
#include "util/net.h"
namespace xllm {

DistManager::DistManager(const runtime::Options& options)
    : server_name_("CollectiveServer") {
  auto master_node_addr = options.master_node_addr().value_or("");
  if (options.enable_single_process()) {
    // Single-process mode: one OS process drives all local devices (one worker
    // thread per device, sharing one HcclCommInitAll world). Independent of
    // master_node_addr — single-node single-process needs none. Multi-node
    // single-process (nnodes > 1) is not yet supported and is rejected inside
    // setup_single_node_workers.
    setup_single_node_workers(options);
  } else if (!master_node_addr.empty()) {
    server_name_.append(std::to_string(options.server_idx()));
    setup_multi_node_workers(options, master_node_addr);
  } else {
    LOG(FATAL)
        << "master_node_addr is empty and single-process mode is not "
           "enabled. Set XLLM_ENABLE_SINGLE_PROCESS=1 for single-process "
           "serving, or --master_node_addr for multi-process/multi-node "
           "serving.";
  }
}

DistManager::~DistManager() {
  // Stop health check
  HealthCheckManager::instance().stop_health_check_thread();

  XllmServer* collective_server =
      ServerRegistry::get_instance().try_get_server(server_name_);
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

void DistManager::setup_single_node_workers(const runtime::Options& options) {
  const auto& devices = options.devices();
  CHECK_GT(devices.size(), 0)
      << "At least one device is required in single-node mode.";
  CHECK_EQ((devices.size() % options.dp_size()), 0)
      << "Device size must be divisible by dp size in single-node serving "
         "mode.";
  CHECK_EQ(options.nnodes(), 1)
      << "Multi-node single-process serving (nnodes = " << options.nnodes()
      << ") is not yet supported. Run single-node single-process "
         "(XLLM_ENABLE_SINGLE_PROCESS=1 with nnodes=1), or use the "
         "multi-process launcher (--master_node_addr, one process per device) "
         "for multi-node.";
  CHECK_EQ(options.node_rank(), 0)
      << "Single-process serving requires node_rank=0 (got "
      << options.node_rank() << ").";

  const std::string& backend = options.backend();
  CHECK(backend.empty() || backend == "llm")
      << "Single-node serving currently supports only the llm backend, got: "
      << backend;

  const int32_t world_size = static_cast<int32_t>(devices.size());
  const int32_t dp_size = options.dp_size();
  const int32_t cp_size = options.cp_size();
  const int32_t ep_size = options.ep_size();
  const int32_t dp_local_tp_size = world_size / dp_size;

  LOG(INFO) << "Single-node serving world_size = " << world_size
            << ", dp_size = " << dp_size << ", cp_size = " << cp_size
            << ", ep_size = " << ep_size << ", tp_size = " << dp_local_tp_size;

  // initialize process groups if there are multiple devices
  if (devices.size() > 1) {
#if defined(USE_NPU)
    // Single-process multi-device shares one HCCL world (HcclCommInitAll) and
    // routes tensor-parallel collectives through the C++ ProcessGroup. Only the
    // TORCH kernel backend uses that path; the ATB backend drives its own
    // process-global comm manager, which cannot rendezvous two ranks inside one
    // process. Fail fast with actionable guidance instead of hanging.
    CHECK_EQ(options.npu_kernel_backend(), "TORCH")
        << "Single-node single-process mode with multiple NPU devices requires "
           "--npu_kernel_backend=TORCH (got '"
        << options.npu_kernel_backend()
        << "'). The ATB backend does not support single-process multi-device "
           "yet; use the multi-process launcher (set --master_node_addr) for "
           "ATB.";
#endif
    process_groups_ = parallel_state::create_local_process_groups(
        devices, /*options=*/options);
  }

  // For data parallelism with dp_size > 1 and dp_size < world_size, build
  // dp-local process groups so each dp group has its own communication
  // channel. Mirrors the old setup_single_node_workers behaviour.
  if (dp_size > 1 && dp_size < world_size) {
    dp_local_process_groups_.reserve(dp_size);
    for (int32_t dp_rank = 0; dp_rank < dp_size; ++dp_rank) {
      auto begin_it = devices.begin();
      std::advance(begin_it, dp_rank * dp_local_tp_size);
      auto end_it = devices.begin();
      std::advance(end_it, (dp_rank + 1) * dp_local_tp_size);
      std::vector<torch::Device> dp_local_devices(begin_it, end_it);
      dp_local_process_groups_.emplace_back(
          parallel_state::create_local_process_groups(dp_local_devices,
                                                      options));
    }
  }

  WorkerType worker_type =
      (options.task_type() == "generate") ? WorkerType::LLM : WorkerType::ELM;

  for (size_t i = 0; i < devices.size(); ++i) {
    const int32_t rank = static_cast<int32_t>(i);
    ProcessGroup* pg = (world_size > 1) ? process_groups_[i].get() : nullptr;
    ProcessGroup* dp_local_pg =
        (dp_size > 1 && dp_size < world_size)
            ? dp_local_process_groups_[i / dp_local_tp_size]
                                      [i % dp_local_tp_size]
                                          .get()
            : nullptr;

    ParallelArgs parallel_args(rank,
                               world_size,
                               dp_size,
                               cp_size,
                               /*process_group=*/pg,
                               ep_size);
    parallel_args.dp_local_process_group_ = dp_local_pg;
    // For single-node single-process mode the language-model layers expect a
    // tensor-parallel group. With dp=ep=cp=1 the global process group also
    // serves as the tp/cp/single-rank group.
    parallel_args.tp_group_ = pg;
    parallel_args.cp_group_ = pg;
    parallel_args.single_rank_group_ = pg;
    parallel_args.moe_tp_group_ = pg;

    workers_.emplace_back(std::make_unique<Worker>(
        parallel_args, devices[i], options, worker_type));
    worker_clients_.emplace_back(
        std::make_shared<WorkerClient>(workers_.back().get(), options));
  }
}

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

  std::vector<std::atomic<bool>> dones(devices.size());
  for (size_t i = 0; i < devices.size(); ++i) {
    dones[i].store(false, std::memory_order_relaxed);
  }

  CHECK_GE(options.nnodes(), 1) << "At least one node is required";
  CHECK_GE(options.node_rank(), 0) << "Node rank must >= 0.";
  const int32_t each_node_ranks = static_cast<int32_t>(devices.size());
  const int32_t world_size = each_node_ranks * options.nnodes();
  const int32_t base_rank = options.node_rank() * each_node_ranks;
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
  // create local workers
  for (int32_t i = 0; i < each_node_ranks; ++i) {
    // worldsize = 8
    // Node1: 0, 1, 2, 3
    // Node2: 0+4, 1+4, 2+4, 3+4
    const int32_t rank = i + base_rank;
    worker_server_options.server_idx(rank);

    // we use spawn process worker to launch a xllm instance
    // when start a offline inference task with multi-gpu/npu/mpu/...
#if defined(USE_CUDA) || defined(USE_MLU) || defined(USE_DCU)
    bool use_spawn_worker = (options.enable_offline_inference() && i > 0) ||
                            force_spawn_for_numa_isolation[i];
    if (force_spawn_for_numa_isolation[i]) {
      LOG(INFO) << "Force spawn worker for local rank " << i << " (device "
                << devices[i].index() << ", NUMA " << device_numa_nodes[i]
                << ") to keep each process within a single NUMA region";
    }
#else
    bool use_spawn_worker = options.enable_offline_inference() && i > 0;
#endif
    ParallelArgs parallel_args(
        rank, world_size, dp_size, cp_size, nullptr, ep_size);

    servers_.emplace_back(std::make_unique<WorkerServer>(i,
                                                         master_node_addr,
                                                         // done,
                                                         dones[i],
                                                         parallel_args,
                                                         devices[i],
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
    for (size_t r = 0; r < world_size; ++r) {
      if (worker_addrs_map.find(r) == worker_addrs_map.end()) {
        LOG(FATAL) << "Not all worker connect to engine server. Miss rank is "
                   << r;
        return;
      }
      /* TODO(CP): support smem  + CP */
      auto channel =
          create_channel(worker_addrs_map[r], r, dp_local_tp_size, options);
      worker_clients_.emplace_back(
          std::make_unique<RemoteWorker>(r,
                                         worker_addrs_map[r],
                                         devices[r % each_node_ranks],
                                         std::move(channel)));
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

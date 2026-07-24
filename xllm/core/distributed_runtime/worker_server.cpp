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

#include "worker_server.h"

#include <brpc/channel.h>
#include <brpc/controller.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <torch/torch.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <utility>

#include "common/metrics.h"
#include "core/distributed_runtime/spawn_worker_server/spawn_worker_protocol.h"
#include "core/framework/config/eplb_config.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/service_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/platform/platform.h"
#if defined(USE_CUDA) || defined(USE_MLU) || defined(USE_DCU)
#include "core/platform/numa_utils.h"
#endif
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state/collective_communicator.h"
#include "framework/parallel_state/dit_collective_communicator.h"
#include "framework/parallel_state/mapping_npu.h"
#include "framework/state_dict/state_dict.h"
#include "runtime/forward_params.h"
#include "runtime/worker.h"
#include "server/xllm_server_registry.h"
#include "util/net.h"
#include "util/threadpool.h"
#include "util/timer.h"
#if defined(USE_NPU)
#include "hccl/hccl.h"
#include "xllm_atb_layers/core/include/atb_speed/base/external_comm_manager.h"
#include "xllm_atb_layers/core/include/atb_speed/utils/singleton.h"
#include "xllm_atb_layers/models/base/param/mapping.h"
#endif

extern char** environ;

namespace xllm {
namespace {
void handle_signal(int signum) { _exit(0); }
}  // namespace

void WorkerServer::create_server(const runtime::Options& options,
                                 std::atomic<bool>& done,
                                 const std::string& master_node_addr,
                                 const torch::Device& d,
                                 ParallelArgs startup_parallel_args,
                                 int32_t world_size,
                                 int32_t global_rank,
                                 int32_t dp_size,
                                 int32_t local_rank,
                                 int32_t ep_size,
                                 int32_t cp_size,
                                 WorkerType worker_type) {
  if (options.enable_offline_inference()) {
    ExecutionConfig::get_instance()
        .enable_graph(options.enable_graph())
        .enable_graph_mode_decode_no_padding(
            options.enable_graph_mode_decode_no_padding())
        .enable_prefill_piecewise_graph(
            options.enable_prefill_piecewise_graph())
        .max_tokens_for_graph_mode(options.max_tokens_for_graph_mode());
  }
#if defined(USE_NPU)
  KernelConfig::get_instance().npu_kernel_backend(options.npu_kernel_backend());
#endif
  Device device(d);
  device.set_device();
  LOG(INFO) << "Create worker server with device: " << device.index();

  std::unique_ptr<ForwardSharedMemoryManager> input_shm_manager = nullptr;
  std::unique_ptr<ForwardSharedMemoryManager> output_shm_manager = nullptr;
  prepare_shm(
      startup_parallel_args, options, input_shm_manager, output_shm_manager);

#if defined(USE_CUDA) || defined(USE_MLU) || defined(USE_DCU)
  // Bind worker thread to the same NUMA node as the device
  // This prevents the thread from spanning across NUMA nodes, which would
  // significantly degrade memory access and other performance aspects
  int32_t numa_node = numa::get_device_numa_node(device.index());
  if (numa_node >= 0) {
    LOG(INFO) << "Worker thread (device " << device.index()
              << ") binding to NUMA node " << numa_node;
    int32_t ret = numa::bind_thread_to_numa_node(numa_node);
    if (ret != 0) {
      LOG(WARNING) << "Failed to bind worker thread to NUMA node " << numa_node
                   << ", continuing without NUMA binding";
    }
  } else {
    LOG(INFO) << "NUMA node detection not available or not needed for device "
              << device.index();
  }
#endif

  int32_t worker_global_rank = global_rank;
  // TODO: FIXME Later
  // std::unique_ptr<WorkerImpl> worker_impl =
  // std::make_unique<LLMWorkerImpl>(...);
  // std::unique_ptr<WorkerServiceImpl> worker_service_impl =
  //    std::make_unique<WorkerServiceImpl>(worker_impl);
  auto worker_service = std::make_shared<WorkerService>(options, device);

  auto addr = net::get_local_ip_addr();
  auto worker_server =
      ServerRegistry::get_instance().register_server(server_name_);
  if (!worker_server->start(worker_service, addr + ":0")) {
    LOG(ERROR) << "failed to start distribute worker server on address: "
               << addr;
    return;
  }

  auto worker_server_addr =
      addr + ":" + std::to_string(worker_server->listen_port());
  LOG(INFO) << "Worker " << worker_global_rank
            << ": server address: " << worker_server_addr;

  // Sync with master node
  proto::AddressInfo addr_info;
  addr_info.set_address(worker_server_addr);
  addr_info.set_global_rank(worker_global_rank);
  proto::CommUniqueIdList uids;
  if (!sync_master_node(master_node_addr, addr_info, uids)) {
    LOG(WARNING) << "Worker#" << worker_global_rank
                 << " failed to sync with master node, stop worker startup.";
    return;
  }

  const ParallelArgs* parallel_args = nullptr;
  std::unique_ptr<CollectiveCommunicatorBase> comm;
  if (worker_type == WorkerType::DIT) {
    auto dit_comm =
        std::make_unique<DiTCollectiveCommunicator>(worker_global_rank,
                                                    world_size,
                                                    options.dp_size(),
                                                    options.tp_size(),
                                                    options.sp_size(),
                                                    options.cfg_size(),
                                                    options.vae_size());
    comm = std::move(dit_comm);
  } else {
    auto common_comm = std::make_unique<CollectiveCommunicator>(
        worker_global_rank, world_size, dp_size, ep_size, cp_size);
    comm = std::move(common_comm);
  }

  // Forward master-allocated TCPStore ports (if any) so the communicator
  // binds on OS-chosen free ports instead of fixed master_port + offset
  // values that can collide with random outgoing TCP source ports inside
  // the local ephemeral range.
  std::vector<int32_t> group_ports(uids.group_ports().begin(),
                                   uids.group_ports().end());
  if (!group_ports.empty()) {
    comm->set_group_ports(std::move(group_ports));
  }

  comm->create_process_groups(master_node_addr, device);
  parallel_args = comm->parallel_args();

  ParallelArgs worker_parallel_args = *parallel_args;
  const int32_t cp_group_size =
      worker_parallel_args.cp_group_ == nullptr
          ? 1
          : worker_parallel_args.cp_group_->world_size();
  const char* cp_partition_stage =
      options.cp_size() <= 1
          ? "disabled"
          : (Platform::uses_model_cp_partition() ? "model" : "worker");
  LOG(INFO) << "Worker CP config: cp_size=" << options.cp_size()
            << ", cp_group_size=" << cp_group_size
            << ", cp_partition_stage=" << cp_partition_stage
            << ", instance_role=" << options.instance_role().to_string();

  std::unique_ptr<Worker> worker = std::make_unique<Worker>(
      worker_parallel_args, device, options, worker_type);
  worker_service->set_worker(std::move(worker));
  bool create_shm =
      options.enable_shm() && input_shm_manager && output_shm_manager;
  if (create_shm) {
    worker_service->create_polling_shm_thread(std::move(input_shm_manager),
                                              std::move(output_shm_manager));
  }

  done.store(true);

  // Wait until Ctrl-C is pressed, then Stop() and Join() the server.
  worker_server->run();
}

void WorkerServer::stop() {
  bool expected = false;
  if (!stopped_.compare_exchange_strong(expected, true)) {
    return;
  }

  auto& registry = ServerRegistry::get_instance();
  auto worker_server = registry.try_get_server(server_name_);
  if (worker_server != nullptr) {
    worker_server->stop();
    registry.unregister_server(server_name_);
  }

  if (worker_thread_ && worker_thread_->joinable()) {
    worker_thread_->join();
  }

  stop_spawn_worker();
  use_spwan_worker_ = false;
}

void WorkerServer::create_spawn_server(int32_t local_rank,
                                       const std::string& master_node_addr,
                                       std::atomic<bool>& done,
                                       const ParallelArgs& parallel_args,
                                       const torch::Device& d,
                                       const runtime::Options& options,
                                       WorkerType& worker_type) {
  std::string local_rank_str = std::to_string(local_rank);
  const char* local_rank_ptr = local_rank_str.c_str();
  std::string global_rank_str = std::to_string(parallel_args.rank());
  const char* global_rank_ptr = global_rank_str.c_str();
  std::string world_size_str = std::to_string(parallel_args.world_size());
  const char* world_size_ptr = world_size_str.c_str();
  std::string device_idx_str = std::to_string(d.index());
  const char* device_idx_ptr = device_idx_str.c_str();
  std::string num_decoding_tokens_str =
      std::to_string(options.num_decoding_tokens());
  const char* num_decoding_tokens_ptr = num_decoding_tokens_str.c_str();
  std::string block_size_str = std::to_string(options.block_size());
  const char* block_size_ptr = block_size_str.c_str();
  std::string max_tokens_per_batch_str =
      std::to_string(options.max_tokens_per_batch());
  const char* max_tokens_per_batch_ptr = max_tokens_per_batch_str.c_str();
  std::string max_seqs_per_batch_str =
      std::to_string(options.max_seqs_per_batch());
  const char* max_seqs_per_batch_ptr = max_seqs_per_batch_str.c_str();
  std::string enable_shm_str = std::to_string(options.enable_shm());
  const char* enable_shm_ptr = enable_shm_str.c_str();
  std::string input_shm_size_str = std::to_string(options.input_shm_size());
  const char* input_shm_size_ptr = input_shm_size_str.c_str();
  std::string output_shm_size_str = std::to_string(options.output_shm_size());
  const char* output_shm_size_ptr = output_shm_size_str.c_str();
  std::string is_local_str = std::to_string(options.is_local());
  const char* is_local_ptr = is_local_str.c_str();
  std::string cp_size_str = std::to_string(options.cp_size());
  const char* cp_size_ptr = cp_size_str.c_str();
  std::string ep_size_str = std::to_string(parallel_args.ep_size());
  const char* ep_size_ptr = ep_size_str.c_str();
  std::string instance_role_str = options.instance_role().to_string();
  const char* instance_role_ptr = instance_role_str.c_str();
  std::string enable_speculative_decode_str =
      std::to_string(options.enable_speculative_decode());
  const char* enable_speculative_decode_ptr =
      enable_speculative_decode_str.c_str();
  std::string num_speculative_tokens_str =
      std::to_string(options.num_speculative_tokens());
  const char* num_speculative_tokens_ptr = num_speculative_tokens_str.c_str();
  std::string speculative_algorithm_str = options.speculative_algorithm();
  const char* speculative_algorithm_ptr = speculative_algorithm_str.c_str();
  std::string enable_graph_str = std::to_string(options.enable_graph());
  const char* enable_graph_ptr = enable_graph_str.c_str();
  std::string enable_graph_mode_decode_no_padding_str =
      std::to_string(options.enable_graph_mode_decode_no_padding());
  const char* enable_graph_mode_decode_no_padding_ptr =
      enable_graph_mode_decode_no_padding_str.c_str();
  std::string enable_prefill_piecewise_graph_str =
      std::to_string(options.enable_prefill_piecewise_graph());
  const char* enable_prefill_piecewise_graph_ptr =
      enable_prefill_piecewise_graph_str.c_str();
  std::string max_tokens_for_graph_mode_str =
      std::to_string(options.max_tokens_for_graph_mode());
  const char* max_tokens_for_graph_mode_ptr =
      max_tokens_for_graph_mode_str.c_str();
  const char* communication_backend_ptr =
      ::xllm::ParallelConfig::get_instance().communication_backend().c_str();
  std::string npu_kernel_backend_str = options.npu_kernel_backend();
  const char* npu_kernel_backend_ptr = npu_kernel_backend_str.c_str();
  const char* rank_tablefile_ptr =
      ::xllm::EPLBConfig::get_instance().rank_tablefile().c_str();
  std::string max_encoder_cache_size_str =
      std::to_string(options.max_encoder_cache_size());
  const char* max_encoder_cache_size_ptr = max_encoder_cache_size_str.c_str();
  std::string dp_size_str = std::to_string(options.dp_size());
  const char* dp_size_ptr = dp_size_str.c_str();
  std::string tp_size_str = std::to_string(options.tp_size());
  const char* tp_size_ptr = tp_size_str.c_str();
  std::string sp_size_str = std::to_string(options.sp_size());
  const char* sp_size_ptr = sp_size_str.c_str();
  std::string cfg_size_str = std::to_string(options.cfg_size());
  const char* cfg_size_ptr = cfg_size_str.c_str();
  const std::string& indexer_cache_dtype =
      KVCacheConfig::get_instance().indexer_cache_dtype();
  const char* indexer_cache_dtype_ptr = indexer_cache_dtype.c_str();
  std::string enable_mtp_draft_body_tp1_str =
      std::to_string(options.enable_mtp_draft_body_tp1());
  const char* enable_mtp_draft_body_tp1_ptr =
      enable_mtp_draft_body_tp1_str.c_str();
  const char* worker_type_ptr = worker_type.to_string();
  std::string spawn_worker_bin_path =
      options.spawn_worker_path() + "/spawn_worker";
  LOG(INFO) << "Spawn worker path: " << spawn_worker_bin_path;
  const char* cp_partition_stage =
      options.cp_size() <= 1
          ? "disabled"
          : (Platform::uses_model_cp_partition() ? "model" : "worker");
  LOG(INFO) << "Spawn worker CP config: cp_size=" << options.cp_size()
            << ", global_world_size=" << parallel_args.world_size()
            << ", global_rank=" << parallel_args.rank()
            << ", ep_size=" << parallel_args.ep_size()
            << ", cp_partition_stage=" << cp_partition_stage
            << ", instance_role=" << options.instance_role().to_string();
  const char* argv[] = {spawn_worker_bin_path.c_str(),
                        master_node_addr.c_str(),
                        local_rank_ptr,
                        global_rank_ptr,
                        world_size_ptr,
                        device_idx_ptr,
                        num_decoding_tokens_ptr,
                        block_size_ptr,
                        max_tokens_per_batch_ptr,
                        max_seqs_per_batch_ptr,
                        enable_shm_ptr,
                        is_local_ptr,
                        options.task_type().c_str(),
                        worker_type_ptr,
                        enable_speculative_decode_ptr,
                        num_speculative_tokens_ptr,
                        speculative_algorithm_ptr,
                        input_shm_size_ptr,
                        output_shm_size_ptr,
                        communication_backend_ptr,
                        npu_kernel_backend_ptr,
                        rank_tablefile_ptr,
                        enable_graph_ptr,
                        enable_graph_mode_decode_no_padding_ptr,
                        enable_prefill_piecewise_graph_ptr,
                        max_tokens_for_graph_mode_ptr,
                        max_encoder_cache_size_ptr,
                        dp_size_ptr,
                        tp_size_ptr,
                        sp_size_ptr,
                        cfg_size_ptr,
                        cp_size_ptr,
                        ep_size_ptr,
                        instance_role_ptr,
                        indexer_cache_dtype_ptr,
                        enable_mtp_draft_body_tp1_ptr,
                        nullptr};
  static_assert(std::size(argv) == spawn_worker_protocol::kArgumentCount + 1);
  pid_t pid;
  int status = posix_spawnp(
      &pid, argv[0], nullptr, nullptr, const_cast<char**>(argv), environ);
  if (status != 0) {
    LOG(ERROR) << "posix_spawnp failed: " << strerror(status);
    return;
  }
  use_spwan_worker_ = true;
  spawned_worker_pid_ = pid;
  LOG(INFO) << "Spawn worker success, pid: " << spawned_worker_pid_;
  done.store(true);
}

void WorkerServer::prepare_shm(
    const ParallelArgs& parallel_args,
    const runtime::Options& options,
    std::unique_ptr<ForwardSharedMemoryManager>& input_shm_manager,
    std::unique_ptr<ForwardSharedMemoryManager>& output_shm_manager) {
  if (options.is_local() && options.enable_shm()) {
    bool is_creator;
    int32_t dp_local_tp_size =
        parallel_args.world_size() / parallel_args.dp_size();
    int32_t dp_group = parallel_args.rank() / dp_local_tp_size;

    std::string name_prefix =
        "xllm_" + net::extract_port(options.master_node_addr().value());
    string name = ForwardSharedMemoryManager::create_unique_name(
        name_prefix, dp_group, ForwardType::RAW_INPUT, parallel_args.rank());
    input_shm_manager = std::make_unique<ForwardSharedMemoryManager>(
        name, options.input_shm_size(), is_creator, ForwardType::RAW_INPUT);
    LOG(INFO) << "Create input shared memory manager with name: " << name;

    name = ForwardSharedMemoryManager::create_unique_name(
        name_prefix, dp_group, ForwardType::RAW_OUTPUT, parallel_args.rank());
    output_shm_manager = std::make_unique<ForwardSharedMemoryManager>(
        name, options.output_shm_size(), is_creator, ForwardType::RAW_OUTPUT);
    LOG(INFO) << "Create output shared memory manager with name: " << name;
  }
}

WorkerServer::WorkerServer(int32_t local_worker_idx,
                           const std::string& master_node_addr,
                           std::atomic<bool>& done,
                           const ParallelArgs& parallel_args,
                           const torch::Device& d,
                           const runtime::Options& options,
                           WorkerType worker_type,
                           bool use_spawn_worker)
    : server_name_("DistributeWorkerServer") {
  server_name_.append(std::to_string(options.server_idx()));
  // Eagle3/DFlash targets capture aux hidden states and don't support spawned
  // workers yet.
  const std::string& speculative_algorithm = options.speculative_algorithm();
  if (use_spawn_worker && options.enable_speculative_decode() &&
      SpeculativeConfig::requires_aux_hidden_capture(speculative_algorithm)) {
    LOG(FATAL) << speculative_algorithm
               << " does not support spawned workers yet. Disable offline "
                  "spawn workers or use --speculative_algorithm=MTP.";
  }

  if (worker_type == WorkerType::LLM || worker_type == WorkerType::ELM ||
      worker_type == WorkerType::VLM || worker_type == WorkerType::EVLM ||
      worker_type == WorkerType::REC || worker_type == WorkerType::MMEVLM ||
      worker_type == WorkerType::DIT) {
    if (use_spawn_worker) {
      // start worker in a spawn process(for offline inference worker.)
      create_spawn_server(local_worker_idx,
                          master_node_addr,
                          done,
                          parallel_args,
                          d,
                          options,
                          worker_type);
      return;
    } else {
      // worker process should handle SIGTREM and SIGINT signals.
      signal(SIGINT, handle_signal);
      signal(SIGTERM, handle_signal);

      // start worker in a thread.
      worker_thread_ =
          std::make_unique<std::thread>(&WorkerServer::create_server,
                                        this,
                                        options,
                                        std::ref(done),
                                        master_node_addr,
                                        d,
                                        parallel_args,
                                        parallel_args.world_size(),
                                        parallel_args.rank(),
                                        parallel_args.dp_size(),
                                        local_worker_idx,
                                        parallel_args.ep_size(),
                                        parallel_args.cp_size(),
                                        worker_type);
    }
  } else {
    // TODO: support other model type later.
    LOG(ERROR) << "Unsupported model type: " << worker_type;
  }
}

bool WorkerServer::sync_master_node(const std::string& master_node_addr,
                                    proto::AddressInfo& addr_info,
                                    proto::CommUniqueIdList& uids) {
  // Brpc connection resources
  brpc::Channel channel;
  brpc::ChannelOptions options;
  options.connection_type = "single";
  options.timeout_ms = 10000;
  options.max_retry = 3;
  if (channel.Init(master_node_addr.c_str(), "", &options) != 0) {
    LOG(ERROR) << "Failed to initialize BRPC channel to " << master_node_addr;
    return false;
  }
  proto::Collective_Stub stub(&channel);

  // Retry until master node ready
  int32_t try_count = 0;
  brpc::Controller cntl;
  constexpr int32_t kSleepTimeSecond = 3;
  while (try_count <
         ::xllm::ServiceConfig::get_instance().max_reconnect_count()) {
    cntl.Reset();
    stub.Sync(&cntl, &addr_info, &uids, nullptr);
    if (cntl.Failed()) {
      LOG(WARNING) << "Worker#" << addr_info.global_rank()
                   << " try connect to engine server error, try again."
                   << " Error message: " << cntl.ErrorText();
      std::this_thread::sleep_for(std::chrono::seconds(kSleepTimeSecond));
    } else {
      LOG(INFO) << "Worker#" << addr_info.global_rank() << " connect to "
                << master_node_addr << " success.";
      break;
    }
    try_count++;
  }

  if (try_count >=
      ::xllm::ServiceConfig::get_instance().max_reconnect_count()) {
    LOG(ERROR) << "Worker#" << addr_info.global_rank() << " connect to "
               << master_node_addr << " failed."
               << " Error message: " << cntl.ErrorText();
    return false;
  }

  return true;
}

WorkerServer::~WorkerServer() { stop(); }

void WorkerServer::stop_spawn_worker() {
  if (!use_spwan_worker_ || spawned_worker_pid_ <= 0) {
    return;
  }

  if (kill(spawned_worker_pid_, SIGTERM) != 0 && errno != ESRCH) {
    LOG(WARNING) << "Failed to send SIGTERM to spawn worker pid "
                 << spawned_worker_pid_ << ", errno: " << errno;
  }

  int status = 0;
  pid_t ret = waitpid(spawned_worker_pid_, &status, 0);
  if (ret == spawned_worker_pid_) {
    LOG(INFO) << "Spawn worker(pid=" << spawned_worker_pid_
              << ") exited with status: " << status;
  } else if (ret < 0 && errno != ECHILD) {
    LOG(WARNING) << "waitpid failed for spawn worker pid "
                 << spawned_worker_pid_ << ", errno: " << errno;
  }

  spawned_worker_pid_ = -1;
}

}  // namespace xllm

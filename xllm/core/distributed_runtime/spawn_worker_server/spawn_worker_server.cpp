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

#include "spawn_worker_server.h"

#if defined(USE_NPU)
#include <acl/acl.h>
#endif
#include <signal.h>
#include <unistd.h>

#include "core/common/global_flags.h"
#include "core/distributed_runtime/worker_server.h"
#include "core/framework/config/distributed_config.h"
#include "core/framework/config/eplb_config.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/platform/device.h"
#include "core/platform/platform.h"
#if defined(USE_CUDA) || defined(USE_MLU) || defined(USE_DCU)
#include "core/platform/numa_utils.h"
#endif
#include "core/runtime/options.h"

namespace xllm {

namespace {
std::string get_backend_from_worker_type(const std::string& worker_type) {
  if (worker_type == "LLM" || worker_type == "ELM") {
    return "llm";
  }
  if (worker_type == "VLM" || worker_type == "EVLM" ||
      worker_type == "MMEVLM") {
    return "vlm";
  }
  if (worker_type == "REC") {
    return "rec";
  }
  if (worker_type == "DIT") {
    return "dit";
  }
  return "";
}
}  // namespace

SpawnWorkerServer::SpawnWorkerServer(const std::string& master_node_addr,
                                     int32_t local_rank,
                                     int32_t global_rank,
                                     int32_t world_size,
                                     int32_t device_idx,
                                     int32_t num_decoding_tokens,
                                     int32_t block_size,
                                     int32_t max_tokens_per_batch,
                                     int32_t max_seqs_per_batch,
                                     bool enable_shm,
                                     uint64_t input_shm_size,
                                     uint64_t output_shm_size,
                                     bool is_local,
                                     bool enable_prefill_sp,
                                     const std::string& task_type,
                                     const std::string& worker_type,
                                     bool enable_speculative_decode,
                                     int32_t num_speculative_tokens,
                                     const std::string& speculative_algorithm,
                                     const std::string& communication_backend,
                                     const std::string& npu_kernel_backend,
                                     const std::string& rank_tablefile,
                                     bool enable_graph,
                                     bool enable_graph_mode_decode_no_padding,
                                     bool enable_prefill_piecewise_graph,
                                     int32_t max_tokens_for_graph_mode,
                                     int64_t max_encoder_cache_size,
                                     int32_t dp_size,
                                     int32_t tp_size,
                                     int32_t sp_size,
                                     int32_t cfg_size) {
  // TODO: pass whole xllm::runtime::Options here from main process.
  xllm::runtime::Options runner_options;
  const std::string backend = get_backend_from_worker_type(worker_type);
  CHECK(!backend.empty()) << "Unsupported worker_type for backend mapping: "
                          << worker_type;
  const bool is_dit_backend = backend == "dit";
  const int32_t effective_sp_size = is_dit_backend ? sp_size : 1;
  const int32_t effective_cfg_size = is_dit_backend ? cfg_size : 1;
  runner_options.block_size(block_size)
      .backend(backend)
      .world_size(world_size)
      .max_tokens_per_batch(max_tokens_per_batch)
      .max_seqs_per_batch(max_seqs_per_batch)
      .num_decoding_tokens(num_decoding_tokens)
      .enable_prefill_sp(enable_prefill_sp)
      .enable_speculative_decode(enable_speculative_decode)
      .num_speculative_tokens(num_speculative_tokens)
      .speculative_algorithm(speculative_algorithm)
      .enable_schedule_overlap(/*enable_schedule_overlap=*/false)
      .enable_offline_inference(/*enable_offline_inference=*/true)
      .master_node_addr(master_node_addr)
      .dp_size(dp_size)
      .tp_size(tp_size)
      .sp_size(effective_sp_size)
      .cfg_size(effective_cfg_size)
      .enable_shm(enable_shm)
      .input_shm_size(input_shm_size)
      .output_shm_size(output_shm_size)
      .is_local(is_local)
      .npu_kernel_backend(npu_kernel_backend)
      .enable_graph(enable_graph)
      .enable_graph_mode_decode_no_padding(enable_graph_mode_decode_no_padding)
      .enable_prefill_piecewise_graph(enable_prefill_piecewise_graph)
      .max_tokens_for_graph_mode(max_tokens_for_graph_mode)
      .task_type(task_type)
      .max_encoder_cache_size(max_encoder_cache_size);
  SchedulerConfig::get_instance()
      .max_tokens_per_batch(max_tokens_per_batch)
      .max_seqs_per_batch(max_seqs_per_batch)
      .enable_schedule_overlap(false);
  ParallelConfig::get_instance()
      .enable_prefill_sp(enable_prefill_sp)
      .dp_size(dp_size)
      .tp_size(tp_size)
      .sp_size(effective_sp_size)
      .cfg_size(effective_cfg_size)
      .communication_backend(communication_backend);
  DistributedConfig::get_instance().master_node_addr(master_node_addr);
  KVCacheConfig::get_instance().block_size(block_size);
  EPLBConfig::get_instance().rank_tablefile(rank_tablefile);

  xllm::Device device{device_idx};
  device.set_device();

#if defined(USE_NPU)
  device.init_device_context();
  KernelConfig::get_instance().npu_kernel_backend(npu_kernel_backend);
  FLAGS_enable_atb_comm_multiprocess = true;
#endif
#if defined(USE_CUDA) || defined(USE_MLU) || defined(USE_DCU)
  // Bind worker process to the same NUMA node as the device
  // This prevents the process from spanning across NUMA nodes, which would
  // significantly degrade memory access and other performance aspects
  int32_t numa_node = numa::get_device_numa_node(device_idx);
  if (numa_node >= 0) {
    LOG(INFO) << "Worker process (device " << device_idx
              << ") binding to NUMA node " << numa_node;
    int32_t ret = numa::bind_process_to_numa_node(numa_node);
    if (ret != 0) {
      LOG(WARNING) << "Failed to bind worker process to NUMA node " << numa_node
                   << ", continuing without NUMA binding";
    }
  } else {
    LOG(INFO) << "NUMA node detection not available or not needed for device "
              << device_idx;
  }
#endif

  ParallelArgs parallel_args(global_rank,
                             world_size,
                             dp_size,
                             /* cp_size = */ 1,
                             /* process_group = */ nullptr,
                             /* ep_size = */ 1);
  worker_server_ = std::make_unique<WorkerServer>(local_rank,
                                                  master_node_addr,
                                                  done_,
                                                  parallel_args,
                                                  device,
                                                  runner_options,
                                                  worker_type,
                                                  false);
}

SpawnWorkerServer::~SpawnWorkerServer() = default;

void SpawnWorkerServer::handle_signal(int signum) {
  (void)signum;
  _exit(0);
}

void SpawnWorkerServer::run() {
  signal(SIGINT, SpawnWorkerServer::handle_signal);
  signal(SIGTERM, SpawnWorkerServer::handle_signal);
  signal(SIGHUP, SpawnWorkerServer::handle_signal);

  // Keep process alive until SIGTERM/SIGINT arrives from parent teardown.
  while (true) {
    pause();
  }
}

}  // namespace xllm

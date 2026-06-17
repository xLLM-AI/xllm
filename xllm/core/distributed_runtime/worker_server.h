/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#pragma once

#include <brpc/server.h>
#include <folly/futures/Future.h>
#include <sys/types.h>
#include <torch/torch.h>

#include <atomic>
#include <cstdint>
#include <thread>

#include "common/macros.h"
#include "distributed_runtime/worker_service.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "runtime/executor.h"
#include "runtime/forward_params.h"
#include "runtime/forward_shared_memory_manager.h"
#include "runtime/options.h"
#include "runtime/worker_impl.h"
#include "worker.pb.h"

namespace xllm {

class WorkerServer {
 public:
  WorkerServer(int32_t local_worker_idx,
               const std::string& master_node_addr,
               std::atomic<bool>& done,
               const ParallelArgs& parallel_args,
               const torch::Device& d,
               const runtime::Options& options,
               WorkerType worker_type,
               bool use_spawn_worker = false);

  virtual ~WorkerServer();

  void create_server(const runtime::Options& options,
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
                     WorkerType worker_type);

  void stop();

 private:
  DISALLOW_COPY_AND_ASSIGN(WorkerServer);

  void create_spawn_server(int32_t local_rank,
                           const std::string& master_node_addr,
                           std::atomic<bool>& done,
                           const ParallelArgs& parallel_args,
                           const torch::Device& d,
                           const runtime::Options& options,
                           WorkerType& worker_type);

  void create_shared_memory_polling(
      std::unique_ptr<ForwardSharedMemoryManager> input_shm_manager,
      std::unique_ptr<ForwardSharedMemoryManager> output_shm_manager);

  void stop_spawn_worker();

  bool sync_master_node(const std::string& master_node_addr,
                        proto::AddressInfo& addr_info,
                        proto::CommUniqueIdList& uids);

  void prepare_shm(
      const ParallelArgs& parallel_args,
      const runtime::Options& options,
      std::unique_ptr<ForwardSharedMemoryManager>& input_shm_manager,
      std::unique_ptr<ForwardSharedMemoryManager>& output_shm_manager);

 private:
  std::unique_ptr<std::thread> worker_thread_;

  // for offline inference spawn process worker.
  bool use_spwan_worker_ = false;
  pid_t spawned_worker_pid_ = -1;
  std::atomic<bool> stopped_{false};
  std::string server_name_;
};

}  // namespace xllm

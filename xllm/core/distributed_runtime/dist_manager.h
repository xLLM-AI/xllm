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

#pragma once

#include "common/macros.h"
#include "distributed_runtime/remote_worker.h"
#include "distributed_runtime/worker_server.h"
#include "distributed_runtime/worker_service.h"
#include "framework/parallel_state/process_group.h"
#include "runtime/forward_shared_memory_manager.h"
#include "runtime/options.h"
namespace xllm {
class DistManager {
 public:
  explicit DistManager(const runtime::Options& options);
  virtual ~DistManager();

  std::vector<std::shared_ptr<WorkerClient>> get_worker_clients() {
    return worker_clients_;
  };

 private:
  DISALLOW_COPY_AND_ASSIGN(DistManager);
  void setup_single_node_workers(const runtime::Options& options);
  void setup_multi_node_workers(const runtime::Options& options,
                                const std::string& master_node_addr);

 private:
  // a list of process groups, with each process group handling a single device
  std::vector<std::unique_ptr<ProcessGroup>> process_groups_;

  // multiple data parallel process group
  std::vector<std::vector<std::unique_ptr<ProcessGroup>>>
      dp_local_process_groups_;

  // worker client which is used for call worker
  // The reason for adding a worker client is to unify the
  // access code for both local and remote workers, thereby
  // introducing an additional worker_client abstraction.
  std::vector<std::shared_ptr<WorkerClient>> worker_clients_;
  // a list of workers, with each worker handling a partial of model.
  // And in distributed serving, worker will be setup in a new process.
  std::vector<std::unique_ptr<Worker>> workers_;
  // For distributed serving
  std::vector<std::unique_ptr<WorkerServer>> servers_;

  std::string server_name_;
};
}  // namespace xllm

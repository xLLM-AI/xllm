/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <gflags/gflags.h>
#include <sys/sysinfo.h>
#include <unistd.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "common/macros.h"
#include "core/distributed_runtime/distributed_worker_manager.h"
#include "core/distributed_runtime/master.h"
#include "core/kv_cache/storage/kv_cache_utils.h"
#include "engine.h"
#include "framework/batch/batch_group.h"
#include "framework/batch/forward_input_factory.h"
#include "framework/block/block_manager_pool.h"
#include "framework/tokenizer/tokenizer.h"
#include "framework/tokenizer/tokenizer_args.h"
#include "runtime/worker.h"
#include "runtime/worker_client.h"
namespace xllm {

class EplbController;
class KVCacheTransferCoordinatorBase;
class ModelLoader;

class LLMEngine : public Engine {
 public:
  // Optional hook that runs synchronously before workers load weights. The
  // caller supplies any required resource preparation; the engine does not
  // retain the callback or own the resources prepared by it.
  using ModelInitCallback = std::function<
      bool(const ModelLoader&, int64_t, int32_t, int32_t, MasterStatus)>;

  // create an engine with the given devices
  LLMEngine(const runtime::Options& options,
            std::shared_ptr<DistributedWorkerManager>
                distributed_worker_manager = nullptr);

  ~LLMEngine() override;

  ForwardOutput step(BatchGroup& batch);

  const runtime::Options& options() const { return options_; }

  // The caller must prepare any required external model resources before
  // initializing without a preparation callback.
  bool init(MasterStatus master_status) override;

  bool init(MasterStatus master_status,
            const ModelInitCallback& prepare_model,
            std::shared_ptr<KVCacheTransferCoordinatorBase>
                transfer_coordinator = nullptr);

  void update_last_step_result(BatchGroup& batch);

  std::shared_ptr<DistributedWorkerManager> get_distributed_worker_manager()
      const {
    return distributed_worker_manager_;
  }

  bool start_profile() override;

  bool stop_profile() override;

 private:
  bool profile_workers(bool is_start);
  std::mutex profile_mutex_;

  template <typename TargetEngine>
  friend class SpeculativeEngineBase;
  // setup workers internal
  void setup_workers(const runtime::Options& options);
  bool init_model(MasterStatus master_status,
                  const ModelInitCallback& prepare_model);
  KVCacheCapacity estimate_kv_cache_capacity();
  bool allocate_kv_cache(
      const KVCacheCapacity& kv_cache_cap,
      std::shared_ptr<KVCacheTransferCoordinatorBase> transfer_coordinator);
  void process_group_test();

 protected:
  // options
  runtime::Options options_;

  // dtype
  torch::ScalarType dtype_;

  // worker client which is used for call worker
  // The reason for adding a worker client is to unify the
  // access code for both local and remote workers, thereby
  // introducing an additional worker_client abstraction.
  std::vector<std::shared_ptr<WorkerClient>> worker_clients_;

  // common frequently used args
  uint32_t dp_size_ = 1;
  uint32_t worker_clients_num_;
  // Effective TP width (MLU=dp_local; NPU=dp_local/cp).
  uint32_t dp_local_tp_size_;
  uint32_t dp_local_size_;
  std::unique_ptr<ForwardInputFactory> forward_input_factory_;

  // For multi-node serving
  // engine brpc server, all workers connect to engine_server_,
  // engine_server_ will send a UniqueId for workers to
  // create process group. And workers send worker brpc server
  // address to engine, engine will create WorkerClient for each worker.
  // Engine call workers to step via these WorkerClients.
  std::shared_ptr<DistributedWorkerManager> distributed_worker_manager_ =
      nullptr;

  std::unique_ptr<EplbController> eplb_controller_;
};

}  // namespace xllm

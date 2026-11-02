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

#include <memory>
#include <vector>

#include "common/macros.h"
#include "core/distributed_runtime/distributed_worker_manager.h"
#include "core/framework/batch/vlm_forward_input_factory.h"
#include "engine.h"
#include "framework/batch/batch_group.h"
#include "framework/block/block_manager_pool.h"
#include "framework/tokenizer/tokenizer.h"
#include "framework/tokenizer/tokenizer_args.h"
#include "runtime/vlm_forward_params.h"

namespace xllm {

class VmmKVCacheLayout;

class VLMEngine : public Engine {
 public:
  // create an engine with the given devices
  VLMEngine(const runtime::Options& options,
            std::shared_ptr<DistributedWorkerManager>
                distributed_worker_manager = nullptr);

  virtual ~VLMEngine() = default;

  ForwardOutput step(BatchGroup& batch);

  const runtime::Options& options() const { return options_; }

  std::shared_ptr<DistributedWorkerManager> get_distributed_worker_manager()
      const {
    return distributed_worker_manager_;
  }

  bool init(MasterStatus master_status) override;

  std::vector<RawForwardOutput> collect_step_outputs(const BatchGroup& batch);

 private:
  template <typename TargetEngine>
  friend class SpeculativeEngineBase;
  bool init_model(MasterStatus master_status);
  KVCacheCapacity estimate_kv_cache_capacity(
      bool initialize_virtual_memory = true);
  bool allocate_kv_cache(const KVCacheCapacity& kv_cache_cap,
                         const VmmKVCacheLayout* shared_vmm_layout = nullptr);
  void setup_workers(const runtime::Options& options);
  void process_group_test();

 private:
  // options
  runtime::Options options_;

  // dtype
  torch::ScalarType dtype_;

  // a list of workers, with each worker handling a partial of model
  std::vector<std::shared_ptr<WorkerClient>> worker_clients_;

  // common frequently used args
  uint32_t dp_size_;
  uint32_t worker_clients_num_;
  uint32_t dp_local_tp_size_;

  std::shared_ptr<DistributedWorkerManager> distributed_worker_manager_ =
      nullptr;

  std::unique_ptr<VlmForwardInputFactory> forward_input_factory_;
};

}  // namespace xllm

/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <gflags/gflags.h>

#include <memory>

#include "common/macros.h"
#include "core/distributed_runtime/dist_manager.h"
#include "engine.h"
#include "framework/batch/batch.h"
#include "framework/block/block_manager_pool.h"
#include "framework/quant_args.h"
#include "framework/tokenizer/tokenizer.h"
#include "framework/tokenizer/tokenizer_args.h"
#include "runtime/worker.h"
#include "runtime/worker_client.h"
#include "util/threadpool.h"

namespace xllm {

class VLMEngine : public Engine {
 public:
  // create an engine with the given devices
  VLMEngine(const runtime::Options& options,
            std::shared_ptr<DistManager> dist_manager = nullptr);

  virtual ~VLMEngine() = default;

  ForwardOutput step(std::vector<Batch>& batch) override;

  const runtime::Options& options() const { return options_; }

  bool init(MasterStatus master_status) override;

  void update_last_step_result(std::vector<Batch>& batch) override;

  // return the active activation memory
  std::vector<int64_t> get_active_activation_memory() const override;

 private:
  template <typename TargetEngine>
  friend class SpeculativeEngineBase;
  bool init_model(MasterStatus master_status);
  KVCacheCapacity estimate_kv_cache_capacity();
  bool allocate_kv_cache(const KVCacheCapacity& kv_cache_cap);
  std::vector<ForwardInput> prepare_inputs(std::vector<Batch>& batch);
  void setup_workers(const runtime::Options& options);
  void process_group_test();

 private:
  // options
  runtime::Options options_;

  // dtype
  torch::ScalarType dtype_;

  // quantization args
  QuantArgs quant_args_;

  // a list of process groups, with each process group handling a single device
  std::vector<std::unique_ptr<ProcessGroup>> process_groups_;

  // a list of workers, with each worker handling a partial of model
  std::vector<std::shared_ptr<WorkerClient>> worker_clients_;

  // common frequently used args
  uint32_t dp_size_;
  uint32_t worker_clients_num_;
  uint32_t dp_local_tp_size_;

  bool layer_forward_interrupted_ = false;

  std::shared_ptr<DistManager> dist_manager_ = nullptr;

  std::unique_ptr<ThreadPool> threadpool_ = nullptr;

  // config for kv cache
  int64_t n_local_kv_heads_ = 0;
  int64_t n_local_linear_v_heads_ = 0;
  int64_t n_local_linear_k_heads_ = 0;
  int64_t head_dim_ = 0;
};

}  // namespace xllm

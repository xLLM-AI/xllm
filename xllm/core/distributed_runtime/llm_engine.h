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

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "core/distributed_runtime/distributed_worker_manager.h"
#include "core/distributed_runtime/engine.h"
#include "core/runtime/forward_params.h"
#include "framework/batch/batch_group.h"
#include "framework/batch/forward_input_factory.h"
#include "framework/block/block_manager_pool.h"
#include "framework/model/model_args.h"
#include "framework/tokenizer/tokenizer.h"
#include "framework/tokenizer/tokenizer_args.h"
namespace xllm {

class EplbController;
class KVCacheTransferCoordinatorBase;
class DraftStepController;

class LLMEngine final : public Engine {
 public:
  // create an engine with the given devices
  explicit LLMEngine(const runtime::Options& options,
                     std::shared_ptr<DistributedWorkerManager>
                         distributed_worker_manager = nullptr);

  ~LLMEngine() override;

  ForwardOutput step(BatchGroup& batch);

  bool init(
      MasterStatus master_status,
      const ModelInitCallback& prepare_model,
      std::shared_ptr<KVCacheTransferCoordinatorBase> transfer_coordinator);

  std::vector<RawForwardOutput> collect_step_outputs(const BatchGroup& batch);

  bool start_profile();

  bool stop_profile();

  DraftStepController* draft_step_controller() {
    return draft_step_controller_.get();
  }

 private:
  bool profile_workers(bool is_start);
  void select_draft_steps(std::vector<LlmForwardInput>& inputs);
  void observe_draft_step_acceptance(const RawForwardOutput& output);
  std::mutex profile_mutex_;
  std::unique_ptr<DSparkDpCoordinator> dspark_dp_coordinator_;

  bool init_model(MasterStatus master_status,
                  const ModelInitCallback& prepare_model);
  bool allocate_kv_cache(
      const KVCacheCapacity& kv_cache_cap,
      std::shared_ptr<KVCacheTransferCoordinatorBase> transfer_coordinator);
  void process_group_test();

  uint32_t dp_local_size_;
  std::unique_ptr<ForwardInputFactory> forward_input_factory_;

  std::unique_ptr<EplbController> eplb_controller_;
  std::unique_ptr<DraftStepController> draft_step_controller_;
};

}  // namespace xllm

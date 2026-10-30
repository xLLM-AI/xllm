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

#include <memory>
#include <vector>

#include "core/distributed_runtime/distributed_worker_manager.h"
#include "core/distributed_runtime/engine.h"
#include "core/framework/batch/vlm_forward_input_factory.h"
#include "core/runtime/forward_params.h"
#include "framework/batch/batch_group.h"
#include "framework/block/block_manager_pool.h"
#include "framework/model/model_args.h"
#include "framework/tokenizer/tokenizer.h"
#include "framework/tokenizer/tokenizer_args.h"

namespace xllm {

class VLMEngine final : public Engine {
 public:
  // create an engine with the given devices
  explicit VLMEngine(const runtime::Options& options,
                     std::shared_ptr<DistributedWorkerManager>
                         distributed_worker_manager = nullptr);

  ~VLMEngine() override = default;

  ForwardOutput step(BatchGroup& batch);

  bool init(MasterStatus master_status, const ModelInitCallback& prepare_model);

  std::vector<RawForwardOutput> collect_step_outputs(const BatchGroup& batch);

 private:
  // The target load never takes a residency callback: plain VLM targets do
  // not reserve weight pages, and routing the draft's callback here would
  // double-reserve the draft's model_id.
  bool init_model(MasterStatus master_status);
  bool allocate_kv_cache(const KVCacheCapacity& kv_cache_cap);
  void process_group_test();

  std::unique_ptr<VlmForwardInputFactory> forward_input_factory_;
};

}  // namespace xllm

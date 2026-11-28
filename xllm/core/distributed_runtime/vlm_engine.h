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

#include "common/macros.h"
#include "core/distributed_runtime/distributed_worker_manager.h"
#include "core/framework/batch/forward_input_factory.h"
#include "core/distributed_runtime/engine.h"
#include "core/runtime/forward_params.h"
#include "framework/batch/batch_group.h"

namespace xllm {

class KVTransferCoordinatorBase;

class VLMEngine final : public Engine {
 public:
  // create an engine with the given devices
  explicit VLMEngine(const runtime::Options& options);

  ~VLMEngine() = default;

  ForwardOutput step(BatchGroup& batch);

  bool init(MasterStatus master_status,
            const ModelInitCallback& prepare_model,
            std::shared_ptr<KVTransferCoordinatorBase> transfer_coordinator);

  std::vector<RawForwardOutput> collect_step_outputs(const BatchGroup& batch);

 private:
  bool init_model(MasterStatus master_status,
                  const ModelInitCallback& prepare_model);
  bool allocate_kv_cache(
      const KVCacheCapacity& kv_cache_cap,
      const VmmKVLayout* shared_vmm_layout,
      std::shared_ptr<KVTransferCoordinatorBase> transfer_coordinator);

  std::unique_ptr<ForwardInputFactory<VlmForwardInput>> forward_input_factory_;
};

}  // namespace xllm

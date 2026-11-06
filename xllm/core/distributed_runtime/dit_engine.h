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

#include "common/macros.h"
#include "core/distributed_runtime/distributed_worker_manager.h"
#include "core/distributed_runtime/engine.h"
#include "core/framework/batch/dit_forward_input_factory.h"
#include "framework/batch/dit_batch.h"

namespace xllm {

class DiTEngine final : public Engine {
 public:
  DiTEngine(const runtime::Options& options,
            std::shared_ptr<DistributedWorkerManager>
                distributed_worker_manager = nullptr);

  ~DiTEngine() = default;

  DiTForwardOutput step(std::vector<DiTBatch>& batch);

  bool init();

  // return the active activation memory
  std::vector<int64_t> get_active_activation_memory() const;

 protected:
  std::unique_ptr<DiTForwardInputFactory> forward_input_factory_;

 private:
  bool init_model();
};

}  // namespace xllm

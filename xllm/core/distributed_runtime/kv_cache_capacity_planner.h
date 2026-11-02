/* Copyright 2026 The xLLM Authors.

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

#include <torch/types.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/kv_cache/layout/kv_cache_estimation.h"

namespace xllm {

class ModelArgs;
class WorkerClient;
namespace runtime {
struct Options;
}

// Plans one engine's cache against its workers' memory. Model capacity rules
// stay in KVCacheEstimator; this runtime boundary supplies memory observations
// and prepares the shared VMM resources before manager construction.
class KVCacheCapacityPlanner final {
 public:
  KVCacheCapacityPlanner(
      const ModelArgs& model_args,
      const runtime::Options& options,
      torch::ScalarType dtype,
      int64_t world_size,
      const std::vector<std::shared_ptr<WorkerClient>>& worker_clients,
      bool is_multimodal = false,
      int64_t reserved_memory_bytes = 0);

  // A speculative pair defers VMM preparation until both state budgets and
  // tensor layouts are known.
  KVCacheCapacity plan(bool initialize_virtual_memory = true) const;

 private:
  KVCacheEstimateContext observe_memory() const;
  void prepare_virtual_memory(KVCacheCapacity& capacity) const;

  const ModelArgs& model_args_;
  const runtime::Options& options_;
  const std::vector<std::shared_ptr<WorkerClient>>& worker_clients_;
  torch::ScalarType dtype_;
  int64_t world_size_;
  bool is_multimodal_;
  int64_t reserved_memory_bytes_;
};

}  // namespace xllm

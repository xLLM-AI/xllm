/* Copyright 2025-2026 The xLLM Authors.

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
#include <optional>
#include <vector>

#include "core/framework/block/block_manager_pool.h"
#include "core/framework/model/model_args.h"
#include "core/kv_cache/layout/kv_cache_shape.h"
#include "core/kv_cache/storage/kv_cache_utils.h"
#include "core/runtime/options.h"

namespace xllm {

class KVCacheTransferCoordinatorBase;
class WorkerClient;

struct KVCacheManagerFactoryResult final {
  std::unique_ptr<BlockManagerPool> manager;
  KVCacheShape shape;
};

// One side of a joint target+draft capacity estimation. Holds references:
// callers keep the args/options alive for the duration of the estimation call.
struct KVCacheModelEstimate final {
  const ModelArgs& args;
  const runtime::Options& options;
  torch::ScalarType dtype;
};

class KVCacheManagerFactory final {
 public:
  // Joint target+draft estimation: both run on the same device list and share
  // one KV budget, so the returned capacity carries the shared block count
  // (workers derive the draft shape from the target shape). Pass draft only
  // when an external draft model is loaded; nullopt for unsupported configs
  // (virtual memory registers one model_id and cannot host a draft).
  static std::optional<KVCacheCapacity> estimate_speculative_capacity(
      const KVCacheModelEstimate& target,
      const std::optional<KVCacheModelEstimate>& draft,
      int64_t world_size,
      const std::vector<std::shared_ptr<WorkerClient>>& worker_clients,
      bool is_multimodal = false,
      int64_t reserved_memory_bytes = 0);

  static KVCacheManagerFactoryResult create(
      const KVCacheCapacity& kv_cache_capacity,
      const ModelArgs& model_args,
      int64_t world_size,
      torch::ScalarType dtype,
      BlockManagerPool::Options options,
      std::shared_ptr<KVCacheTransferCoordinatorBase> transfer_coordinator,
      int32_t dp_size = 1,
      std::optional<HostCacheValidationOptions> host_validation_options =
          std::nullopt);
};

}  // namespace xllm

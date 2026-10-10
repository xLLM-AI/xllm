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
#include "core/kv_cache/layout/kv_cache_shape.h"
#include "core/kv_cache/storage/kv_cache_utils.h"

namespace xllm {

class KVCacheTransferCoordinatorBase;
class ModelArgs;
class WorkerClient;
namespace runtime {
struct Options;
}

struct KVCacheManagerFactoryResult final {
  std::unique_ptr<KVCacheManager> manager;
  KVCacheShape shape;
};

class KVCacheManagerFactory final {
 public:
  static KVCacheCapacity estimate_capacity(
      const ModelArgs& model_args,
      const runtime::Options& options,
      torch::ScalarType dtype,
      int64_t world_size,
      const std::vector<std::shared_ptr<WorkerClient>>& worker_clients,
      bool is_multimodal = false);

  static KVCacheManagerFactoryResult create(
      const KVCacheCapacity& kv_cache_capacity,
      const ModelArgs& model_args,
      int64_t world_size,
      BlockManagerPool::Options options,
      std::shared_ptr<KVCacheTransferCoordinatorBase> transfer_coordinator,
      int32_t dp_size = 1,
      std::optional<HostCacheValidationOptions> host_validation_options =
          std::nullopt);
};

}  // namespace xllm

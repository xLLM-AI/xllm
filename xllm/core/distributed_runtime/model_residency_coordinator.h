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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/common/types.h"

namespace xllm {

class DistributedWorkerManager;
class ModelLoader;

// Coordinates one model's preparation, suspension and restoration across
// workers. Resource ownership and shared capacity live in separate services.
class ModelResidencyCoordinator final {
 public:
  struct Options {
    bool enabled = false;
    std::string model_id;
  };

  ModelResidencyCoordinator(
      Options options,
      std::shared_ptr<DistributedWorkerManager> distributed_worker_manager);

  // Prepare the model's page budget before workers load its weights.
  bool initialize_model(const ModelLoader& model_loader,
                        int64_t num_layers,
                        int32_t dp_size,
                        int32_t tp_size,
                        MasterStatus master_status);

  // Apply the initial sleep state after workers allocate the KV cache.
  bool finish_initialization(MasterStatus master_status);

  // Reports the shared page pool and all models' weight segments for
  // heartbeats.
  void get_virtual_memory_info(
      std::vector<size_t>& worker_free_phy_pages,
      std::unordered_map<std::string, std::vector<WeightSegment>>&
          model_weight_segments) const;

  bool sleep(MasterStatus master_status);
  bool wakeup(const WakeupOptions& options);

 private:
  bool suspend_memory(bool skip_weight_release);

  const Options options_;
  std::shared_ptr<DistributedWorkerManager> distributed_worker_manager_;
};

}  // namespace xllm

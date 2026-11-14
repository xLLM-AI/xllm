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

#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "common/types.h"
#include "core/distributed_runtime/distributed_worker_manager.h"
#include "core/distributed_runtime/model_residency_coordinator.h"
#include "framework/block/block_manager_pool.h"
#include "framework/model/model_args.h"
#include "framework/tokenizer/tokenizer.h"
#include "framework/tokenizer/tokenizer_args.h"
#include "runtime/options.h"
#include "runtime/worker_client.h"

namespace xllm {

class ModelLoader;
class KVCacheCapacity;
class KVCacheShape;

// Prepares model residency (virtual memory pages) before worker weights load.
using ModelInitCallback = std::function<
    bool(const ModelLoader&, int64_t, int32_t, int32_t, MasterStatus)>;

inline ModelInitCallback make_model_init_callback(
    ModelResidencyCoordinator* coordinator) {
  return [coordinator](const ModelLoader& model_loader,
                       int64_t num_layers,
                       int32_t dp_size,
                       int32_t tp_size,
                       MasterStatus status) {
    return coordinator->initialize_model(
        model_loader, num_layers, dp_size, tp_size, status);
  };
}

class Engine {
 public:
  explicit Engine(const runtime::Options& options,
                  std::shared_ptr<DistributedWorkerManager>
                      distributed_worker_manager = nullptr)
      : options_(options),
        distributed_worker_manager_(std::move(distributed_worker_manager)) {}

  ~Engine() = default;

  std::shared_ptr<DistributedWorkerManager> get_distributed_worker_manager()
      const {
    return distributed_worker_manager_;
  }

  const Tokenizer* tokenizer() const { return tokenizer_.get(); }

  BlockManagerPool* block_manager_pool() const {
    return block_manager_pool_.get();
  }

  const ModelArgs& model_args() const { return args_; }

  const TokenizerArgs& tokenizer_args() const { return tokenizer_args_; }

 protected:
  bool dispatch_model_load(const ModelLoader& model_loader,
                           const ModelArgs& model_args,
                           MasterStatus master_status,
                           const ModelInitCallback& prepare_model);

  bool init_draft_model(const ModelInitCallback& prepare_model);

  bool is_multimodal() const {
    return EngineType(options_.backend()) == EngineType::VLM;
  }

  void log_capacity_and_validate_chunked_prefill(
      const KVCacheCapacity& kv_cache_cap,
      bool enable_state_cache) const;

  bool allocate_kv_cache_on_workers(const KVCacheShape& kv_cache_shape,
                                    bool use_transfer_path = false);

  std::optional<KVCacheCapacity> init_draft_and_estimate_capacity(
      const ModelInitCallback& prepare_model);

  void setup_workers();

  bool load_model_to_workers(const std::string& model_path,
                             MasterStatus master_status);

  void process_group_test();

  ModelArgs args_;

  runtime::Options options_;

  std::vector<std::shared_ptr<WorkerClient>> worker_clients_;

  uint32_t dp_size() const { return options_.dp_size(); }

  // Effective TP width (MLU=dp_local; NPU=dp_local/cp).
  uint32_t dp_local_tp_size_ = 1;

  size_t worker_clients_num() const { return worker_clients_.size(); }

  TokenizerArgs tokenizer_args_;

  // Device allocations absent from the worker memory snapshot.
  int64_t reserved_memory_bytes_ = 0;

  std::unique_ptr<BlockManagerPool> block_manager_pool_;

  ModelArgs draft_args_;

  std::unique_ptr<Tokenizer> tokenizer_;

  torch::ScalarType dtype_;

  std::shared_ptr<DistributedWorkerManager> distributed_worker_manager_ =
      nullptr;
};

}  // namespace xllm

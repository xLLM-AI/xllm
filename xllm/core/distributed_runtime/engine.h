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
#include <mutex>
#include <optional>
#include <vector>

#include "common/types.h"
#include "core/kv_cache/layout/kv_cache_capacity.h"
#include "core/kv_cache/layout/vmm_kv_layout.h"
#include "core/kv_cache/manager/kv_cache_manager.h"
#include "framework/model/model_args.h"
#include "framework/tokenizer/tokenizer.h"
#include "framework/tokenizer/tokenizer_args.h"
#include "runtime/options.h"

namespace xllm {

class DistributedWorkerManager;
class KVCacheShape;
class ModelLoader;
class WorkerClient;

using ModelInitCallback = std::function<
    bool(const ModelLoader&, int64_t, int32_t, int32_t, MasterStatus)>;

class Engine {
 public:
  explicit Engine(const runtime::Options& options) : options_(options) {}

  std::shared_ptr<DistributedWorkerManager> get_distributed_worker_manager()
      const {
    return distributed_worker_manager_;
  }

  const Tokenizer* tokenizer() const { return tokenizer_.get(); }

  KVCacheManager* kv_cache_manager() const {
    return kv_cache_manager_.get();
  }

  const ModelArgs& model_args() const { return args_; }

  const TokenizerArgs& tokenizer_args() const { return tokenizer_args_; }

  bool start_profile();

  bool stop_profile();

 protected:
  ~Engine() = default;

  void reconcile_vocab_size(int64_t tokenizer_vocab_size);

  struct KVCachePlan {
    KVCacheCapacity capacity_;
    std::optional<VmmKVLayout> shared_vmm_layout_;
  };

  bool dispatch_model_load(const ModelLoader& model_loader,
                           MasterStatus master_status,
                           const ModelInitCallback& prepare_model);

  std::optional<ModelArgs> init_draft_model();

  bool is_multimodal() const {
    return EngineType(options_.backend()) == EngineType::VLM;
  }

  void log_capacity_and_validate_chunked_prefill(
      const KVCacheCapacity& kv_cache_cap,
      bool enable_state_cache) const;

  bool allocate_kv_cache_on_workers(const KVCacheShape& kv_cache_shape,
                                    bool use_transfer_path = false);

  std::optional<KVCachePlan> init_draft_and_estimate_capacity();

  void setup_workers();

  bool load_model_to_workers(const std::string& model_path,
                             MasterStatus master_status);

  void process_group_test();

  ModelArgs args_;

  runtime::Options options_;

  std::vector<std::shared_ptr<WorkerClient>> worker_clients_;

  uint32_t dp_size() const { return options_.dp_size(); }

  uint32_t dp_local_tp_size_ = 1;

  size_t worker_clients_num() const { return worker_clients_.size(); }

  TokenizerArgs tokenizer_args_;

  // Device allocations absent from the worker memory snapshot.
  int64_t reserved_memory_bytes_ = 0;

  std::unique_ptr<KVCacheManager> kv_cache_manager_;

  // tokenizer
  std::unique_ptr<Tokenizer> tokenizer_;

  torch::ScalarType dtype_;

  std::shared_ptr<DistributedWorkerManager> distributed_worker_manager_ =
      nullptr;

 private:
  bool profile_workers(bool is_start);
  std::mutex profile_mutex_;
};

}  // namespace xllm

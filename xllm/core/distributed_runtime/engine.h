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
#include "framework/block/block_manager_pool.h"
#include "framework/model/model_args.h"
#include "framework/tokenizer/tokenizer.h"
#include "framework/tokenizer/tokenizer_args.h"
#include "runtime/options.h"
#include "runtime/worker_client.h"

namespace xllm {

class ModelLoader;
struct KVCacheCapacity;

// Prepares model residency (virtual memory pages) before worker weights load.
using ModelInitCallback = std::function<
    bool(const ModelLoader&, int64_t, int32_t, int32_t, MasterStatus)>;

class Engine {
 public:
  explicit Engine(const runtime::Options& options,
                  std::shared_ptr<DistributedWorkerManager>
                      distributed_worker_manager = nullptr,
                  bool is_multimodal = false)
      : options_(options),
        distributed_worker_manager_(std::move(distributed_worker_manager)),
        is_multimodal_(is_multimodal) {}

  virtual ~Engine() = default;

  std::shared_ptr<DistributedWorkerManager> get_distributed_worker_manager()
      const {
    return distributed_worker_manager_;
  }

  // return the tokenizer
  const Tokenizer* tokenizer() const { return tokenizer_.get(); }

  bool is_multimodal() const { return is_multimodal_; }

  const runtime::Options& options() const { return options_; }

  // return the block manager
  BlockManagerPool* block_manager_pool() const {
    return block_manager_pool_.get();
  }

  // return the model args
  const ModelArgs& model_args() const { return args_; }

  // return the tokenizer args
  const TokenizerArgs& tokenizer_args() const { return tokenizer_args_; }

 protected:
  // Fans the model load out to every worker, running the residency
  // preparation callback first when set.
  bool dispatch_model_load(const ModelLoader& model_loader,
                           const ModelArgs& model_args,
                           MasterStatus master_status,
                           const ModelInitCallback& prepare_model);

  bool init_draft_model(const ModelInitCallback& prepare_model);

  // Draft leaves are always plain text LLMs even under a VLM target.
  runtime::Options draft_estimate_options() const;

  // Joint target+draft KV capacity estimation; nullopt on failure.
  std::optional<KVCacheCapacity> init_draft_and_estimate_capacity(
      const ModelInitCallback& prepare_model);

  // Idempotent; applies the speculative worker-sizing policy.
  void setup_workers();

  const bool is_multimodal_ = false;

  // model args
  ModelArgs args_;

  runtime::Options options_;

  // worker client which is used for call worker; unifies local and remote
  // worker access.
  std::vector<std::shared_ptr<WorkerClient>> worker_clients_;

  // Derived from options_; engines must call this instead of caching.
  uint32_t dp_size() const { return options_.dp_size(); }

  // Effective TP width (MLU=dp_local; NPU=dp_local/cp).
  uint32_t dp_local_tp_size_ = 1;

  size_t worker_clients_num() const { return worker_clients_.size(); }

  // Tokenizer args
  TokenizerArgs tokenizer_args_;

  // Future device allocations absent from the worker memory snapshot, set by
  // the concrete engine before capacity estimation.
  int64_t reserved_memory_bytes_ = 0;

  std::unique_ptr<BlockManagerPool> block_manager_pool_;

  // Loaded by init_draft_model when an external draft model is configured.
  ModelArgs draft_args_;

  // tokenizer
  std::unique_ptr<Tokenizer> tokenizer_;

  torch::ScalarType dtype_;

  // engine brpc server for multi-node serving
  std::shared_ptr<DistributedWorkerManager> distributed_worker_manager_ =
      nullptr;
};

}  // namespace xllm

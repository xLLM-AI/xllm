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

#include "common/macros.h"
#include "engine.h"
#include "framework/batch/batch_group.h"
#include "framework/block/block_manager_pool.h"
#include "framework/kv_cache/kv_cache_utils.h"
#include "framework/model/model_args.h"
#include "framework/tokenizer/tokenizer.h"
#include "framework/tokenizer/tokenizer_args.h"
#include "llm_engine.h"
#include "vlm_engine.h"

namespace xllm {

template <typename TargetEngine>
class SpeculativeEngineBase : public Engine {
 public:
  // create an engine with the given devices
  explicit SpeculativeEngineBase(const runtime::Options& options);

  ~SpeculativeEngineBase() override;

  bool init(MasterStatus master_status) override;

  bool init(MasterStatus master_status,
            const LLMEngine::ModelInitCallback& prepare_model,
            std::shared_ptr<KVCacheTransferCoordinatorBase>
                transfer_coordinator = nullptr);

  // step the engine forward
  ForwardOutput step(BatchGroup& batch);

  const Tokenizer* tokenizer() const override { return engine_->tokenizer(); }

  BlockManagerPool* block_manager_pool() const override {
    return engine_->block_manager_pool();
  }

  const ModelArgs& model_args() const override { return model_args_; }

  // Return the options used by the target engine.
  const runtime::Options& options() const { return engine_->options(); }

  std::shared_ptr<DistributedWorkerManager> get_distributed_worker_manager()
      const {
    return distributed_worker_manager_;
  }

  const TokenizerArgs& tokenizer_args() const override {
    return engine_->tokenizer_args();
  }

  void update_last_step_result(BatchGroup& batch);

 protected:
  SpeculativeEngineBase(const runtime::Options& options, bool use_draft_engine);

 private:
  bool init_model(MasterStatus master_status,
                  const LLMEngine::ModelInitCallback& prepare_model);

  bool allocate_kv_cache(
      std::shared_ptr<KVCacheTransferCoordinatorBase> transfer_coordinator);

  bool should_skip_external_draft_kv_cache() const;

  int64_t calculate_kv_cache(const KVCacheCapacity& target_kv_cache_cap,
                             const KVCacheCapacity& draft_kv_cache_cap) const;

  // dtype
  torch::ScalarType dtype_;

  // options
  const runtime::Options options_;

  // engine
  std::unique_ptr<TargetEngine> engine_;

  // draft engine
  std::unique_ptr<LLMEngine> draft_engine_;

  // whether this speculative engine uses an external draft engine
  const bool use_draft_engine_;

  ModelArgs model_args_;

  std::shared_ptr<DistributedWorkerManager> distributed_worker_manager_ =
      nullptr;
};

class SuffixSpeculativeEngine : public SpeculativeEngineBase<LLMEngine> {
 public:
  explicit SuffixSpeculativeEngine(const runtime::Options& options);
  ~SuffixSpeculativeEngine() override = default;
};

}  // namespace xllm

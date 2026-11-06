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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/framework/batch/batch_group.h"
#include "core/framework/block/block_manager_pool.h"
#include "core/framework/model/model_args.h"
#include "core/framework/tokenizer/tokenizer.h"
#include "core/runtime/forward_params.h"
#include "core/scheduler/scheduler_callbacks.h"
#include "scheduler/profile/profile_manager.h"

namespace xllm {

// Minimal Tokenizer stand-in for scheduler tests: nothing decodes, callers
// only need a non-null tokenizer pointer.
class FakeSchedulingTokenizer : public Tokenizer {
 public:
  bool encode(const std::string_view& /*text*/,
              std::vector<int32_t>* /*ids*/,
              bool /*add_special_tokens*/) const override {
    NOT_IMPLEMENTED();
  }
  std::string decode(const Slice<int32_t>& /*ids*/,
                     bool /*skip_special_tokens*/) const override {
    return "";
  }
  std::optional<int32_t> token_to_id(
      const std::string_view& /*token*/) const override {
    NOT_IMPLEMENTED();
  }
  std::string id_to_token(int32_t /*id*/) const override { NOT_IMPLEMENTED(); }
  size_t vocab_size() const override { NOT_IMPLEMENTED(); }
  std::unique_ptr<Tokenizer> clone() const override {
    return std::make_unique<FakeSchedulingTokenizer>();
  }
};

// Skeleton engine double for make_scheduler_callbacks consumers: owns a
// tokenizer, a block-manager pool and model args, and answers the scheduler's
// engine-facing queries. step/collect_step_outputs default to empty outputs;
// tests override behavior by subclassing or wrapping the callbacks. Subclasses
// may swap the pool and retarget model_args_ through the protected members.
class FakeSchedulingEngine {
 public:
  virtual ~FakeSchedulingEngine() = default;

  FakeSchedulingEngine()
      : tokenizer_(std::make_unique<FakeSchedulingTokenizer>()) {}

  explicit FakeSchedulingEngine(std::unique_ptr<BlockManagerPool> block_manager)
      : tokenizer_(std::make_unique<FakeSchedulingTokenizer>()),
        block_manager_(std::move(block_manager)) {}

  virtual ForwardOutput step(BatchGroup& /*batch*/) { return {}; }
  virtual std::vector<RawForwardOutput> collect_step_outputs(
      const BatchGroup& /*batch*/) {
    return {};
  }
  const Tokenizer* tokenizer() const { return tokenizer_.get(); }
  BlockManagerPool* block_manager_pool() const { return block_manager_.get(); }
  const ModelArgs& model_args() const { return model_args_; }

 protected:
  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<BlockManagerPool> block_manager_;
  ModelArgs model_args_;
};

// Derives a ProfileManager from any engine-shaped test double: flat
// dependency set plus callbacks wrapped from the engine's step/collect.
template <typename EngineT>
ProfileManager make_profile_manager(EngineT& engine,
                                    ProfileManager::Options options) {
  return ProfileManager(engine.block_manager_pool(),
                        &engine.model_args(),
                        options,
                        make_scheduler_callbacks(&engine));
}

}  // namespace xllm

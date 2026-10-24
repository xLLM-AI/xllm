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

#include "common/macros.h"
#include "core/framework/model/model_family_args.h"
#include "core/kv_cache/storage/kv_cache.h"
#include "executor_impl_factory.h"
#include "framework/model/causal_lm.h"
#include "framework/model/model_params_view.h"
#include "runtime/executor_impl.h"
#include "runtime/options.h"

namespace xllm {

class BaseExecutorImpl : public ExecutorImpl {
 public:
  BaseExecutorImpl(CausalLM* model,
                   const ModelArgsBundle& args,
                   const torch::Device& device,
                   const runtime::Options& options);

  ~BaseExecutorImpl() override = default;

  ModelOutput run(const torch::Tensor& tokens,
                  const torch::Tensor& positions,
                  std::vector<KVCache>& kv_caches,
                  const ModelParamsView& params) override;

 private:
  // not own
  CausalLM* model_;

  ModelArgsBundle args_;
  torch::Device device_;
  runtime::Options options_;
};
REGISTER_EXECUTOR("llm", BaseExecutorImpl);
}  // namespace xllm

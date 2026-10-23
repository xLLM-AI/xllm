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

#include <vector>

#include "core/framework/model/model_family_args.h"
#include "core/kv_cache/storage/kv_cache.h"
#include "framework/model/model_output.h"
#include "framework/model/rec_model.h"
#include "framework/model/rec_model_params.h"
#include "runtime/options.h"
#include "runtime/rec_executor_backend.h"
#include "runtime/rec_executor_impl_factory.h"

namespace xllm {

class RecExecutorImpl final : public RecExecutorBackend {
 public:
  RecExecutorImpl(RecModel* model,
                  const TransformerModelArgs& args,
                  const torch::Device& device,
                  const runtime::Options& options);

  ~RecExecutorImpl() override = default;

  ModelOutput run(const torch::Tensor& tokens,
                  const torch::Tensor& positions,
                  std::vector<KVCache>& kv_caches,
                  const RecModelParams& params) override;

 private:
  RecModel* model_;  // not owned
};

REGISTER_REC_EXECUTOR("rec", RecExecutorImpl);

}  // namespace xllm

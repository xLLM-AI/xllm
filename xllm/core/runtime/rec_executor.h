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

#include <memory>
#include <vector>

#include "core/framework/model/model_family_args.h"
#include "core/kv_cache/storage/kv_cache.h"
#include "framework/model/model_output.h"
#include "framework/model/rec_model.h"
#include "framework/model/rec_model_params.h"
#include "runtime/options.h"
#include "runtime/rec_executor_backend.h"

namespace xllm {

class RecExecutor final {
 public:
  RecExecutor(RecModel* model,
              const ModelArgs& args,
              const torch::Device& device,
              const runtime::Options& options);

  ~RecExecutor() = default;

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      RecModelParams& params);

 private:
  std::unique_ptr<RecExecutorBackend> impl_;
};

}  // namespace xllm

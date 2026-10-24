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

#include "runtime/rec_executor_impl.h"

#include "common/metrics.h"
#include "core/framework/model/model_family_args.h"

namespace xllm {

RecExecutorImpl::RecExecutorImpl(RecModel* model,
                                 const ModelArgs& /*args*/,
                                 const torch::Device& /*device*/,
                                 const runtime::Options& /*options*/)
    : model_(model) {}

ModelOutput RecExecutorImpl::run(const torch::Tensor& tokens,
                                 const torch::Tensor& positions,
                                 std::vector<KVCache>& kv_caches,
                                 const RecModelParams& params) {
  COUNTER_INC(num_model_execution_total_eager);
  return model_->forward(tokens, positions, kv_caches, params);
}

}  // namespace xllm

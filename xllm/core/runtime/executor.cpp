/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "executor.h"

#include "core/framework/config/execution_config.h"
#include "core/framework/config/model_config.h"
#include "executor_impl_factory.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm {

Executor::Executor(CausalLM* model,
                   const ModelArgs& args,
                   const torch::Device& device,
                   const runtime::Options& options) {
  const auto& model_config = ModelConfig::get_instance();
  std::string backend;
  if (ModelConfig::is_python_model_impl(model_config.model_impl())) {
    backend = "python";
  } else if (options.backend() != "vlm" && options.enable_graph()) {
    backend = Platform::type_str();
  } else {
    backend = options.backend();
  }
  impl_ = ExecutorImplFactory::get_instance().create_executor_impl(
      model, args, device, options, backend);
}

ForwardInput Executor::prepare_inputs(Batch& batch) {
  return impl_->prepare_inputs(batch);
}

ModelOutput Executor::forward(const torch::Tensor& tokens,
                              const torch::Tensor& positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelInputParams& params) {
  return impl_->run(tokens, positions, kv_caches, params);
}

void Executor::prepare_graph_input(const torch::Tensor& tokens,
                                   const torch::Tensor& positions,
                                   std::vector<KVCache>& kv_caches,
                                   const ModelInputParams& params) {
  impl_->prepare_graph_input(tokens, positions, kv_caches, params);
}

}  // namespace xllm

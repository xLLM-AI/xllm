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

#include "executor.h"

#include "core/framework/config/model_config.h"
#include "executor_impl_factory.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm {

Executor::Executor(CausalLM* model,
                   const ModelContext& context,
                   const runtime::Options& options) {
  const auto& model_config = ModelConfig::get_instance();
  std::string executor_type = model->executor_type();
  if (executor_type.empty()) {
    if (ModelConfig::is_python_model_impl(model_config.model_impl())) {
      executor_type = "python";
    } else if (options.backend() != "vlm" && options.enable_graph()) {
      executor_type = Platform::type_str();
    } else {
      executor_type = options.backend();
    }
  }
  auto executor_options = options;
  if (executor_type != "python" && executor_type != "mlu" &&
      options.is_draft_model()) {
    executor_options.num_speculative_tokens(context.num_speculative_tokens());
  }
  impl_ = ExecutorImplFactory::get_instance().create_executor_impl(
      model,
      context.get_model_args(),
      context.get_tensor_options().device(),
      executor_options,
      executor_type);
}

void Executor::set_dp_process_group(ProcessGroup* group) {
  impl_->set_dp_process_group(group);
}

ModelOutput Executor::forward(const torch::Tensor& tokens,
                              const torch::Tensor& positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelParamsView& params) {
  return impl_->run(tokens, positions, kv_caches, params);
}

void Executor::prepare_graph_input(const torch::Tensor& tokens,
                                   const torch::Tensor& positions,
                                   std::vector<KVCache>& kv_caches,
                                   const ModelParamsView& params) {
  impl_->prepare_graph_input(tokens, positions, kv_caches, params);
}

bool Executor::prepare_static_mtp_graph_tasks(
    const SpecVerifyGraphTaskSignal& signal,
    const Stream& signal_stream) {
  return impl_->prepare_static_mtp_graph_tasks(signal, signal_stream);
}

}  // namespace xllm

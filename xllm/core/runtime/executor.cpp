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

#include "core/framework/config/execution_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/model/model_family_args.h"
#include "core/runtime/py_executor_impl.h"
#include "executor_impl_factory.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm {

Executor::Executor(CausalLM* model,
                   const ModelArgsBundle& args,
                   const torch::Device& device,
                   const runtime::Options& options) {
  const auto& model_config = ModelConfig::get_instance();
  std::string executor_type = model->executor_type();
  if (executor_type.empty()) {
    if (ModelConfig::is_python_model_impl(model_config.model_impl())) {
      executor_type = "python";
    } else if (options.npu_kernel_backend() == "XLITE") {
      executor_type = "xlite";
    } else if (options.backend() != "vlm" && options.enable_graph()) {
      executor_type = Platform::type_str();
    } else {
      executor_type = options.backend();
    }
  }
  impl_ = ExecutorImplFactory::get_instance().create_executor_impl(
      model, args, device, options, executor_type);
}

bool Executor::supports_prepared_attention_metadata() const {
  return impl_->supports_prepared_attention_metadata();
}

bool Executor::supports_adaptive_unified_graph() const {
  return impl_->supports_adaptive_unified_graph();
}

void Executor::warmup_adaptive_unified_graph(
    Executor& draft_executor,
    const AdaptiveDSparkGraphTask& task,
    ProcessGroup* sampling_group) {
  impl_->warmup_adaptive_unified_graph(
      *draft_executor.impl_, task, sampling_group);
}

torch::Tensor Executor::run_adaptive_unified_graph(
    Executor& draft_executor,
    const AdaptiveDSparkGraphTask& task,
    ProcessGroup* sampling_group) {
  return impl_->run_adaptive_unified_graph(
      *draft_executor.impl_, task, sampling_group);
}

void Executor::prepare_attention_metadata(std::vector<KVCache>& kv_caches,
                                          ModelParamsView& params) {
  impl_->prepare_attention_metadata(kv_caches, params);
}

void Executor::warmup_prepared_graph(const torch::Tensor& tokens,
                                     const torch::Tensor& positions,
                                     std::vector<KVCache>& kv_caches,
                                     const ModelParamsView& params) {
  impl_->warmup_prepared_graph(tokens, positions, kv_caches, params);
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

PyExecutorImpl* Executor::python_impl() {
  return dynamic_cast<PyExecutorImpl*>(impl_.get());
}

}  // namespace xllm

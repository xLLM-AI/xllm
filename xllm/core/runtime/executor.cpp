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

#include "core/common/extension_registry.h"
#include "core/framework/config/model_config.h"
#include "core/runtime/vlm_executor_impl.h"
#include "executor_impl_factory.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm {

namespace {

std::string select_executor_type(const CausalLM& model,
                                 const ModelContext& context,
                                 const runtime::Options& options) {
  std::string executor_type = model.executor_type();
  if (!executor_type.empty()) {
    return executor_type;
  }
  if (ModelConfig::is_python_model_impl(context.get_model_impl())) {
    return "python";
  }
  if (options.backend() != "vlm" && options.enable_graph()) {
    return Platform::type_str();
  }
  return options.backend();
}

}  // namespace

Executor::Executor(CausalLM* model,
                   const ModelContext& context,
                   const runtime::Options& options)
    : impl_(ExecutorImplFactory::get_instance().create_executor_impl(
          model,
          context.get_model_args(),
          context.get_tensor_options().device(),
          options,
          select_executor_type(*model, context, options))) {}

Executor::Executor(CausalVLM* model,
                   const ModelContext& context,
                   const runtime::Options& options) {
  auto& language_model = model->language_model();
  const std::string executor_type =
      select_executor_type(language_model, context, options);
  const auto& args = context.get_model_args();
  const auto device = context.get_tensor_options().device();
  if (executor_type == "vlm" &&
      !ExtensionRegistry::get_instance().has_implementation(
          options.extension_impl(), /*component=*/"executor")) {
    impl_ = std::make_unique<VlmExecutorImpl>(*model, args, device, options);
  } else {
    impl_ = ExecutorImplFactory::get_instance().create_executor_impl(
        &language_model, args, device, options, executor_type);
  }
}

void Executor::set_dp_process_group(ProcessGroup* group) {
  impl_->set_dp_process_group(group);
}

void Executor::clear_multimodal_cache() { impl_->clear_multimodal_cache(); }

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

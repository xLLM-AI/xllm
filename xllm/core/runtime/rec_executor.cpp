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

#include "runtime/rec_executor.h"

#include <string>

#include "core/framework/config/model_config.h"
#include "core/framework/model/model_family_args.h"
#include "core/framework/model/rec_model_params.h"
#include "platform/platform.h"
#include "runtime/rec_executor_impl_factory.h"

namespace xllm {

RecExecutor::RecExecutor(RecModel* model,
                         const TransformerModelArgs& args,
                         const torch::Device& device,
                         const runtime::Options& options) {
  const auto& model_config = ModelConfig::get_instance();
  std::string backend;
  CHECK(!ModelConfig::is_python_model_impl(model_config.model_impl()))
      << "Rec execution requires a native Rec model implementation.";
  CHECK_NE(options.npu_kernel_backend(), "XLITE")
      << "XLITE does not support native Rec inputs.";
  CHECK(!Platform::is_ilu() || !options.enable_graph())
      << "ILU does not support native Rec graph execution. "
         "Set --enable_graph=false.";
  if (options.backend() != "vlm" && options.enable_graph()) {
    backend = Platform::type_str();
  } else {
    backend = options.backend();
  }
  impl_ = RecExecutorImplFactory::get_instance().create_executor_impl(
      model, args, device, options, backend);
}

ModelOutput RecExecutor::forward(const torch::Tensor& tokens,
                                 const torch::Tensor& positions,
                                 std::vector<KVCache>& kv_caches,
                                 RecModelParams& params) {
  return impl_->run(tokens, positions, kv_caches, params);
}

}  // namespace xllm

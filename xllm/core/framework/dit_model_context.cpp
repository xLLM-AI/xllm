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

#include "core/framework/dit_model_context.h"

#include <torch/torch.h>
#if defined(USE_NPU)
#ifdef TORCH_HIGHER_THAN_PTA6
// #include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif
#include <torch_npu/csrc/libs/init_npu.h>
#endif

namespace xllm {
DiTModelContext::DiTModelContext(
    const ParallelArgs& input_parallel_args,
    const std::unordered_map<std::string, ModelArgs>& model_args,
    const std::unordered_map<std::string, QuantArgs>& quant_args,
    const torch::TensorOptions& tensor_options,
    const DiTCacheConfig& dit_config,
    const std::string& model_type)
    : parallel_args_(input_parallel_args),
      model_args_(std::move(model_args)),
      quant_args_(std::move(quant_args)),
      tensor_options_(tensor_options),
      dit_config_(dit_config),
      model_type_(model_type) {
#if defined(USE_NPU)
  int32_t device_id = tensor_options.device().index();
  void* stream = c10_npu::getCurrentNPUStream(device_id).stream();
  atb::CreateContext(&context_);
  context_->SetExecuteStream(stream);
  context_->SetAsyncTilingCopyStatus(true);
#endif
}

const ModelArgs& DiTModelContext::get_model_args(
    const std::string& component) const {
  const auto& itor = model_args_.find(component);
  if (itor != model_args_.end()) {
    return itor->second;
  } else {
    LOG(FATAL) << "model args not found, component:" << component;
    static ModelArgs args;
    return args;
  }
}

const QuantArgs& DiTModelContext::get_quant_args(
    const std::string& component) const {
  const auto& itor = quant_args_.find(component);
  if (itor != quant_args_.end()) {
    return itor->second;
  } else {
    LOG(FATAL) << "qunat args not found, component:" << component;
    static QuantArgs args;
    return args;
  }
}

#if defined(USE_NPU) || defined(USE_CUDA) || defined(USE_MLU) || \
    defined(USE_DCU)
ModelContext DiTModelContext::get_model_context(
    const std::string& component) const {
#if defined(USE_NPU)
  return ModelContext(parallel_args_,
                      get_model_args(component),
                      get_quant_args(component),
                      tensor_options_,
                      context_);
#else
  return ModelContext(parallel_args_,
                      get_model_args(component),
                      get_quant_args(component),
                      tensor_options_);
#endif
}
#endif

}  // namespace xllm

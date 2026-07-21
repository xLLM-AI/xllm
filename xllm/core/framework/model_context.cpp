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

#include "core/framework/model_context.h"

#include <torch/torch.h>

#include "common/global_flags.h"
#include "core/framework/config/execution_config.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "util/env_var.h"
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

namespace {

bool should_enable_async_tiling_copy_stream() {
  // ATB copy-stream teardown is not reversible for the same context on the
  // current CANN/PTA stack, so contexts that may enter graph capture must not
  // pre-create the helper stream.
  if (::xllm::ExecutionConfig::get_instance().enable_graph()) {
    return false;
  }
  return util::get_bool_env("ATB_USE_TILING_COPY_STREAM", false);
}

}  // namespace

ModelContext::ModelContext(const ParallelArgs& input_parallel_args,
                           const ModelArgs& model_args,
                           const QuantArgs& quant_args,
                           const torch::TensorOptions& tensor_options)
    : parallel_args_(input_parallel_args),
      model_args_(model_args),
      quant_args_(quant_args),
      tensor_options_(tensor_options) {
#if defined(USE_NPU)
  int32_t device_id = tensor_options.device().index();
  aclError ret = aclrtSetDevice(device_id);
  atb::CreateContext(&context_);
  void* stream = c10_npu::getCurrentNPUStream(device_id).stream();
  context_->SetExecuteStream(stream);
  if (should_enable_async_tiling_copy_stream()) {
    context_->SetAsyncTilingCopyStatus(true);
  }
  atb_workspace_ = std::make_shared<AtbWorkspace>(tensor_options.device());
#endif
  derive_optimization_config();
}

#if defined(USE_NPU)
ModelContext::ModelContext(const ParallelArgs& input_parallel_args,
                           const ModelArgs& model_args,
                           const QuantArgs& quant_args,
                           const torch::TensorOptions& tensor_options,
                           atb::Context* context)
    : parallel_args_(input_parallel_args),
      model_args_(model_args),
      quant_args_(quant_args),
      tensor_options_(tensor_options),
      context_(context) {}
#endif

ModelContext ModelContext::with_parallel_args(
    const ParallelArgs& parallel_args) const {
#if defined(USE_NPU)
  ModelContext derived(
      parallel_args, model_args_, quant_args_, tensor_options_, context_);
  derived.atb_workspace_ = atb_workspace_;
#else
  ModelContext derived(
      parallel_args, model_args_, quant_args_, tensor_options_);
#endif
  derived.model_id_ = model_id_;
  derived.optimization_config_ = optimization_config_;
  return derived;
}

void ModelContext::derive_optimization_config() {
  // default disable fused kernel
  optimization_config_.enable_fused_spec_kernel = false;
  optimization_config_.enable_fused_mla_kernel = false;
  optimization_config_.enable_fused_indexer_qk = false;
  optimization_config_.enable_spec_token_broadcast = false;

  // determine whether to enable fused kernel based on backend
  if (Platform::is_npu()) {
    // Unify speculative sampling results across TP ranks to guard against
    // per-rank RNG divergence under enable_schedule_overlap.
    optimization_config_.enable_spec_token_broadcast = true;
  } else if (Platform::is_dcu()) {
    // DCU currently uses the unfused speculative sampling path.
    optimization_config_.enable_fused_spec_kernel = false;
  } else if (Platform::is_mlu()) {
    // TODO: enable fused spec kernel for mlu backend
    // The current implementation of fused spec kernel is not stable.
    optimization_config_.enable_fused_spec_kernel = false;
    // fused mla only support smoothquant mode
    optimization_config_.enable_fused_mla_kernel =
        quant_args_.quant_method() == kQuantMethodSmoothquant;
    // The current implementation of fused indexer qk is missing the
    //  weights and bias loading.
    optimization_config_.enable_fused_indexer_qk = true;
    // Unify speculative sampling results across TP ranks to guard against
    // per-rank RNG divergence.
    optimization_config_.enable_spec_token_broadcast = true;
  }
}

}  // namespace xllm

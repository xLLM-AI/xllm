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

#include "fused_moe.h"

#include <glog/logging.h>

namespace xllm {
namespace layer {

FusedMoEImpl::FusedMoEImpl(const ModelArgs& /*model_args*/,
                           const FusedMoEArgs& /*moe_args*/,
                           const QuantArgs& /*quant_args*/,
                           const ParallelArgs& /*parallel_args*/,
                           const torch::TensorOptions& /*options*/) {
  NOT_IMPLEMENTED_WITH_MSG(
      "FusedMoE is not supported for this backend. Please use CUDA, MLU or "
      "ILU backend for MoE models.");
}

torch::Tensor FusedMoEImpl::forward_experts(
    const torch::Tensor& /*hidden_states*/,
    const torch::Tensor& /*router_logits*/,
    bool /*enable_all2all_communication*/) {
  NOT_IMPLEMENTED_WITH_MSG(
      "FusedMoE is not supported for this backend. Please use CUDA, MLU or "
      "ILU backend for MoE models.");
  return torch::Tensor();
}

torch::Tensor FusedMoEImpl::forward(const torch::Tensor& /*hidden_states*/,
                                    const ModelInputParams& /*input_params*/) {
  NOT_IMPLEMENTED_WITH_MSG(
      "FusedMoE is not supported for this backend. Please use CUDA, MLU or "
      "ILU backend for MoE models.");
  return torch::Tensor();
}

void FusedMoEImpl::load_state_dict(const StateDict& /*state_dict*/) {
  NOT_IMPLEMENTED_WITH_MSG(
      "FusedMoE is not supported for this backend. Please use CUDA, MLU or "
      "ILU backend for MoE models.");
}

}  // namespace layer
}  // namespace xllm

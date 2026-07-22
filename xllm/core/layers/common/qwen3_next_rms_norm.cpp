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

#include "qwen3_next_rms_norm.h"

#include <glog/logging.h>

#include "xllm/core/kernels/ops_api.h"

namespace xllm {
namespace layer {

Qwen3NextRMSNormImpl::Qwen3NextRMSNormImpl(int64_t dim,
                                           double eps,
                                           const torch::TensorOptions& options)
    : norm_dim_(dim), eps_(eps) {
  weight_ = register_parameter("weight", torch::empty({dim}, options), false);
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>>
Qwen3NextRMSNormImpl::forward(torch::Tensor& input,
                              std::optional<torch::Tensor> residual) {
  if (!residual.has_value()) {
    // No residual: use original gemma_rms_norm
    xllm::kernel::GemmaRMSNormParams norm_params;
    norm_params.x = input;
    norm_params.gamma = weight_;
    norm_params.epsilon = eps_;
    xllm::kernel::gemma_rms_norm(norm_params);
    return std::make_tuple(norm_params.norm_out, std::nullopt);
  }

  // With residual: use the fused NPU AddRmsNorm path.
  xllm::kernel::FusedLayerNormParams fused_params;
  fused_params.input = input;
  fused_params.residual = residual;
#if !defined(USE_NPU)
  // NPU backend allocates outputs internally in npu::add_rms_norm,
  // so skip pre-allocation to avoid wasted memory.
  fused_params.output = torch::empty_like(input);
  fused_params.residual_out = torch::empty_like(residual.value());
#endif
#if defined(USE_NPU)
  fused_params.weight = weight_;
  fused_params.add_gamma_offset = true;
#else
  fused_params.weight = 1.0 + weight_;
#endif
  fused_params.eps = eps_;
  fused_params.mode = "rmsnorm";
  xllm::kernel::fused_layernorm(fused_params);
  return std::make_tuple(fused_params.output, fused_params.residual_out);
}

void Qwen3NextRMSNormImpl::load_state_dict(const StateDict& state_dict) {
  LOAD_WEIGHT(weight);
}

}  // namespace layer
}  // namespace xllm

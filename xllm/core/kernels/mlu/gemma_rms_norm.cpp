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

#include "mlu_ops_api.h"

namespace xllm::kernel::mlu {
torch::Tensor gemma_rms_norm(const torch::Tensor& x,
                             const torch::Tensor& gamma,
                             double eps,
                             torch::Tensor& norm_out,
                             const std::optional<torch::Tensor>& residual,
                             std::optional<torch::Tensor>& residual_out) {
  norm_out = torch::empty_like(x);
  if (residual.has_value() && !residual_out.has_value()) {
    residual_out = residual.value();
  }
  const bool store_output_before_norm = residual.has_value();
  tmo::torch_api::fused_layernorm(x,
                                  norm_out,
                                  residual,
                                  gamma,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                                  residual_out,
                                  std::nullopt,
                                  std::nullopt,
                                  "rmsnorm",
                                  eps,
                                  store_output_before_norm,
                                  /*store_output_after_norm=*/false,
                                  /*dynamic_quant=*/false,
                                  /*mx_quant=*/false,
                                  /*transpose_4d_1_2=*/false,
                                  /*gamma_add_coef=*/1.0,
                                  /*z_gated=*/std::nullopt,
                                  /*gated_after_norm=*/false);
  return norm_out;
}
}  // namespace xllm::kernel::mlu

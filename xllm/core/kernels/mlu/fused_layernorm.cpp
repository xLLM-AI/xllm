/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

void fused_layernorm(const torch::Tensor& input,
                     torch::Tensor& output,
                     const std::optional<torch::Tensor>& residual,
                     const std::optional<torch::Tensor>& weight,
                     const std::optional<torch::Tensor>& beta,
                     const std::optional<torch::Tensor>& bias,
                     const std::optional<torch::Tensor>& quant_scale,
                     const std::optional<torch::Tensor>& residual_out,
                     const std::optional<torch::Tensor>& smooth_quant_scale,
                     const std::optional<torch::Tensor>& normed_out,
                     const std::string& mode,
                     double eps,
                     bool store_output_before_norm,
                     bool store_output_after_norm,
                     bool dynamic_quant) {
  tmo::torch_api::fused_layernorm(input,
                                  output,
                                  residual,
                                  weight,
                                  beta,
                                  bias,
                                  quant_scale,
                                  residual_out,
                                  smooth_quant_scale,
                                  normed_out,
                                  mode,
                                  eps,
                                  store_output_before_norm,
                                  store_output_after_norm,
                                  dynamic_quant);
}

}  // namespace xllm::kernel::mlu

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

#include <glog/logging.h>

#include "mlu_ops_api.h"

namespace xllm::kernel::mlu {
torch::Tensor gated_layer_norm(torch::Tensor& x,
                               const torch::Tensor& weight,
                               const torch::Tensor& bias,
                               double eps,
                               const std::optional<torch::Tensor>& gate,
                               int64_t group_size,
                               bool norm_before_gate) {
  (void)bias;
  CHECK_EQ(group_size, x.size(-1))
      << "MLU fused gated_layer_norm only supports whole-last-dim groups";

  torch::Tensor output = torch::empty_like(x);
  tmo::torch_api::fused_layernorm(x,
                                  output,
                                  std::nullopt,
                                  weight,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                                  "rmsnorm",
                                  eps,
                                  /*store_output_before_norm=*/false,
                                  /*store_output_after_norm=*/false,
                                  /*dynamic_quant=*/false,
                                  /*mx_quant=*/false,
                                  /*transpose_4d_1_2=*/false,
                                  /*gamma_add_coef=*/0.0,
                                  gate,
                                  /*gated_after_norm=*/norm_before_gate);
  return output;
}
}  // namespace xllm::kernel::mlu

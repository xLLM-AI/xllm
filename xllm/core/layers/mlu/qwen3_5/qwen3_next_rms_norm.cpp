/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "layers/mlu/qwen3_5/qwen3_next_rms_norm.h"

#include "kernels/ops_api.h"

namespace xllm {
namespace layer {

std::tuple<torch::Tensor, std::optional<torch::Tensor>>
Qwen3NextRMSNormMluImpl::forward(torch::Tensor& input,
                                 std::optional<torch::Tensor> residual) {
  kernel::GemmaRMSNormParams norm_params;
  norm_params.x = input;
  norm_params.gamma = weight();
  norm_params.epsilon = eps();
  norm_params.residual = residual;
  if (residual.has_value()) {
    norm_params.residual_out = residual.value();
  }
  kernel::gemma_rms_norm(norm_params);
  return std::make_tuple(norm_params.norm_out, norm_params.residual_out);
}

}  // namespace layer
}  // namespace xllm

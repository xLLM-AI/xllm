/* Copyright 2026 The xLLM Authors.

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

#include "ada_layer_norm.h"

#include "kernels/ops_api.h"

namespace xllm {
namespace layer {

AdaLayerNormImpl::AdaLayerNormImpl(int64_t dim,
                                   double eps,
                                   bool elementwise_affine,
                                   const torch::TensorOptions& options)
    : norm_dim_(dim), eps_(eps), elementwise_affine_(elementwise_affine) {
  if (elementwise_affine_) {
    weight_ = register_parameter(
        "weight", torch::ones({dim}, options), /*requires_grad=*/false);
    bias_ = register_parameter(
        "bias", torch::zeros({dim}, options), /*requires_grad=*/false);
  }
}

torch::Tensor AdaLayerNormImpl::forward(const torch::Tensor& input,
                                        const torch::Tensor& scale,
                                        const torch::Tensor& shift) {
  xllm::kernel::AdaLayerNormParams params;
  params.input = input;
  params.scale = scale;
  params.shift = shift;
  if (weight_.defined()) {
    params.weight = weight_;
  }
  if (bias_.defined()) {
    params.bias = bias_;
  }
  params.eps = eps_;

  return xllm::kernel::fused_adalayer_norm(params);
}

void AdaLayerNormImpl::load_state_dict(const StateDict& state_dict) {
  if (elementwise_affine_) {
    LOAD_WEIGHT(weight);
    LOAD_WEIGHT(bias);
  }
}

}  // namespace layer
}  // namespace xllm

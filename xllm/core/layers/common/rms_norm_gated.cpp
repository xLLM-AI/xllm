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

#include "rms_norm_gated.h"

#include <glog/logging.h>

#include "framework/state_dict/utils.h"
#include "xllm/core/kernels/ops_api.h"

namespace xllm {
namespace layer {

RmsNormGatedImpl::RmsNormGatedImpl(int64_t dim,
                                   double eps,
                                   const torch::TensorOptions& options)
    : norm_dim_(dim), eps_(eps) {
  weight_ = register_parameter(
      "weight", torch::empty({dim}, options), /*requires_grad=*/false);
}

torch::Tensor RmsNormGatedImpl::forward(torch::Tensor& input,
                                        std::optional<torch::Tensor> gate) {
  xllm::kernel::GatedLayerNormParams params;
  params.x = input;
  params.weight = weight_;
  torch::Tensor bias;
  params.bias = bias;
  params.eps = eps_;
  if (gate.has_value()) {
    params.z = gate;
  }
  params.group_size = input.size(-1);
  params.is_rms_norm = true;
  auto ret = xllm::kernel::gated_layer_norm(params);
  return ret;
}

void RmsNormGatedImpl::load_state_dict(const StateDict& state_dict) {
  LOAD_WEIGHT(weight);
}

}  // namespace layer
}  // namespace xllm

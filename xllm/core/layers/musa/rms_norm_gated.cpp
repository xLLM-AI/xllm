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

#include "layers/musa/rms_norm_gated.h"

#include <glog/logging.h>

#include "framework/state_dict/utils.h"
#include "kernels/musa/gdn_ops.h"
#include "kernels/ops_api.h"

namespace xllm::layer::musa {

RmsNormGatedImpl::RmsNormGatedImpl(int64_t dim,
                                   int64_t max_rows,
                                   double eps,
                                   const torch::TensorOptions& options)
    : max_rows_(max_rows), eps_(eps) {
  CHECK_GT(max_rows_, 0);
  weight_ = register_parameter(
      "weight", torch::empty({dim}, options), /*requires_grad=*/false);
}

torch::Tensor RmsNormGatedImpl::forward(torch::Tensor& input,
                                        std::optional<torch::Tensor> gate) {
  xllm::kernel::GatedLayerNormParams params;
  params.x = input;
  params.weight = weight_;
  params.bias = torch::Tensor();
  params.eps = eps_;
  if (gate.has_value()) {
    params.z = gate;
  }
  params.group_size = input.size(-1);
  params.is_rms_norm = true;

  std::optional<torch::Tensor> output_buf = std::nullopt;
  if (input.dim() >= 1 && input.numel() > 0 && input.stride(-1) == 1) {
    const int64_t last_dim = input.size(-1);
    const int64_t rows = input.numel() / last_dim;
    if (rows <= max_rows_) {
      const bool needs_realloc =
          !output_buf_.defined() || output_buf_.device() != input.device() ||
          output_buf_.scalar_type() != input.scalar_type() ||
          output_buf_.dim() != 2 || output_buf_.size(0) < rows ||
          output_buf_.size(1) != last_dim;
      if (needs_realloc) {
        output_buf_ = torch::empty({max_rows_, last_dim}, input.options());
      }
      output_buf = output_buf_.narrow(/*dim=*/0, /*start=*/0, /*length=*/rows)
                       .view(input.sizes());
    }
  }

  return xllm::kernel::musa::gated_layer_norm(params, output_buf);
}

void RmsNormGatedImpl::load_state_dict(const StateDict& state_dict) {
  LOAD_WEIGHT(weight);
}

}  // namespace xllm::layer::musa

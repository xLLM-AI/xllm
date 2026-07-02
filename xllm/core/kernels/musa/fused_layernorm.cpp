/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <torch/torch.h>

#include <optional>
#include <string>
#include <vector>

#include "ATen/Tensor.h"
#include "MTTOplib/Ops.h"

namespace xllm::kernel::musa {
void fused_layernorm(const torch::Tensor& input,
                     torch::Tensor& output,
                     const std::optional<torch::Tensor>& residual,
                     const torch::Tensor& weight,
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
  xllm_musa::FusedRMSNorm(input, output, weight, eps);
}

}

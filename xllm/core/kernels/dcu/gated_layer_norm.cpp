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

#include "dcu_ops_api.h"

namespace xllm::kernel::dcu {

// Torch fallback: DCU does not have a fused gated_layer_norm kernel yet.
torch::Tensor gated_layer_norm(torch::Tensor& x,
                               const torch::Tensor& weight,
                               const torch::Tensor& bias,
                               double eps,
                               const std::optional<torch::Tensor>& gate,
                               int64_t group_size,
                               bool norm_before_gate) {
  auto origin_dtype = x.dtype();
  auto x_fp32 = x.to(torch::kFloat32);
  auto weight_fp32 = weight.to(torch::kFloat32);
  torch::Tensor gate_value;
  if (gate.has_value()) {
    gate_value = gate.value().to(torch::kFloat32);
  }

  if (gate.has_value() && !norm_before_gate) {
    x_fp32 = x_fp32 * torch::silu(gate_value);
  }

  int64_t hidden_size = x_fp32.size(-1);
  torch::Tensor normalized;
  if (group_size == hidden_size) {
    auto variance = x_fp32.pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
    normalized = x_fp32 * torch::rsqrt(variance + eps);
    normalized = normalized * weight_fp32;
  } else {
    int64_t num_groups = hidden_size / group_size;
    auto x_grouped = x_fp32.reshape({-1, num_groups, group_size});
    auto variance = x_grouped.pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
    auto normalized_grouped = x_grouped * torch::rsqrt(variance + eps);
    normalized = normalized_grouped.reshape({-1, num_groups * group_size});
    normalized = normalized * weight_fp32;
  }

  if (gate.has_value() && norm_before_gate) {
    normalized = normalized * torch::silu(gate_value);
  }

  return normalized.to(origin_dtype);
}

}  // namespace xllm::kernel::dcu

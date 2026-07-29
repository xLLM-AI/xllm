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

#pragma once

// Two shared quant helpers used by both LLM (linear.cpp) and DiT
// (add_matmul.cpp, dit_parallel_linear.h).

#include <glog/logging.h>
#include <torch/torch.h>

#include <optional>
#include <string>
#include <vector>

#include "core/framework/quant_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "kernels/ops_api.h"

namespace xllm {
namespace layer {

inline void resolve_weight_quant_method_for_linear_load(
    const QuantArgs& quant_args,
    const StateDict& state_dict,
    const std::vector<std::string>* local_prefixes,
    std::optional<std::string>& resolved_weight_quant_method) {
  const auto prefixes =
      local_prefixes == nullptr ? std::vector<std::string>{} : *local_prefixes;
  auto resolved =
      quant_args.get_quant_method_from_prefixes(state_dict, prefixes);
  if (resolved.has_value()) {
    resolved_weight_quant_method = resolved.value();
    return;
  }
  if (quant_args.is_compressed_tensors_w8a8_dynamic()) {
    bool is_w8a8_dynamic = false;
    if (prefixes.empty()) {
      torch::Tensor weight = state_dict.get_tensor("weight");
      is_w8a8_dynamic = state_dict.has("weight_scale") && weight.defined() &&
                        weight.scalar_type() == torch::kInt8;
    } else {
      is_w8a8_dynamic = true;
      for (const std::string& prefix : prefixes) {
        torch::Tensor weight = state_dict.get_tensor(prefix + "weight");
        if (!state_dict.has(prefix + "weight_scale") || !weight.defined() ||
            weight.scalar_type() != torch::kInt8) {
          is_w8a8_dynamic = false;
          break;
        }
      }
    }
    if (is_w8a8_dynamic) {
      resolved_weight_quant_method = "w8a8_dynamic";
      return;
    }
  }
  if (!quant_args.quant_descs().empty()) {
    LOG(WARNING) << "[LinearLoad][QuantMethod] quant_descs is not empty but "
                    "quant method was not resolved from state_dict prefixes. "
                    "state_dict.prefix="
                 << state_dict.prefix();
  }
  resolved_weight_quant_method = std::nullopt;
}

inline torch::Tensor npu_w8a8_dynamic_linear_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale,
    const std::optional<torch::Tensor>& bias,
    at::ScalarType output_dtype,
    const std::optional<torch::Tensor>& weight_offset = std::nullopt) {
  kernel::NpuQuantizeParams quant_params;
  quant_params.input = input;
  // quant_params.dst_type = at::kChar;

  torch::Tensor quantized_input;
  std::optional<torch::Tensor> pertoken_scale;
  std::tie(quantized_input, pertoken_scale) =
      kernel::dynamic_quant(quant_params);
  CHECK(pertoken_scale.has_value() && pertoken_scale->defined())
      << "dynamic_quant must return per-token scale for w8a8_dynamic.";

  kernel::QuantMatmulParams quant_matmul_params;
  quant_matmul_params.x1 = quantized_input;
  quant_matmul_params.x2 = weight;
  quant_matmul_params.transpose2 = true;
  quant_matmul_params.scale = weight_scale;
  quant_matmul_params.pertoken_scale = pertoken_scale->reshape({-1});
  quant_matmul_params.output_dtype = output_dtype;
  if (weight_offset.has_value() && weight_offset->defined()) {
    quant_matmul_params.offset = weight_offset;
  }
  if (bias.has_value() && bias->defined()) {
    quant_matmul_params.bias = bias;
  }
  auto output = kernel::quant_matmul(quant_matmul_params);
  return output;
}

}  // namespace layer
}  // namespace xllm

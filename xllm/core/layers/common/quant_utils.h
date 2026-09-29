/* Copyright 2026 The xLLM Authors.

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

#pragma once

// Shared quantization loading and execution helpers for Linear, MoE and DiT.

#include <glog/logging.h>
#include <torch/types.h>

#include <optional>
#include <string>
#include <vector>

#include "core/framework/quant_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "kernels/ops_api.h"

namespace xllm {
namespace layer {

inline void check_ct_scale(const torch::Tensor& scale,
                           const std::string& name) {
  if (!scale.defined()) {
    return;
  }
  CHECK(scale.is_floating_point()) << name;
  CHECK(scale.dim() == 1 || (scale.dim() == 2 && scale.size(1) == 1))
      << "Compressed-tensors channel scale must be [N] or [N, 1]: " << name;
  CHECK(torch::isfinite(scale).all().item<bool>() &&
        scale.gt(0).all().item<bool>())
      << "Compressed-tensors scale must be finite and positive: " << name;
}

inline void resolve_weight_quant_method_for_linear_load(
    const QuantArgs& quant_args,
    const StateDict& state_dict,
    const std::vector<std::string>* local_prefixes,
    std::optional<std::string>& resolved_weight_quant_method) {
  const auto prefixes = local_prefixes == nullptr || local_prefixes->empty()
                            ? std::vector<std::string>{""}
                            : *local_prefixes;
  if (quant_args.is_compressed_tensors_w8a8_dynamic()) {
    for (const std::string& prefix : prefixes) {
      CHECK(!state_dict.has(prefix + "smooth"))
          << "Compressed-tensors INT8 does not support smooth tensors: "
          << state_dict.prefix() << prefix << "smooth";
    }
  }
  auto resolved =
      quant_args.get_quant_method_from_prefixes(state_dict, prefixes);
  if (resolved.has_value()) {
    resolved_weight_quant_method = resolved.value();
    return;
  }
  if (quant_args.is_compressed_tensors_w8a8_dynamic()) {
    // Resolve from the scheme, even when weight and scale arrive in separate
    // safetensors shards. Only ignored modules may load floating weights.
    std::optional<bool> ignored;
    for (const std::string& prefix : prefixes) {
      std::string module = std::string(state_dict.prefix()) + prefix;
      if (!module.empty() && module.back() == '.') {
        module.pop_back();
      }
      const bool skip = quant_args.should_ignore_module(module);
      CHECK(!ignored.has_value() || ignored.value() == skip)
          << "Cannot fuse quantized and ignored projections: " << module;
      ignored = skip;
      check_ct_scale(state_dict.get_tensor(prefix + "weight_scale"), module);
      torch::Tensor weight = state_dict.get_tensor(prefix + "weight");
      CHECK(!weight.defined() || (skip ? weight.is_floating_point()
                                       : weight.scalar_type() == torch::kInt8))
          << "Weight dtype disagrees with compressed-tensors scheme: "
          << module;
    }
    resolved_weight_quant_method =
        ignored.value() ? std::nullopt
                        : std::make_optional<std::string>("w8a8_dynamic");
    return;
  }

  if (!quant_args.quant_descs().empty()) {
    LOG(WARNING) << "[LinearLoad][QuantMethod] quant_descs is not empty but "
                    "quant method was not resolved from state_dict prefixes. "
                    "state_dict.prefix="
                 << state_dict.prefix();
  }
  resolved_weight_quant_method = std::nullopt;
}

inline torch::Tensor npu_w8a8_dynamic_quantized_linear_forward(
    const torch::Tensor& quantized_input,
    const torch::Tensor& per_token_scale,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale,
    const std::optional<torch::Tensor>& bias,
    at::ScalarType output_dtype,
    const std::optional<torch::Tensor>& weight_offset = std::nullopt) {
  CHECK_EQ(quantized_input.scalar_type(), torch::kInt8)
      << "w8a8_dynamic quantized input must be int8.";
  CHECK(per_token_scale.defined())
      << "w8a8_dynamic per-token scale must be defined.";
  CHECK_EQ(per_token_scale.scalar_type(), torch::kFloat32)
      << "w8a8_dynamic per-token scale must be float32.";

  kernel::QuantMatmulParams quant_matmul_params;
  quant_matmul_params.x1 = quantized_input;
  quant_matmul_params.x2 = weight;
  quant_matmul_params.transpose2 = true;
  quant_matmul_params.scale = weight_scale;
  quant_matmul_params.pertoken_scale = per_token_scale.reshape({-1});
  quant_matmul_params.output_dtype = output_dtype;
  if (weight_offset.has_value() && weight_offset->defined()) {
    quant_matmul_params.offset = weight_offset;
  }
  if (bias.has_value() && bias->defined()) {
    quant_matmul_params.bias = bias;
  }
  return kernel::quant_matmul(quant_matmul_params);
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

  torch::Tensor quantized_input;
  std::optional<torch::Tensor> per_token_scale;
  std::tie(quantized_input, per_token_scale) =
      kernel::dynamic_quant(quant_params);
  CHECK(per_token_scale.has_value() && per_token_scale->defined())
      << "dynamic_quant must return per-token scale for w8a8_dynamic.";

  return npu_w8a8_dynamic_quantized_linear_forward(quantized_input,
                                                   per_token_scale.value(),
                                                   weight,
                                                   weight_scale,
                                                   bias,
                                                   output_dtype,
                                                   weight_offset);
}

}  // namespace layer
}  // namespace xllm

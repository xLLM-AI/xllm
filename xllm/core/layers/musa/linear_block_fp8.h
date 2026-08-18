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

#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace xllm {

struct QuantArgs;
class StateDict;

}  // namespace xllm

namespace xllm::layer::musa {

bool is_block_fp8_quant(const QuantArgs& quant_args);

void check_quantization_supported(
    const QuantArgs& quant_args,
    const std::optional<std::string>& resolved_weight_quant_method =
        std::nullopt);

void check_replicated_weight_supported(const StateDict& state_dict);

void maybe_resolve_block_fp8_unquantized(
    const StateDict& state_dict,
    const std::vector<std::string>* prefixes,
    const torch::TensorOptions& options,
    torch::Tensor& weight,
    bool& weight_is_loaded,
    bool weight_scale_inv_is_loaded,
    bool& block_fp8_resolved_unquantized);

void register_block_fp8_parameters(torch::nn::Module& module,
                                   int64_t out_features,
                                   int64_t in_features,
                                   const torch::TensorOptions& options,
                                   torch::Tensor& weight,
                                   torch::Tensor& weight_scale_inv);

torch::Tensor matmul_forward(const torch::Tensor& input,
                             const torch::Tensor& weight,
                             const std::optional<torch::Tensor>& bias,
                             torch::Tensor& output_buffer);

torch::Tensor block_fp8_or_bf16_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale_inv,
    const std::optional<torch::Tensor>& bias,
    torch::Tensor& output_buffer);

}  // namespace xllm::layer::musa

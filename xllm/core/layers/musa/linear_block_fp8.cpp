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

#include "layers/musa/linear_block_fp8.h"

#include <glog/logging.h>

#include <algorithm>
#include <tuple>

#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "kernels/musa/musa_ops_api.h"

namespace xllm::layer::musa {

namespace {

constexpr int64_t kBlockFp8Size = 128;
constexpr int64_t kMatmulOutputBufMaxRows = 256;

void check_block_fp8_weight_dtype(const torch::Tensor& weight) {
  if (!weight.defined() || !c10::isFloat8Type(weight.scalar_type())) {
    return;
  }
  CHECK_EQ(weight.scalar_type(), torch::kFloat8_e4m3fn)
      << "MUSA block-FP8 supports only float8_e4m3fn weights; got "
      << weight.scalar_type();
}

std::optional<torch::Tensor> get_matmul_output_buffer(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    torch::Tensor& output_buffer) {
  if (input.dim() != 2 || weight.dim() != 2 || input.size(0) <= 0 ||
      input.size(0) > kMatmulOutputBufMaxRows) {
    return std::nullopt;
  }
  const int64_t rows = input.size(0);
  const int64_t columns = weight.size(0);
  const bool needs_realloc =
      !output_buffer.defined() || output_buffer.size(0) < rows ||
      output_buffer.size(1) != columns ||
      output_buffer.scalar_type() != input.scalar_type() ||
      output_buffer.device() != input.device();
  if (needs_realloc) {
    output_buffer =
        torch::empty({kMatmulOutputBufMaxRows, columns}, input.options());
  }
  return output_buffer.narrow(/*dim=*/0, /*start=*/0, /*length=*/rows);
}

torch::Tensor block_fp8_linear_forward(const torch::Tensor& input,
                                       const torch::Tensor& weight,
                                       const torch::Tensor& weight_scale_inv,
                                       const std::optional<torch::Tensor>& bias,
                                       torch::Tensor& output_buffer) {
  CHECK_EQ(weight.scalar_type(), torch::kFloat8_e4m3fn);
  CHECK_EQ(input.scalar_type(), torch::kBFloat16);
  std::vector<int64_t> input_shape = input.sizes().vec();
  const int64_t k = input.size(-1);
  CHECK_EQ(k % kBlockFp8Size, 0) << "native block-fp8 GEMM requires K % "
                                 << kBlockFp8Size << " == 0, got K=" << k;
  torch::Tensor input_2d = input.reshape({-1, k}).contiguous();
  torch::Tensor quantized_input;
  torch::Tensor input_scale;
  std::tie(quantized_input, input_scale) =
      xllm::kernel::musa::per_token_group_quant_fp8(
          input_2d, /*group_size=*/kBlockFp8Size);
  CHECK_EQ(weight_scale_inv.scalar_type(), torch::kFloat32);
  CHECK(weight_scale_inv.is_contiguous());

  torch::Tensor output = xllm::kernel::musa::gemm_fp8_nt_groupwise(
      quantized_input,
      weight,
      input_scale,
      weight_scale_inv,
      /*output_dtype=*/torch::kBFloat16,
      /*output=*/get_matmul_output_buffer(input_2d, weight, output_buffer));
  if (bias.has_value() && bias->defined()) {
    output.add_(bias->to(output.scalar_type()));
  }
  input_shape.back() = weight.size(0);
  return output.reshape(input_shape);
}

}  // namespace

torch::Tensor matmul_forward(const torch::Tensor& input,
                             const torch::Tensor& weight,
                             const std::optional<torch::Tensor>& bias,
                             torch::Tensor& output_buffer) {
  return xllm::kernel::musa::matmul(
      input,
      weight,
      bias,
      /*output_buf=*/get_matmul_output_buffer(input, weight, output_buffer));
}

bool is_block_fp8_quant(const QuantArgs& quant_args) {
  const std::vector<int64_t>& weight_block_size =
      quant_args.weight_block_size();
  return quant_args.quant_method() == kQuantMethodFp8 &&
         weight_block_size.size() == 2 &&
         weight_block_size[0] == kBlockFp8Size &&
         weight_block_size[1] == kBlockFp8Size;
}

void check_quantization_supported(
    const QuantArgs& quant_args,
    const std::optional<std::string>& resolved_weight_quant_method) {
  CHECK_NE(quant_args.quant_method(), kQuantMethodSmoothquant)
      << "MUSA linear does not support SmoothQuant.";
  if (quant_args.quant_method() == kQuantMethodFp8) {
    CHECK(is_block_fp8_quant(quant_args))
        << "MUSA linear supports only FP8 weight_block_size=[128, 128].";
  }
  if (!resolved_weight_quant_method.has_value()) {
    return;
  }
  CHECK(resolved_weight_quant_method.value() != "w8a8" &&
        resolved_weight_quant_method.value() != "w8a8_dynamic")
      << "MUSA linear does not support W8A8 checkpoints.";
}

void check_replicated_weight_supported(const StateDict& state_dict) {
  const torch::Tensor checkpoint_weight = state_dict.get_tensor("weight");
  CHECK(!checkpoint_weight.defined() ||
        !c10::isFloat8Type(checkpoint_weight.scalar_type()))
      << "MUSA replicated linear does not support FP8 checkpoint weights.";
}

void maybe_resolve_block_fp8_unquantized(
    const StateDict& state_dict,
    const std::vector<std::string>* prefixes,
    const torch::TensorOptions& options,
    torch::Tensor& weight,
    bool& weight_is_loaded,
    bool weight_scale_inv_is_loaded,
    bool& block_fp8_resolved_unquantized) {
  if (prefixes == nullptr) {
    check_block_fp8_weight_dtype(state_dict.get_tensor("weight"));
  } else {
    for (const std::string& prefix : *prefixes) {
      check_block_fp8_weight_dtype(state_dict.get_tensor(prefix + "weight"));
    }
  }

  if (block_fp8_resolved_unquantized || weight_scale_inv_is_loaded) {
    return;
  }

  const bool missing_scale =
      prefixes == nullptr
          ? state_dict.has("weight") && !state_dict.has("weight_scale_inv")
          : std::any_of(prefixes->begin(),
                        prefixes->end(),
                        [&state_dict](const std::string& prefix) {
                          return state_dict.has(prefix + "weight") &&
                                 !state_dict.has(prefix + "weight_scale_inv");
                        });
  if (!missing_scale) {
    return;
  }

  block_fp8_resolved_unquantized = true;
  weight.set_data(torch::empty(weight.sizes(), options));
  weight_is_loaded = false;
}

void register_block_fp8_parameters(torch::nn::Module& module,
                                   int64_t out_features,
                                   int64_t in_features,
                                   const torch::TensorOptions& options,
                                   torch::Tensor& weight,
                                   torch::Tensor& weight_scale_inv) {
  weight = module.register_parameter(
      "weight",
      torch::empty({out_features, in_features},
                   options.dtype(torch::kFloat8_e4m3fn)),
      /*requires_grad=*/false);
  const int64_t n_tiles = (out_features + kBlockFp8Size - 1) / kBlockFp8Size;
  const int64_t k_tiles = (in_features + kBlockFp8Size - 1) / kBlockFp8Size;
  weight_scale_inv = module.register_parameter(
      "weight_scale_inv",
      torch::empty({n_tiles, k_tiles}, options.dtype(torch::kFloat32)),
      /*requires_grad=*/false);
}

torch::Tensor block_fp8_or_bf16_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale_inv,
    const std::optional<torch::Tensor>& bias,
    torch::Tensor& output_buffer) {
  if (weight.scalar_type() == torch::kFloat8_e4m3fn) {
    return block_fp8_linear_forward(
        input, weight, weight_scale_inv, bias, output_buffer);
  }
  if (c10::isFloat8Type(weight.scalar_type())) {
    LOG(FATAL) << "MUSA block-FP8 supports only float8_e4m3fn weights; got "
               << weight.scalar_type();
  }
  return matmul_forward(input, weight, bias, output_buffer);
}

}  // namespace xllm::layer::musa

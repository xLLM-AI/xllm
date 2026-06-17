/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "rms_norm.h"

#include <glog/logging.h>

#include "kernels/ops_api.h"
#include "platform/device.h"

namespace xllm {
namespace layer {

const static std::string kLayerNormMode = "layernorm";
const static std::string kRmsNormMode = "rmsnorm";

RMSNormImpl::RMSNormImpl(int64_t dim,
                         double eps,
                         const torch::TensorOptions& options)
    : norm_dim_(dim), eps_(eps), mode_(kRmsNormMode) {
  weight_ = register_parameter("weight",
                               torch::empty({dim}, options),
                               /*requires_grad=*/false);
}

RMSNormImpl::RMSNormImpl(const ModelContext& context)
    : RMSNormImpl(context.get_model_args().hidden_size(),
                  context.get_model_args().rms_norm_eps(),
                  context.get_tensor_options()) {}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> RMSNormImpl::forward(
    torch::Tensor& input,
    std::optional<torch::Tensor> residual,
    std::optional<torch::Tensor> inplace_output) {
  auto org_shape = input.sizes().vec();
  input = input.reshape({-1, norm_dim_});

  torch::Tensor output;
  if (Device::type_str() != "npu") {
    if (inplace_output.has_value()) {
      output = inplace_output.value();
      output = output.reshape({-1, norm_dim_});
    } else {
      output = torch::empty_like(input);
    }
  }

  std::optional<torch::Tensor> residual_out;
  if (residual.has_value()) {
    residual.value() = residual.value().reshape({-1, norm_dim_});
    if (Device::type_str() == "mlu" || Device::type_str() == "ilu") {
      residual_out = residual.value();
    }
  }

  xllm::kernel::FusedLayerNormParams fused_layernorm_params;
  fused_layernorm_params.input = input;
  fused_layernorm_params.residual = residual;
  fused_layernorm_params.output = output;
  fused_layernorm_params.residual_out = residual_out;
  fused_layernorm_params.weight = weight_;
  fused_layernorm_params.eps = eps_;
  fused_layernorm_params.mode = mode_;
  fused_layernorm_params.store_output_before_norm = residual_out.has_value();
  if (bias_.defined()) {
    fused_layernorm_params.beta = bias_;
  }

  xllm::kernel::fused_layernorm(fused_layernorm_params);

  output = fused_layernorm_params.output;
  residual_out = fused_layernorm_params.residual_out;

  output = output.view(org_shape);
  if (residual_out.has_value()) {
    residual_out.value() = residual_out.value().view(org_shape);
  }
  return std::make_tuple(output, residual_out);
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>>
RMSNormImpl::forward_fp8(torch::Tensor& input,
                         const torch::Tensor& fp8_scale,
                         std::optional<torch::Tensor> residual) {
  // Only supported on CUDA for now
  CHECK(Device::type_str() == "cuda")
      << "forward_fp8 is only supported on CUDA";
  CHECK(mode_ == kRmsNormMode)
      << "forward_fp8 only supports RMSNorm mode, not LayerNorm";

  if (residual.has_value()) {
    // Fused Add + RMSNorm + FP8 Quantization
    xllm::kernel::FusedAddRmsNormStaticFp8QuantParams params;
    params.input = input;
    params.residual = residual.value();
    params.weight = weight_;
    params.scale = fp8_scale;
    params.epsilon = eps_;

    auto [output, updated_residual] =
        xllm::kernel::fused_add_rms_norm_static_fp8_quant(params);

    return std::make_tuple(output, updated_residual);
  } else {
    // RMSNorm + FP8 Quantization (no residual)
    xllm::kernel::RmsNormStaticFp8QuantParams params;
    params.input = input;
    params.weight = weight_;
    params.scale = fp8_scale;
    params.epsilon = eps_;

    auto output = xllm::kernel::rms_norm_static_fp8_quant(params);

    return std::make_tuple(output, std::nullopt);
  }
}

void RMSNormImpl::load_state_dict(const StateDict& state_dict) {
  LOAD_WEIGHT(weight);

#if defined(USE_NPU)
  // Bias loading policy for anti-outlier:
  // rmsnorm mode: bias is optional and only enabled when checkpoint
  //    contains "bias" (anti-outlier path).
  if (state_dict.has("bias")) {
    if (!bias_.defined()) {
      bias_ = register_parameter(
          "bias", torch::empty({norm_dim_}, weight_.options()), false);
    }
    LOAD_WEIGHT(bias);
  }
#else
  if (bias_.defined()) {
    LOAD_WEIGHT(bias);
  }
#endif
}

void RMSNormImpl::set_layernorm_mode() {
  mode_ = kLayerNormMode;
  bias_ = register_parameter(
      "bias", torch::empty({norm_dim_}, weight_.options()), false);
}

}  // namespace layer
}  // namespace xllm

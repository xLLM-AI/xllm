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

#include "add_matmul.h"

#include "core/framework/state_dict/utils.h"
#include "core/layers/common/quant_utils.h"
#include "kernels/ops_api.h"

namespace xllm {
namespace layer {

namespace F = torch::nn::functional;

AddMatmulImpl::AddMatmulImpl(int64_t in,
                             int64_t out,
                             bool with_bias,
                             const torch::TensorOptions& options)
    : with_bias_(with_bias), options_(options) {
  weight_ =
      register_parameter("weight", torch::empty({out, in}, options_), false);
  if (with_bias) {
    bias_ = register_parameter("bias", torch::empty(out, options_), false);
  }
}

torch::Tensor AddMatmulImpl::forward(const torch::Tensor& x) {
  if (with_bias_) {
    auto sizes = x.sizes();
    if (sizes.size() == 3) {
      torch::Tensor t = x.reshape({sizes[0] * sizes[1], sizes[2]});
      return torch::addmm(bias_, t, weight_.t())
          .reshape({sizes[0], sizes[1], weight_.size(0)});
    } else {
      return torch::addmm(bias_, x, weight_.t());
    }
  } else {
    return torch::matmul(x, weight_.t());
  }
}

void AddMatmulImpl::load_state_dict(const xllm::StateDict& state_dict) {
  xllm::weight::load_weight(state_dict, "weight", weight_, weight_is_loaded_);
  if (with_bias_) {
    xllm::weight::load_weight(state_dict, "bias", bias_, bias_is_loaded_);
  }
}

void AddMatmulImpl::verify_loaded_weights(const std::string& prefix) const {
  CHECK(weight_is_loaded_) << "weight is not loaded for " << prefix + "weight";
  if (with_bias_) {
    CHECK(bias_is_loaded_) << "bias is not loaded for " << prefix + "bias";
  }
}

FusedAddMatmulImpl::FusedAddMatmulImpl(int64_t in,
                                       int64_t out,
                                       bool with_bias,
                                       const torch::TensorOptions& options)
    : AddMatmulImpl(in, out, with_bias, options) {}

void FusedAddMatmulImpl::load_state_dict(
    const xllm::StateDict& state_dict,
    const std::vector<std::string>& names) {
  std::vector<torch::Tensor> weights;
  std::vector<torch::Tensor> biases;

  for (const auto& name : names) {
    auto weight = state_dict.get_tensor(name + ".weight");
    if (weight.defined()) weights.push_back(weight);

    if (with_bias_) {
      auto bias = state_dict.get_tensor(name + ".bias");
      if (bias.defined()) biases.push_back(bias);
    }
  }

  if (weights.size() > 0) {
    auto fused_weight = torch::cat(weights, 0);
    weight_.data().copy_(fused_weight);
    weight_is_loaded_ = true;
  }

  if (with_bias_ && biases.size() > 0) {
    auto fused_bias = torch::cat(biases, 0);
    bias_.data().copy_(fused_bias);
    bias_is_loaded_ = true;
  }
}

AddMatmulWeightTransposedImpl::AddMatmulWeightTransposedImpl(
    int64_t in,
    int64_t out,
    bool with_bias,
    const torch::TensorOptions& options,
    const QuantArgs& quant_args)
    : AddMatmulImpl(in, out, with_bias, options),
      quant_args_(quant_args),
      output_dtype_(c10::typeMetaToScalarType(options.dtype())) {
  if (!quant_args_.quant_descs().empty()) {
    weight_scale_ =
        register_parameter("weight_scale",
                           torch::empty({out}, options.dtype(torch::kFloat32)),
                           /*requires_grad=*/false);
    weight_offset_ =
        register_parameter("weight_offset",
                           torch::empty({out}, options.dtype(torch::kFloat32)),
                           /*requires_grad=*/false);
  }
}

torch::Tensor AddMatmulWeightTransposedImpl::forward(const torch::Tensor& x) {
  if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
    CHECK(weight_scale_is_loaded_ && weight_scale_.defined())
        << "weight_scale is required for w8a8_dynamic quant matmul.";
    auto bias = with_bias_ ? std::optional<torch::Tensor>(bias_) : std::nullopt;
    // Offset only valid when out dtype is INT8 (NPU kernel requirement).
    auto w_offset = (weight_offset_.defined() && output_dtype_ == torch::kInt8)
                        ? std::optional<torch::Tensor>(weight_offset_)
                        : std::nullopt;
    return npu_w8a8_dynamic_linear_forward(
        x, weight_, weight_scale_, bias, output_dtype_, w_offset);
  }
  // use addmm when bias is provided
  if (with_bias_) {
    auto sizes = x.sizes();
    if (sizes.size() == 3) {
      torch::Tensor t = x.reshape({sizes[0] * sizes[1], sizes[2]});
      return torch::addmm(bias_, t, weight_)
          .reshape({sizes[0], sizes[1], weight_.size(1)});
    } else {
      return torch::addmm(bias_, x, weight_);
    }
  } else {
    return torch::matmul(x, weight_);
  }
}

void AddMatmulWeightTransposedImpl::load_state_dict(
    const StateDict& state_dict) {
  resolve_weight_quant_method_for_linear_load(
      quant_args_, state_dict, nullptr, resolved_weight_quant_method_);

  if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
    std::vector<weight::LazyParameterSpec> specs;
    specs.push_back(
        weight::LazyParameterSpec{&weight_,
                                  &weight_is_loaded_,
                                  "weight",
                                  {weight_.size(0), weight_.size(1)},
                                  options_.dtype(torch::kInt8)});
    weight::ensure_parameter_storage(this, specs);

    std::vector<weight::LazyParameterSpec> scale_specs;
    scale_specs.push_back(
        weight::LazyParameterSpec{&weight_scale_,
                                  &weight_scale_is_loaded_,
                                  "weight_scale",
                                  {weight_.size(0)},
                                  options_.dtype(torch::kFloat32)});
    scale_specs.push_back(
        weight::LazyParameterSpec{&weight_offset_,
                                  &weight_offset_is_loaded_,
                                  "weight_offset",
                                  {weight_.size(0)},
                                  options_.dtype(torch::kFloat32)});
    weight::ensure_parameter_storage(this, scale_specs);

    if (state_dict.has("weight")) {
      weight::load_weight(state_dict, "weight", weight_, weight_is_loaded_);
    }
    if (state_dict.has("weight_scale")) {
      weight::load_weight(
          state_dict, "weight_scale", weight_scale_, weight_scale_is_loaded_);
    }
    if (state_dict.has("weight_offset")) {
      weight::load_weight(state_dict,
                          "weight_offset",
                          weight_offset_,
                          weight_offset_is_loaded_);
    }
  } else {
    if (state_dict.has("weight")) {
      weight::load_weight(state_dict, "weight", weight_, weight_is_loaded_);

      torch::Tensor transposed = weight_.data().transpose(0, 1).contiguous();
      weight_.set_data(transposed);
    }
  }
  if (with_bias_) {
    weight::load_weight(state_dict, "bias", bias_, bias_is_loaded_);
  }
}

}  // namespace layer
}  // namespace xllm

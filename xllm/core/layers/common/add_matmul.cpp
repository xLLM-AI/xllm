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
    const torch::TensorOptions& options)
    : AddMatmulImpl(in, out, with_bias, options) {}

torch::Tensor AddMatmulWeightTransposedImpl::forward(const torch::Tensor& x) {
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
  // only transpoes weights when state_dict has the key
  // or it would be transposed multiple times when having
  // multiple state dicts
  if (state_dict.has("weight")) {
    xllm::weight::load_weight(state_dict, "weight", weight_, weight_is_loaded_);
    // weight need to be transposed when using addmm

    torch::Tensor transposed = weight_.data().transpose(0, 1).contiguous();
    weight_.set_data(transposed);
  }
  if (with_bias_) {
    weight::load_weight(state_dict, "bias", bias_, bias_is_loaded_);
  }
}

}  // namespace layer
}  // namespace xllm

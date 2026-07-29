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

#pragma once

#include <torch/torch.h>

#include <optional>
#include <string>

#include "core/framework/quant_args.h"
#include "core/framework/state_dict/state_dict.h"

namespace xllm {
namespace layer {

class AddMatmulImpl : public torch::nn::Module {
 public:
  AddMatmulImpl(int64_t in,
                int64_t out,
                bool with_bias,
                const torch::TensorOptions& options);

  virtual torch::Tensor forward(const torch::Tensor& x);

  virtual void load_state_dict(const xllm::StateDict& state_dict);

  void verify_loaded_weights(const std::string& prefix) const;

 protected:
  torch::Tensor weight_;
  torch::Tensor bias_;

  bool with_bias_;
  torch::TensorOptions options_;

  bool weight_is_loaded_ = false;
  bool bias_is_loaded_ = false;
};
TORCH_MODULE(AddMatmul);

class FusedAddMatmulImpl : public AddMatmulImpl {
 public:
  FusedAddMatmulImpl(int64_t in,
                     int64_t out,
                     bool with_bias,
                     const torch::TensorOptions& options);

  void load_state_dict(const xllm::StateDict& state_dict,
                       const std::vector<std::string>& names);
};
TORCH_MODULE(FusedAddMatmul);

class AddMatmulWeightTransposedImpl : public AddMatmulImpl {
 public:
  AddMatmulWeightTransposedImpl(int64_t in,
                                int64_t out,
                                bool with_bias,
                                const torch::TensorOptions& options,
                                const QuantArgs& quant_args = QuantArgs());

  torch::Tensor forward(const torch::Tensor& x) override;

  void load_state_dict(const xllm::StateDict& state_dict) override;

 protected:
  QuantArgs quant_args_;
  std::optional<std::string> resolved_weight_quant_method_;
  torch::Tensor weight_scale_;
  torch::Tensor weight_offset_;
  bool weight_scale_is_loaded_ = false;
  bool weight_offset_is_loaded_ = false;
  at::ScalarType output_dtype_;
};

TORCH_MODULE(AddMatmulWeightTransposed);

}  // namespace layer
}  // namespace xllm

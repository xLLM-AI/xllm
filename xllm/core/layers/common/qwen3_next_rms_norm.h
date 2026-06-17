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

#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"

namespace xllm {
namespace layer {

class Qwen3NextRMSNormImpl : public torch::nn::Module {
 public:
  Qwen3NextRMSNormImpl(int64_t dim,
                       double eps,
                       const torch::TensorOptions& options);

  std::tuple<torch::Tensor, std::optional<torch::Tensor>> forward(
      torch::Tensor& input,
      std::optional<torch::Tensor> residual = std::nullopt);
  torch::Tensor weight() const { return weight_; }
  bool is_weight_loaded() const { return weight_is_loaded_; }
  double eps() const { return eps_; }

  void load_state_dict(const StateDict& state_dict);

 private:
  DEFINE_WEIGHT(weight);
  int64_t norm_dim_;
  double eps_;
};
TORCH_MODULE(Qwen3NextRMSNorm);

}  // namespace layer
}  // namespace xllm

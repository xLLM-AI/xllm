/* Copyright 2026 The xLLM Authors.

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

#include <glog/logging.h>
#include <torch/torch.h>

#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"

namespace xllm {
namespace layer {

// Fused adaptive LayerNorm (AdaLayerNorm).
// Computes: LayerNorm(x) * (1 + scale) + shift in a single fused kernel.
// The caller passes the raw scale/shift (the +1 is applied by the kernel).
class AdaLayerNormImpl : public torch::nn::Module {
 public:
  AdaLayerNormImpl(int64_t dim,
                   double eps,
                   bool elementwise_affine,
                   const torch::TensorOptions& options);

  // input:  [B, S, H]
  // scale:  [B, H], [B, 1, H], or [B, S, H] (raw scale, +1 applied internally)
  // shift:  same shape constraints as scale
  torch::Tensor forward(const torch::Tensor& input,
                        const torch::Tensor& scale,
                        const torch::Tensor& shift);

  void load_state_dict(const StateDict& state_dict);

  void verify_loaded_weights(const std::string& prefix) const {
    if (elementwise_affine_) {
      CHECK(weight_is_loaded_)
          << "weight is not loaded for " << prefix + "weight";
      CHECK(bias_is_loaded_) << "bias is not loaded for " << prefix + "bias";
    }
  }

  torch::Tensor weight() const { return weight_; }
  torch::Tensor bias() const { return bias_; }
  double eps() const { return eps_; }

 private:
  DEFINE_WEIGHT(weight);
  DEFINE_WEIGHT(bias);
  int64_t norm_dim_;
  double eps_;
  bool elementwise_affine_;
};
TORCH_MODULE(AdaLayerNorm);

}  // namespace layer
}  // namespace xllm

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

#include <torch/torch.h>

#include <array>
#include <vector>

namespace xllm {
class QuantArgs;
}

namespace xllm::layer {

enum class GdnProjectionMode { SEPARATE, GROUPED };

// Select the supported path before loading weights or executing device work.
GdnProjectionMode resolve_gdn_projection_mode(const QuantArgs& quant_args,
                                              torch::ScalarType dtype);

// Stable field order for both weights and outputs: QKV, Z, B, A.
using GdnProjectionTensors = std::array<torch::Tensor, 4>;

// Retains already TP-sharded weights. Grouped projections use transpose views
// of the original allocations, preserving Decode weights and Q/K/V ordering.
class GdnPrefillProjection final {
 public:
  GdnPrefillProjection(GdnProjectionTensors weights, GdnProjectionMode mode);

  GdnProjectionTensors forward(const torch::Tensor& input) const;

  const GdnProjectionTensors& weights() const { return weights_; }

 private:
  GdnProjectionTensors weights_;
  const GdnProjectionMode mode_;
  std::vector<torch::Tensor> transposed_weights_;
};

}  // namespace xllm::layer

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

#include "core/layers/npu_torch/qwen3_5_gdn_projection.h"

#include <glog/logging.h>
#include <torch/nn/functional/linear.h>
#include <torch_npu/csrc/aten/CustomFunctions.h>

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "core/framework/quantization/quant_args.h"

namespace xllm::layer {

GdnProjectionMode resolve_gdn_projection_mode(const QuantArgs& quant_args,
                                              torch::ScalarType dtype) {
  const bool supports_grouped =
      dtype == torch::kBFloat16 && quant_args.quant_method().empty() &&
      quant_args.quantize_type().empty() && quant_args.quant_descs().empty() &&
      quant_args.compressed_groups().empty() &&
      !quant_args.is_compressed_tensors_w8a8_dynamic();
  return supports_grouped ? GdnProjectionMode::GROUPED
                          : GdnProjectionMode::SEPARATE;
}

GdnPrefillProjection::GdnPrefillProjection(GdnProjectionTensors weights,
                                           GdnProjectionMode mode)
    : weights_(std::move(weights)), mode_(mode) {
  CHECK(mode_ == GdnProjectionMode::SEPARATE ||
        mode_ == GdnProjectionMode::GROUPED);
  for (size_t index = 0; index < weights_.size(); ++index) {
    const auto& weight = weights_[index];
    CHECK(weight.defined())
        << "GDN projection weight " << index << " is missing";
    CHECK_EQ(weight.dim(), 2);
    CHECK_EQ(weight.scalar_type(), torch::kBFloat16);
    CHECK_EQ(weight.device(), weights_[0].device());
    CHECK_EQ(weight.size(1), weights_[0].size(1));
    CHECK_GT(weight.size(0), 0);
    CHECK_GT(weight.size(1), 0);
  }
  CHECK_EQ(weights_[2].size(0), weights_[3].size(0))
      << "GDN B/A head counts must match";
  if (mode_ == GdnProjectionMode::GROUPED) {
    CHECK(weights_[0].device().is_privateuseone())
        << "Grouped GDN projection requires NPU";
    transposed_weights_.reserve(weights_.size());
    for (const auto& weight : weights_) {
      transposed_weights_.emplace_back(weight.t());
    }
  }
}

GdnProjectionTensors GdnPrefillProjection::forward(
    const torch::Tensor& input) const {
  CHECK_EQ(input.dim(), 2);
  CHECK_EQ(input.size(1), weights_[0].size(1));
  CHECK_EQ(input.scalar_type(), weights_[0].scalar_type());
  CHECK_EQ(input.device(), weights_[0].device());
  if (mode_ == GdnProjectionMode::SEPARATE) {
    return {torch::nn::functional::linear(input, weights_[0]),
            torch::nn::functional::linear(input, weights_[1]),
            torch::nn::functional::linear(input, weights_[2]),
            torch::nn::functional::linear(input, weights_[3])};
  }
  const std::vector<torch::Tensor> inputs(weights_.size(), input);
  // Independent matmuls with separate contiguous outputs. This is not MoE's
  // group_list-based M-axis partitioning.
  auto outputs = at_npu::native::custom_ops::npu_grouped_matmul(
      inputs,
      transposed_weights_,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::optional<torch::Tensor>{},
      std::nullopt,
      std::nullopt,
      std::nullopt,
      /*split_item=*/0,
      /*group_type=*/-1);
  CHECK_EQ(outputs.size(), weights_.size());
  return {outputs[0], outputs[1], outputs[2], outputs[3]};
}

}  // namespace xllm::layer

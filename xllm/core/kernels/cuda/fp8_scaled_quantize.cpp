/* Copyright 2025-2026 The xLLM Authors.

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

#include "cuda_ops_api.h"

namespace xllm::kernel::cuda {

std::tuple<torch::Tensor, torch::Tensor> fp8_scaled_quantize(
    const torch::Tensor& input,
    const std::optional<torch::Tensor>& output /* = std::nullopt */,
    const std::optional<torch::Tensor>& scale /* = std::nullopt */) {
  // Prepare output tensor
  torch::Tensor result_output;
  if (output.has_value() && output.value().defined()) {
    result_output = output.value();
  } else {
    result_output =
        torch::empty_like(input, input.options().dtype(torch::kFloat8_e4m3fn));
  }

  torch::Tensor result_scale;
  if (scale.has_value() && scale.value().defined()) {
    // Static quantization - use pre-computed scale
    result_scale = scale.value();
  } else {
    // Dynamic quantization - compute scale on the fly
    // 448 is max value for FP8 e4m3
    auto amax = input.abs().max();
    result_scale = (amax / 448.0f).clamp_min(1e-12f).to(torch::kFloat32);
  }

  // Call underlying kernel
  static_scaled_fp8_quant(result_output, input, result_scale);

  return std::make_tuple(result_output, result_scale);
}

}  // namespace xllm::kernel::cuda

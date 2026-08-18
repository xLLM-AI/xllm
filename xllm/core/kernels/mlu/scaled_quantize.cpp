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

#include "mlu_ops_api.h"

namespace xllm::kernel::mlu {
std::tuple<torch::Tensor, torch::Tensor> scaled_quantize(
    const torch::Tensor& x,
    const torch::Tensor& smooth,
    const std::optional<torch::Tensor>& zero /* = std::nullopt */,
    const std::optional<torch::Tensor>& token_count /* = std::nullopt */,
    const std::optional<torch::Tensor>& gather_index /* = std::nullopt */,
    const std::optional<torch::Tensor>&
        gather_index_start_position /* = std::nullopt */,
    const std::optional<torch::Tensor>& output /* = std::nullopt */,
    const std::optional<torch::Tensor>& output_scale /* = std::nullopt */,
    const std::string& act_mode /* = "none" */,
    double active_coef /* = 1.0 */,
    bool is_gated /* = false */,
    at::ScalarType quant_type /* = at::kChar */
) {
  torch::Tensor quant_input = x;
  std::string quant_act_mode = act_mode;
  bool quant_is_gated = is_gated;
  if (act_mode == "gelu_pytorch_tanh") {
    std::vector<int64_t> active_shape(x.sizes().begin(), x.sizes().end());
    if (is_gated) {
      active_shape.back() /= 2;
    }
    quant_input = torch::empty(active_shape, x.options());
    active(x,
           quant_input,
           std::nullopt,
           std::nullopt,
           act_mode,
           is_gated,
           /*start_expert_id=*/0,
           /*expert_size=*/0);
    quant_act_mode = "none";
    quant_is_gated = false;
  }

  if (quant_act_mode == "none") {
    quant_is_gated = false;
  }

  auto input_sizes = quant_input.sizes();
  std::vector<int64_t> output_shape(input_sizes.begin(), input_sizes.end());
  std::vector<int64_t> output_scale_shape(input_sizes.begin(),
                                          input_sizes.end() - 1);

  if (gather_index.has_value()) {
    int64_t output_tokens = gather_index.value().size(0);
    output_shape[0] = output_tokens;
    output_scale_shape[0] = output_tokens;
  }

  if (quant_is_gated) {
    output_shape.back() = output_shape.back() / 2;
  }

  torch::Tensor result_output;
  torch::Tensor result_output_scale;

  if (output.has_value()) {
    result_output = output.value();
  } else {
    result_output = torch::empty(output_shape, x.options().dtype(quant_type));
  }

  if (output_scale.has_value()) {
    result_output_scale = output_scale.value();
  } else {
    result_output_scale =
        torch::empty(output_scale_shape, x.options().dtype(at::kFloat));
  }

  tmo::torch_api::scaled_quantize(
      quant_input,
      result_output,
      result_output_scale,
      smooth,
      zero,
      token_count,
      gather_index,
      gather_index_start_position,
      /*scale_upper_bound*/ std::nullopt,
      /*quant_algo=*/std::string("dynamic_per_token"),
      quant_act_mode,
      active_coef,
      quant_is_gated);

  return std::make_tuple(result_output, result_output_scale);
}

}  // namespace xllm::kernel::mlu

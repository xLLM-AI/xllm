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

#include <glog/logging.h>

#include "mlu_ops_api.h"

namespace xllm::kernel::mlu {

torch::Tensor scaled_matmul(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const std::optional<torch::Tensor>& a_scale,
    const torch::Tensor& b_scale,
    torch::ScalarType output_dtype,
    const std::optional<torch::Tensor>& bias /* = std::nullopt */,
    const std::optional<torch::Tensor>& c /* = std::nullopt */,
    const std::string& act_mode /* = "none" */,
    int64_t quant_bit_size /* = 8 */,
    double alpha /* = 1.0 */,
    double beta /* = 1.0 */,
    bool use_hp_active /* = false */,
    int64_t a_quant_bit_size /* = -1 */,
    const std::optional<torch::Tensor>& a_calib /* = std::nullopt */,
    const std::optional<torch::Tensor>& b_calib /* = std::nullopt */,
    const std::optional<torch::Tensor>& output /* = std::nullopt */
) {
  // Check: only support w8a8 quantization for now.
  CHECK(quant_bit_size == 8 && a_quant_bit_size == 8)
      << "scaled_matmul only supports w8a8 quantization (quant_bit_size "
         "scaled_matmul only supports w8a8 quantization (quant_bit_size "
         "== 8, a_quant_bit_size == 8) for now. "
         "Got quant_bit_size = "
      << quant_bit_size << ", a_quant_bit_size = " << a_quant_bit_size;

  // Only support smooth_quant algorithm for now
  std::string quant_algo = "smooth_quant";
  std::string a_quant_layout = (a_scale.value().dim() == 1)
                                   ? "quantize_per_token"
                                   : "quantize_group_wise";
  std::string b_quant_layout = "quantize_per_channel";
  if (b_scale.dim() > 1) {
    if (b_scale.size(0) < b.size(0)) {
      b_quant_layout = "quantize_per_block";
    } else {
      b_quant_layout = "quantize_group_wise";
    }
  }
  std::optional<torch::Tensor> gemm_output_scale = std::nullopt;

  at::ScalarType torch_half = at::ScalarType::Half;
  at::ScalarType torch_bfloat16 = at::ScalarType::BFloat16;

  CHECK(output_dtype == torch_half || output_dtype == torch_bfloat16)
      << "output dtype must be half or bfloat16, but got: " << output_dtype;

  // Select output tensor
  torch::Tensor output_tensor;
  if (output.has_value()) {
    output_tensor = output.value();
  } else {
    output_tensor = at::empty(
        {a.size(0), b.size(0)},
        torch::TensorOptions().dtype(output_dtype).device(a.device()));
  }

  // Call underlying kernel for smooth_quant
  tmo::torch_api::scaled_matmul(output_tensor,
                                a,
                                b,
                                a_scale,
                                /*a_zero=*/std::nullopt,
                                a_calib,
                                b_scale,
                                /*b_zero=*/std::nullopt,
                                b_calib,
                                bias,
                                c,
                                /*c_scale=*/std::nullopt,
                                /*c_zero=*/std::nullopt,
                                gemm_output_scale,
                                /*gemm_output_zero=*/std::nullopt,
                                quant_algo,
                                a_quant_layout,
                                b_quant_layout,
                                a_quant_bit_size,
                                quant_bit_size,
                                act_mode,
                                use_hp_active,
                                /*act_coef=*/1.0,
                                alpha,
                                beta,
                                /*trans_a=*/false,
                                /*trans_b=*/true);
  return output_tensor;
}

}  // namespace xllm::kernel::mlu

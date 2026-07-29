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

#include "kernels/dcu/aiter_quant_adapter.h"

#include <glog/logging.h>

#include "quant_api.h"

namespace xllm {
namespace kernel {
namespace dcu {
namespace aiter {
namespace {

void check_per_token_quant_inputs(const torch::Tensor& input,
                                  const std::optional<torch::Tensor>& output) {
  CHECK(input.defined()) << "dcu::aiter: input must be defined";
  CHECK(input.is_cuda()) << "dcu::aiter: input must be on DCU";
  CHECK(input.is_contiguous()) << "dcu::aiter: input must be contiguous";
  CHECK_EQ(input.dim(), 2) << "dcu::aiter: input must be [M,K], got "
                           << input.sizes();
  CHECK(input.scalar_type() == torch::kFloat32 ||
        input.scalar_type() == torch::kFloat16 ||
        input.scalar_type() == torch::kBFloat16)
      << "dcu::aiter: input must be fp32, fp16, or bf16";

  if (output.has_value() && output.value().defined()) {
    const torch::Tensor& output_tensor = output.value();
    CHECK(output_tensor.is_cuda()) << "dcu::aiter: output must be on DCU";
    CHECK(output_tensor.device() == input.device())
        << "dcu::aiter: output must be on the same DCU device";
    CHECK(output_tensor.is_contiguous())
        << "dcu::aiter: output must be contiguous";
    CHECK(output_tensor.scalar_type() == torch::kFloat8_e4m3fn ||
          output_tensor.scalar_type() == torch::kFloat8_e5m2)
        << "dcu::aiter: output must be FP8";
    CHECK_EQ(output_tensor.sizes(), input.sizes())
        << "dcu::aiter: output shape must match input";
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> per_token_quant_fp8(
    const torch::Tensor& input,
    const std::optional<torch::Tensor>& output) {
  check_per_token_quant_inputs(input, output);

  torch::Tensor result_output;
  if (output.has_value() && output.value().defined()) {
    result_output = output.value();
  } else {
    result_output =
        torch::empty_like(input, input.options().dtype(torch::kFloat8_e4m3fn));
  }
  torch::Tensor result_scale =
      torch::empty({input.size(0), 1}, input.options().dtype(torch::kFloat32));

  ::aiter::native::dynamic_per_token_scaled_quant(result_output,
                                                  input,
                                                  result_scale,
                                                  std::nullopt,
                                                  /*shuffle_scale=*/false,
                                                  std::nullopt,
                                                  /*num_rows_factor=*/1);
  return std::make_tuple(result_output, result_scale);
}

}  // namespace aiter
}  // namespace dcu
}  // namespace kernel
}  // namespace xllm

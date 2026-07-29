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

#include "layers/mlu/qwen3_5/qwen3_next_rms_norm.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "platform/device.h"
#include "platform/platform.h"

namespace xllm {
namespace layer {
namespace {

TEST(Qwen3NextRMSNormMluTest, FusesResidualAndReturnsUpdatedState) {
  constexpr int64_t kHiddenSize = 128;
  constexpr double kEps = 1e-6;
  torch::Device device(Platform::type_torch(), 0);
  torch::TensorOptions cpu_options =
      torch::TensorOptions().dtype(torch::kFloat);
  torch::TensorOptions mlu_options = torch::TensorOptions()
                                         .dtype(torch::kBFloat16)
                                         .device(device)
                                         .requires_grad(false);

  torch::Tensor input_cpu =
      torch::arange(2 * kHiddenSize, cpu_options).reshape({2, kHiddenSize});
  input_cpu = (input_cpu.remainder(17) - 8.0f) / 32.0f;
  torch::Tensor residual_cpu =
      torch::arange(2 * kHiddenSize, cpu_options).reshape({2, kHiddenSize});
  residual_cpu = (residual_cpu.remainder(13) - 6.0f) / 16.0f;
  torch::Tensor weight_cpu =
      (torch::arange(kHiddenSize, cpu_options).remainder(11) - 5.0f) / 64.0f;

  torch::Tensor input = input_cpu.to(mlu_options);
  torch::Tensor residual = residual_cpu.to(mlu_options);
  Qwen3NextRMSNormMlu norm(kHiddenSize, kEps, mlu_options);
  norm->weight().copy_(weight_cpu.to(mlu_options));

  auto [output, residual_out] = norm->forward(input, residual);
  Device(device).synchronize_default_stream();

  ASSERT_TRUE(residual_out.has_value());
  torch::Tensor quantized_input = input.to(torch::kCPU).to(torch::kFloat);
  torch::Tensor quantized_residual =
      residual_cpu.to(torch::kBFloat16).to(torch::kFloat);
  torch::Tensor expected_residual = quantized_input + quantized_residual;
  torch::Tensor variance =
      expected_residual.pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
  torch::Tensor quantized_weight =
      weight_cpu.to(torch::kBFloat16).to(torch::kFloat);
  torch::Tensor expected_output = expected_residual *
                                  torch::rsqrt(variance + kEps) *
                                  (1.0f + quantized_weight);
  torch::Tensor actual_residual =
      residual_out.value().to(torch::kCPU).to(torch::kFloat);
  torch::Tensor actual_output = output.to(torch::kCPU).to(torch::kFloat);

  EXPECT_TRUE(torch::allclose(actual_residual,
                              expected_residual,
                              /*rtol=*/1e-2,
                              /*atol=*/1e-2));
  EXPECT_TRUE(torch::allclose(actual_output,
                              expected_output,
                              /*rtol=*/1e-2,
                              /*atol=*/1e-2));
}

}  // namespace
}  // namespace layer
}  // namespace xllm

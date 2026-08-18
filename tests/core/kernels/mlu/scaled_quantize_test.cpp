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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "kernels/mlu/mlu_ops_api.h"

namespace xllm {
namespace {

torch::Device mlu_device() {
  return torch::Device(torch::kPrivateUse1, /*index=*/0);
}

torch::Tensor run_active(const torch::Tensor& input,
                         const std::string& act_mode,
                         bool is_gated = false) {
  std::vector<int64_t> output_shape(input.sizes().begin(), input.sizes().end());
  if (is_gated) {
    output_shape.back() /= 2;
  }
  torch::Tensor output = torch::empty(output_shape, input.options());
  kernel::mlu::active(input,
                      output,
                      std::nullopt,
                      std::nullopt,
                      act_mode,
                      is_gated,
                      /*start_expert_id=*/0,
                      /*expert_size=*/0);
  return output;
}

torch::Tensor quant_input() {
  return torch::tensor({{1.8828125,
                         4.03125,
                         -3.875,
                         -2.3125,
                         1.59375,
                         -3.265625,
                         4.25,
                         1.1640625,
                         -1.390625,
                         0.32421875,
                         1.5625,
                         -1.765625,
                         -3.875,
                         0.03369140625,
                         0.09130859375,
                         0.1005859375},
                        {-0.73046875,
                         3.203125,
                         -1.3984375,
                         -0.484375,
                         2.0625,
                         -3.140625,
                         1.3359375,
                         -1.109375,
                         2.390625,
                         -2.71875,
                         0.1845703125,
                         0.48828125,
                         -4.03125,
                         -3.640625,
                         1.9140625,
                         -1.453125}},
                       torch::TensorOptions().dtype(torch::kFloat32))
      .to(torch::kBFloat16);
}

torch::Tensor quant_smooth() {
  return torch::tensor({1.4454224,
                        0.2591031,
                        0.6292535,
                        0.3822536,
                        1.2996036,
                        0.9782565,
                        0.8601090,
                        0.8752215,
                        0.4137633,
                        1.2126962,
                        1.0186944,
                        0.4824153,
                        1.2822158,
                        0.9849257,
                        0.2746393,
                        1.4034402},
                       torch::TensorOptions().dtype(torch::kFloat32));
}

TEST(ActiveMluTest, GeluModesMatchTorchApproximations) {
  torch::Device device = mlu_device();
  torch::DeviceGuard guard(device);
  torch::Tensor cpu_input =
      torch::linspace(
          -5.0, 5.0, 321, torch::TensorOptions().dtype(torch::kFloat32))
          .reshape({1, 321})
          .to(torch::kBFloat16);
  torch::Tensor input = cpu_input.to(device);

  torch::Tensor tanh_output = run_active(input, "gelu_pytorch_tanh");
  torch::Tensor exact_output = run_active(input, "gelu");
  torch::Tensor tanh_reference = torch::gelu(cpu_input, "tanh");
  torch::Tensor exact_reference = torch::gelu(cpu_input, "none");

  EXPECT_EQ(tanh_output.sizes(), input.sizes());
  EXPECT_EQ(tanh_output.scalar_type(), input.scalar_type());
  EXPECT_EQ(tanh_output.device(), input.device());
  EXPECT_TRUE(torch::allclose(tanh_output.cpu(),
                              tanh_reference,
                              /*rtol=*/5e-3,
                              /*atol=*/5e-5));
  EXPECT_TRUE(torch::allclose(exact_output.cpu(),
                              exact_reference,
                              /*rtol=*/5e-3,
                              /*atol=*/5e-3));
  EXPECT_FALSE(torch::equal(tanh_output.cpu(), exact_output.cpu()));
}

TEST(ScaledQuantizeMluTest, GeluPytorchTanhMatchesUnfusedReference) {
  torch::Device device = mlu_device();
  torch::DeviceGuard guard(device);
  torch::Tensor cpu_input = quant_input();
  torch::Tensor input = cpu_input.to(device);
  torch::Tensor smooth = quant_smooth().to(device);
  torch::Tensor activated = torch::gelu(cpu_input, "tanh").to(device);

  auto [expected_output, expected_scale] =
      kernel::mlu::scaled_quantize(activated,
                                   smooth,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   "none");
  torch::Tensor output = torch::empty_like(expected_output);
  torch::Tensor output_scale = torch::empty_like(expected_scale);
  auto [actual_output, actual_scale] =
      kernel::mlu::scaled_quantize(input,
                                   smooth,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   output,
                                   output_scale,
                                   "gelu_pytorch_tanh");

  EXPECT_EQ(actual_output.data_ptr(), output.data_ptr());
  EXPECT_EQ(actual_scale.data_ptr(), output_scale.data_ptr());
  EXPECT_TRUE(torch::equal(actual_output.cpu(), expected_output.cpu()));
  EXPECT_TRUE(torch::allclose(actual_scale.cpu(),
                              expected_scale.cpu(),
                              /*rtol=*/1e-6,
                              /*atol=*/1e-7));
}

TEST(ScaledQuantizeMluTest, GatedGeluPytorchTanhReducesShapeOnce) {
  torch::Device device = mlu_device();
  torch::DeviceGuard guard(device);
  torch::Tensor input = quant_input().to(device);
  torch::Tensor smooth =
      quant_smooth().slice(/*dim=*/0, /*start=*/0, /*end=*/8).to(device);
  torch::Tensor activated = run_active(input,
                                       "gelu_pytorch_tanh",
                                       /*is_gated=*/true);
  auto [expected_output, expected_scale] =
      kernel::mlu::scaled_quantize(activated,
                                   smooth,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   "none");

  auto [actual_output, actual_scale] =
      kernel::mlu::scaled_quantize(input,
                                   smooth,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   "gelu_pytorch_tanh",
                                   /*active_coef=*/1.0,
                                   /*is_gated=*/true);

  EXPECT_EQ(actual_output.sizes(), expected_output.sizes());
  EXPECT_TRUE(torch::equal(actual_output.cpu(), expected_output.cpu()));
  EXPECT_TRUE(torch::allclose(actual_scale.cpu(),
                              expected_scale.cpu(),
                              /*rtol=*/1e-6,
                              /*atol=*/1e-7));
}

}  // namespace
}  // namespace xllm

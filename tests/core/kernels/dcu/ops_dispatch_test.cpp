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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>
#include <string>

#include "core/kernels/ops_api.h"

namespace xllm::kernel {
namespace {

TEST(FusedLayerNormDispatchDcuTest, MatchesCpuLayerNormWithBiasAndResidual) {
  ASSERT_TRUE(torch::cuda::is_available());
  for (torch::ScalarType dtype : {torch::kFloat32, torch::kBFloat16}) {
    for (bool add_residual : {false, true}) {
      SCOPED_TRACE(dtype == torch::kFloat32 ? "float32" : "bfloat16");
      SCOPED_TRACE(add_residual ? "with residual" : "without residual");
      auto options = torch::TensorOptions().device(torch::kCUDA).dtype(dtype);
      FusedLayerNormParams params;
      params.input = (torch::arange(96, options).reshape({3, 32}) / 17 + 2);
      params.output = torch::empty_like(params.input);
      params.weight = torch::linspace(0.5, 1.5, 32, options);
      params.beta = torch::linspace(-0.3, 0.7, 32, options);
      params.eps = 1e-6;
      params.mode = "layernorm";
      auto reference_input = params.input.cpu().to(torch::kFloat32);
      if (add_residual) {
        params.residual = torch::full_like(params.input, 0.25);
        reference_input =
            reference_input + params.residual->cpu().to(torch::kFloat32);
      }
      auto expected = torch::layer_norm(reference_input,
                                        {32},
                                        params.weight.cpu().to(torch::kFloat32),
                                        params.beta->cpu().to(torch::kFloat32),
                                        params.eps);
      fused_layernorm(params);
      const double tolerance = dtype == torch::kFloat32 ? 1e-5 : 0.04;
      EXPECT_TRUE(torch::allclose(params.output.cpu().to(torch::kFloat32),
                                  expected,
                                  tolerance,
                                  tolerance));
      if (add_residual) {
        ASSERT_TRUE(params.residual_out.has_value());
        EXPECT_TRUE(
            torch::allclose(params.residual_out->cpu().to(torch::kFloat32),
                            reference_input,
                            tolerance,
                            tolerance));
      }
    }
  }
}

TEST(FusedLayerNormDispatchDcuTest, RmsNormModeStillMatchesReference) {
  ASSERT_TRUE(torch::cuda::is_available());
  for (bool add_residual : {false, true}) {
    SCOPED_TRACE(add_residual ? "with residual" : "without residual");
    auto options =
        torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    FusedLayerNormParams params;
    params.input = torch::arange(96, options).reshape({3, 32}) / 17 + 2;
    params.output = torch::empty_like(params.input);
    params.weight = torch::linspace(0.5, 1.5, 32, options);
    params.eps = 1e-6;
    params.mode = "rmsnorm";
    auto reference_input = params.input.cpu();
    if (add_residual) {
      params.residual = torch::full_like(params.input, 0.25);
      reference_input = reference_input + params.residual->cpu();
    }
    auto expected =
        reference_input *
        torch::rsqrt(reference_input.square().mean(-1, true) + params.eps) *
        params.weight.cpu();
    fused_layernorm(params);
    EXPECT_TRUE(torch::allclose(params.output.cpu(), expected, 1e-5, 1e-5));
    if (add_residual) {
      ASSERT_TRUE(params.residual_out.has_value());
      EXPECT_TRUE(torch::allclose(
          params.residual_out->cpu(), reference_input, 1e-5, 1e-5));
    }
  }
}

void check_activation_dispatch(bool is_gated) {
  for (torch::ScalarType dtype : {torch::kFloat32, torch::kBFloat16}) {
    for (const std::string mode :
         {"silu", "gelu", "gelu_tanh", "gelu_pytorch_tanh"}) {
      SCOPED_TRACE(mode);
      SCOPED_TRACE(dtype == torch::kFloat32 ? "float32" : "bfloat16");
      auto options = torch::TensorOptions().device(torch::kCUDA).dtype(dtype);
      const int64_t width = is_gated ? 64 : 33;
      ActivationParams params;
      params.input =
          torch::linspace(-4, 4, 3 * width, options).reshape({3, width});
      params.output =
          torch::full({3, is_gated ? width / 2 : width}, NAN, options);
      params.act_mode = mode;
      params.is_gated = is_gated;
      auto cpu_input = params.input.cpu().to(torch::kFloat32);
      auto activated_input =
          is_gated ? cpu_input.slice(-1, 0, width / 2) : cpu_input;
      torch::Tensor expected;
      if (mode == "silu") {
        expected = torch::silu(activated_input);
      } else {
        expected =
            torch::gelu(activated_input, mode == "gelu" ? "none" : "tanh");
      }
      if (is_gated) {
        expected = expected * cpu_input.slice(-1, width / 2, width);
      }
      active(params);
      const double tolerance = dtype == torch::kFloat32 ? 1e-5 : 0.02;
      EXPECT_TRUE(torch::allclose(params.output.cpu().to(torch::kFloat32),
                                  expected,
                                  tolerance,
                                  tolerance));
    }
  }
}

TEST(ActivationDispatchDcuTest, NonGatedPreservesEveryFeature) {
  ASSERT_TRUE(torch::cuda::is_available());
  check_activation_dispatch(false);
}

TEST(ActivationDispatchDcuTest, GatedStillMatchesReference) {
  ASSERT_TRUE(torch::cuda::is_available());
  check_activation_dispatch(true);
}

}  // namespace
}  // namespace xllm::kernel

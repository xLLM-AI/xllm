/* Copyright 2025-2026 The xLLM Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ===========================================================================*/

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "cuda_ops_api.h"
#include "cutlass_extensions/common.hpp"

class CutlassScaledMMTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!torch::cuda::is_available()) {
      GTEST_SKIP() << "CUDA not available, skipping test.";
    }
    device_ = torch::Device(torch::kCUDA);

    // Check if FP8 is supported
    int compute_capability = xllm::kernel::cuda::get_sm_version_num();
    if (compute_capability < 89) {
      GTEST_SKIP() << "FP8 requires compute capability >= 8.9 (Ada Lovelace or "
                      "Hopper), current: "
                   << compute_capability;
    }
  }

  torch::Device device_ = torch::kCPU;
};

// Test basic FP8 W8A8 matrix multiplication
TEST_F(CutlassScaledMMTest, BasicFP8W8A8Test) {
  const int64_t M = 128;
  const int64_t N = 256;
  const int64_t K = 512;

  auto fp32_options =
      torch::TensorOptions().device(device_).dtype(torch::kFloat32);

  auto fp16_options =
      torch::TensorOptions().device(device_).dtype(torch::kFloat16);

  // Create input matrices in FP8 format (row-major for A, column-major for B)
  // Scale down the input range to avoid FP8 quantization error
  torch::Tensor a = torch::randn({M, K}, fp32_options) * 0.5f;
  torch::Tensor b = torch::randn({K, N}, fp32_options) * 0.5f;

  // Convert to FP8
  torch::Tensor a_fp8 = a.to(torch::kFloat8_e4m3fn);
  torch::Tensor b_fp8 = b.to(torch::kFloat8_e4m3fn);

  // Transpose B to column-major format
  b_fp8 = b_fp8.t().contiguous().t();

  // Create scales (per-tensor scaling)
  torch::Tensor a_scales = torch::ones({1}, fp32_options);
  torch::Tensor b_scales = torch::ones({1}, fp32_options);

  // Create output tensor (must be Float16 for FP8 GEMM)
  torch::Tensor c = torch::zeros({M, N}, fp16_options);

  // Call cutlass_scaled_mm
  ASSERT_NO_THROW({
    xllm::kernel::cuda::cutlass_scaled_mm(
        c, a_fp8, b_fp8, a_scales, b_scales, std::nullopt);
  });

  // Verify output shape
  EXPECT_EQ(c.size(0), M);
  EXPECT_EQ(c.size(1), N);

  // Compute reference result using FP32
  torch::Tensor c_ref = torch::matmul(a, b);

  // Check if results are close (allowing for FP8 quantization error)
  auto max_diff = (c.to(torch::kFloat32) - c_ref).abs().max().item<float>();
  LOG(INFO) << "Max difference between FP8 and FP32 result: " << max_diff;

  // FP8 has limited precision (4-bit exponent, 3-bit mantissa), so we use a
  // loose tolerance
  EXPECT_LT(max_diff, 2.0f);
}

// Test FP8 W8A8 with bias
TEST_F(CutlassScaledMMTest, FP8W8A8WithBiasTest) {
  const int64_t M = 64;
  const int64_t N = 128;
  const int64_t K = 256;

  auto fp32_options =
      torch::TensorOptions().device(device_).dtype(torch::kFloat32);

  auto fp16_options =
      torch::TensorOptions().device(device_).dtype(torch::kFloat16);

  // Create input matrices
  // Scale down the input range to avoid FP8 quantization error
  torch::Tensor a = torch::randn({M, K}, fp32_options) * 0.5f;
  torch::Tensor b = torch::randn({K, N}, fp32_options) * 0.5f;
  torch::Tensor bias = torch::randn({N}, fp16_options) * 0.5f;

  // Convert to FP8
  torch::Tensor a_fp8 = a.to(torch::kFloat8_e4m3fn);
  torch::Tensor b_fp8 = b.to(torch::kFloat8_e4m3fn);

  // Transpose B to column-major format
  b_fp8 = b_fp8.t().contiguous().t();

  // Create scales
  torch::Tensor a_scales = torch::ones({1}, fp32_options);
  torch::Tensor b_scales = torch::ones({1}, fp32_options);

  // Create output tensor (must be Float16 for FP8 GEMM)
  torch::Tensor c = torch::zeros({M, N}, fp16_options);

  // Call cutlass_scaled_mm with bias
  ASSERT_NO_THROW({
    xllm::kernel::cuda::cutlass_scaled_mm(
        c, a_fp8, b_fp8, a_scales, b_scales, bias);
  });

  // Verify output shape
  EXPECT_EQ(c.size(0), M);
  EXPECT_EQ(c.size(1), N);

  // Compute reference result
  torch::Tensor c_ref =
      torch::matmul(a, b) + bias.to(torch::kFloat32).unsqueeze(0);

  // Check if results are close (allowing for FP8 quantization error)
  auto max_diff = (c.to(torch::kFloat32) - c_ref).abs().max().item<float>();
  LOG(INFO) << "Max difference with bias: " << max_diff;

  // FP8 has limited precision, so we use a loose tolerance
  EXPECT_LT(max_diff, 2.0f);
}

// Test FP8 W8A8 with per-token/per-channel scaling
TEST_F(CutlassScaledMMTest, FP8W8A8WithScalingTest) {
  const int64_t M = 64;
  const int64_t N = 128;
  const int64_t K = 256;

  auto fp32_options =
      torch::TensorOptions().device(device_).dtype(torch::kFloat32);

  auto fp16_options =
      torch::TensorOptions().device(device_).dtype(torch::kFloat16);

  // Create input matrices
  torch::Tensor a = torch::randn({M, K}, fp32_options) * 0.5f;
  torch::Tensor b = torch::randn({K, N}, fp32_options) * 0.5f;

  // Create scales for per-token/per-channel quantization
  torch::Tensor a_scales = torch::rand({M}, fp32_options) * 0.1f + 0.9f;
  torch::Tensor b_scales = torch::rand({N}, fp32_options) * 0.1f + 0.9f;

  // Apply scaling and convert to FP8
  torch::Tensor a_scaled = a / a_scales.unsqueeze(1);
  torch::Tensor b_scaled = b / b_scales.unsqueeze(0);

  torch::Tensor a_fp8 = a_scaled.to(torch::kFloat8_e4m3fn);
  torch::Tensor b_fp8 = b_scaled.to(torch::kFloat8_e4m3fn);

  // Transpose B to column-major format
  b_fp8 = b_fp8.t().contiguous().t();

  // Create output tensor (must be Float16 for FP8 GEMM)
  torch::Tensor c = torch::zeros({M, N}, fp16_options);

  // Call cutlass_scaled_mm
  ASSERT_NO_THROW({
    xllm::kernel::cuda::cutlass_scaled_mm(
        c, a_fp8, b_fp8, a_scales, b_scales, std::nullopt);
  });

  // Verify output shape
  EXPECT_EQ(c.size(0), M);
  EXPECT_EQ(c.size(1), N);

  // Compute reference result
  torch::Tensor c_ref = torch::matmul(a, b);

  // Check if results are close
  auto max_diff = (c.to(torch::kFloat32) - c_ref).abs().max().item<float>();
  auto mean_diff = (c.to(torch::kFloat32) - c_ref).abs().mean().item<float>();
  LOG(INFO) << "Max difference with scaling: " << max_diff;
  LOG(INFO) << "Mean difference with scaling: " << mean_diff;

  EXPECT_LT(max_diff, 2.0f);
  EXPECT_LT(mean_diff, 0.5f);
}

// Test different matrix sizes
TEST_F(CutlassScaledMMTest, DifferentSizesTest) {
  std::vector<std::tuple<int64_t, int64_t, int64_t>> test_sizes = {
      {16, 32, 64},     // Small
      {128, 128, 128},  // Square
      {256, 512, 384},  // Medium
      {512, 1024, 768}  // Large
  };

  auto fp32_options =
      torch::TensorOptions().device(device_).dtype(torch::kFloat32);

  auto fp16_options =
      torch::TensorOptions().device(device_).dtype(torch::kFloat16);

  for (const auto& [M, N, K] : test_sizes) {
    LOG(INFO) << "Testing size: M=" << M << ", N=" << N << ", K=" << K;

    // Create input matrices
    torch::Tensor a = torch::randn({M, K}, fp32_options);
    torch::Tensor b = torch::randn({K, N}, fp32_options);

    torch::Tensor a_fp8 = a.to(torch::kFloat8_e4m3fn);
    torch::Tensor b_fp8 = b.to(torch::kFloat8_e4m3fn);

    // Transpose B to column-major format
    b_fp8 = b_fp8.t().contiguous().t();

    torch::Tensor a_scales = torch::ones({1}, fp32_options);
    torch::Tensor b_scales = torch::ones({1}, fp32_options);
    // Create output tensor (must be Float16 for FP8 GEMM)
    torch::Tensor c = torch::zeros({M, N}, fp16_options);

    // Should not throw for valid sizes
    ASSERT_NO_THROW({
      xllm::kernel::cuda::cutlass_scaled_mm(
          c, a_fp8, b_fp8, a_scales, b_scales, std::nullopt);
    });

    // Verify output shape
    EXPECT_EQ(c.size(0), M);
    EXPECT_EQ(c.size(1), N);
  }
}

// Test error handling for invalid inputs
TEST_F(CutlassScaledMMTest, InvalidInputTest) {
  const int64_t M = 64;
  const int64_t N = 128;
  const int64_t K = 256;

  auto fp32_options =
      torch::TensorOptions().device(device_).dtype(torch::kFloat32);

  auto fp16_options =
      torch::TensorOptions().device(device_).dtype(torch::kFloat16);

  torch::Tensor a_fp8 =
      torch::randn({M, K}, fp32_options).to(torch::kFloat8_e4m3fn);
  torch::Tensor b_fp8 =
      torch::randn({K, N}, fp32_options).to(torch::kFloat8_e4m3fn);
  b_fp8 = b_fp8.t().contiguous().t();

  torch::Tensor a_scales = torch::ones({1}, fp32_options);
  torch::Tensor b_scales = torch::ones({1}, fp32_options);
  // Create output tensor (must be Float16 for FP8 GEMM)
  torch::Tensor c = torch::zeros({M, N}, fp16_options);

  // Test mismatched dimensions
  torch::Tensor b_wrong =
      torch::randn({K + 1, N}, fp32_options).to(torch::kFloat8_e4m3fn);
  b_wrong = b_wrong.t().contiguous().t();

  EXPECT_THROW(
      {
        xllm::kernel::cuda::cutlass_scaled_mm(
            c, a_fp8, b_wrong, a_scales, b_scales, std::nullopt);
      },
      c10::Error);

  // Test invalid bias size
  torch::Tensor bias_wrong = torch::randn(
      {N + 1}, torch::TensorOptions().device(device_).dtype(torch::kFloat16));

  EXPECT_THROW(
      {
        xllm::kernel::cuda::cutlass_scaled_mm(
            c, a_fp8, b_fp8, a_scales, b_scales, bias_wrong);
      },
      c10::Error);
}

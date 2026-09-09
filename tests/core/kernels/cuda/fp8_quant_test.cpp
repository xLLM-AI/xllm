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
// clang-format off
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "cuda_ops_api.h"
#include "cutlass_extensions/common.hpp"
// clang-format on
class StaticScaledFP8QuantTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!torch::cuda::is_available()) {
      GTEST_SKIP() << "CUDA not available, skipping test.";
    }
    device_ = torch::Device(torch::kCUDA);

    // Check if FP8 is supported (requires compute capability >= 8.9)
    int compute_capability = xllm::kernel::cuda::get_sm_version_num();
    if (compute_capability < 89) {
      GTEST_SKIP() << "FP8 requires compute capability >= 8.9 (Ada Lovelace or "
                      "Hopper), current: "
                   << compute_capability;
    }
  }

  torch::Device device_ = torch::kCPU;
};

// Test basic FP8 quantization with float32 input
TEST_F(StaticScaledFP8QuantTest, BasicFloat32InputTest) {
  const int64_t num_tokens = 128;
  const int64_t hidden_size = 256;

  auto fp32_options =
      torch::TensorOptions().device(device_).dtype(torch::kFloat32);
  auto fp8_options =
      torch::TensorOptions().device(device_).dtype(torch::kFloat8_e4m3fn);

  // Create input tensor with values in a reasonable range for FP8
  torch::Tensor input = torch::randn({num_tokens, hidden_size}, fp32_options);

  // Create scale tensor (scale should be positive)
  torch::Tensor scale = torch::tensor({1.0f}, fp32_options);

  // Create output tensor
  torch::Tensor out = torch::empty({num_tokens, hidden_size}, fp8_options);

  // Call static_scaled_fp8_quant
  ASSERT_NO_THROW(
      { xllm::kernel::cuda::static_scaled_fp8_quant(out, input, scale); });

  // Verify output shape
  EXPECT_EQ(out.size(0), num_tokens);
  EXPECT_EQ(out.size(1), hidden_size);

  // Convert back to float and verify the quantization is reasonable
  torch::Tensor out_fp32 = out.to(torch::kFloat32);

  // Check that output values are finite
  EXPECT_TRUE(out_fp32.isfinite().all().item<bool>());

  // Verify that quantization preserves relative ordering for most values
  auto input_sign = input.sign();
  auto out_sign = out_fp32.sign();
  auto sign_match_ratio =
      (input_sign == out_sign).to(torch::kFloat32).mean().item<float>();
  LOG(INFO) << "Sign match ratio: " << sign_match_ratio;
  EXPECT_GT(sign_match_ratio, 0.9f);
}

// Test FP8 quantization with float16 input
TEST_F(StaticScaledFP8QuantTest, Float16InputTest) {
  const int64_t num_tokens = 64;
  const int64_t hidden_size = 512;

  auto fp16_options =
      torch::TensorOptions().device(device_).dtype(torch::kFloat16);
  auto fp32_options =
      torch::TensorOptions().device(device_).dtype(torch::kFloat32);
  auto fp8_options =
      torch::TensorOptions().device(device_).dtype(torch::kFloat8_e4m3fn);

  // Create float16 input tensor
  torch::Tensor input = torch::randn({num_tokens, hidden_size}, fp16_options);

  // Create scale tensor
  torch::Tensor scale = torch::tensor({0.5f}, fp32_options);

  // Create output tensor
  torch::Tensor out = torch::empty({num_tokens, hidden_size}, fp8_options);

  // Call static_scaled_fp8_quant
  ASSERT_NO_THROW(
      { xllm::kernel::cuda::static_scaled_fp8_quant(out, input, scale); });

  // Verify output shape
  EXPECT_EQ(out.size(0), num_tokens);
  EXPECT_EQ(out.size(1), hidden_size);

  // Verify output is finite
  torch::Tensor out_fp32 = out.to(torch::kFloat32);
  EXPECT_TRUE(out_fp32.isfinite().all().item<bool>());
}

// Test FP8 quantization with bfloat16 input
TEST_F(StaticScaledFP8QuantTest, BFloat16InputTest) {
  const int64_t num_tokens = 32;
  const int64_t hidden_size = 128;

  auto bf16_options =
      torch::TensorOptions().device(device_).dtype(torch::kBFloat16);
  auto fp32_options =
      torch::TensorOptions().device(device_).dtype(torch::kFloat32);
  auto fp8_options =
      torch::TensorOptions().device(device_).dtype(torch::kFloat8_e4m3fn);

  // Create bfloat16 input tensor
  torch::Tensor input = torch::randn({num_tokens, hidden_size}, bf16_options);

  // Create scale tensor
  torch::Tensor scale = torch::tensor({2.0f}, fp32_options);

  // Create output tensor
  torch::Tensor out = torch::empty({num_tokens, hidden_size}, fp8_options);

  // Call static_scaled_fp8_quant
  ASSERT_NO_THROW(
      { xllm::kernel::cuda::static_scaled_fp8_quant(out, input, scale); });

  // Verify output shape
  EXPECT_EQ(out.size(0), num_tokens);
  EXPECT_EQ(out.size(1), hidden_size);

  // Verify output is finite
  torch::Tensor out_fp32 = out.to(torch::kFloat32);
  EXPECT_TRUE(out_fp32.isfinite().all().item<bool>());
}

// Test FP8 quantization with different scales
TEST_F(StaticScaledFP8QuantTest, DifferentScalesTest) {
  const int64_t num_tokens = 64;
  const int64_t hidden_size = 256;

  auto fp32_options =
      torch::TensorOptions().device(device_).dtype(torch::kFloat32);
  auto fp8_options =
      torch::TensorOptions().device(device_).dtype(torch::kFloat8_e4m3fn);

  // Create input tensor with small values
  torch::Tensor input =
      torch::randn({num_tokens, hidden_size}, fp32_options) * 0.1f;

  std::vector<float> test_scales = {0.1f, 0.5f, 1.0f, 2.0f, 10.0f};

  for (float scale_val : test_scales) {
    LOG(INFO) << "Testing with scale: " << scale_val;

    torch::Tensor scale = torch::tensor({scale_val}, fp32_options);
    torch::Tensor out = torch::empty({num_tokens, hidden_size}, fp8_options);

    ASSERT_NO_THROW(
        { xllm::kernel::cuda::static_scaled_fp8_quant(out, input, scale); });

    // Verify output shape
    EXPECT_EQ(out.size(0), num_tokens);
    EXPECT_EQ(out.size(1), hidden_size);

    // Verify output is finite
    torch::Tensor out_fp32 = out.to(torch::kFloat32);
    EXPECT_TRUE(out_fp32.isfinite().all().item<bool>());
  }
}

// Test FP8 quantization with various tensor sizes
TEST_F(StaticScaledFP8QuantTest, DifferentSizesTest) {
  std::vector<std::pair<int64_t, int64_t>> test_sizes = {
      {1, 64},      // Single token
      {16, 128},    // Small batch
      {64, 256},    // Medium batch
      {128, 512},   // Large batch
      {256, 1024},  // Very large batch
      {512, 4096},  // Large hidden size
  };

  auto fp32_options =
      torch::TensorOptions().device(device_).dtype(torch::kFloat32);
  auto fp8_options =
      torch::TensorOptions().device(device_).dtype(torch::kFloat8_e4m3fn);

  for (const auto& [num_tokens, hidden_size] : test_sizes) {
    LOG(INFO) << "Testing size: num_tokens=" << num_tokens
              << ", hidden_size=" << hidden_size;

    torch::Tensor input = torch::randn({num_tokens, hidden_size}, fp32_options);
    torch::Tensor scale = torch::tensor({1.0f}, fp32_options);
    torch::Tensor out = torch::empty({num_tokens, hidden_size}, fp8_options);

    ASSERT_NO_THROW(
        { xllm::kernel::cuda::static_scaled_fp8_quant(out, input, scale); });

    EXPECT_EQ(out.size(0), num_tokens);
    EXPECT_EQ(out.size(1), hidden_size);
  }
}

// Test FP8 quantization with 3D tensor (batched)
TEST_F(StaticScaledFP8QuantTest, BatchedTensor3DTest) {
  const int64_t batch_size = 4;
  const int64_t seq_len = 32;
  const int64_t hidden_size = 128;

  auto fp32_options =
      torch::TensorOptions().device(device_).dtype(torch::kFloat32);
  auto fp8_options =
      torch::TensorOptions().device(device_).dtype(torch::kFloat8_e4m3fn);

  // Create 3D input tensor [batch_size, seq_len, hidden_size]
  torch::Tensor input =
      torch::randn({batch_size, seq_len, hidden_size}, fp32_options);

  torch::Tensor scale = torch::tensor({1.0f}, fp32_options);
  torch::Tensor out =
      torch::empty({batch_size, seq_len, hidden_size}, fp8_options);

  // Call static_scaled_fp8_quant
  ASSERT_NO_THROW(
      { xllm::kernel::cuda::static_scaled_fp8_quant(out, input, scale); });

  // Verify output shape
  EXPECT_EQ(out.size(0), batch_size);
  EXPECT_EQ(out.size(1), seq_len);
  EXPECT_EQ(out.size(2), hidden_size);

  // Verify output is finite
  torch::Tensor out_fp32 = out.to(torch::kFloat32);
  EXPECT_TRUE(out_fp32.isfinite().all().item<bool>());
}

// Test quantization accuracy with known values
TEST_F(StaticScaledFP8QuantTest, QuantizationAccuracyTest) {
  const int64_t num_tokens = 64;
  const int64_t hidden_size = 128;

  auto fp32_options =
      torch::TensorOptions().device(device_).dtype(torch::kFloat32);
  auto fp8_options =
      torch::TensorOptions().device(device_).dtype(torch::kFloat8_e4m3fn);

  // Create input tensor with values scaled to FP8 range
  // FP8 e4m3 has max value around 448
  torch::Tensor input =
      torch::randn({num_tokens, hidden_size}, fp32_options) * 0.5f;

  torch::Tensor scale = torch::tensor({1.0f}, fp32_options);
  torch::Tensor out = torch::empty({num_tokens, hidden_size}, fp8_options);

  xllm::kernel::cuda::static_scaled_fp8_quant(out, input, scale);

  // Convert back to float for comparison
  torch::Tensor out_fp32 = out.to(torch::kFloat32);

  // Calculate quantization error
  // Note: FP8 quantization has limited precision (4-bit exponent, 3-bit
  // mantissa) so we expect some error
  auto abs_error = (out_fp32 - input).abs();
  auto max_error = abs_error.max().item<float>();
  auto mean_error = abs_error.mean().item<float>();

  LOG(INFO) << "Max quantization error: " << max_error;
  LOG(INFO) << "Mean quantization error: " << mean_error;

  // FP8 has limited precision, but error should be bounded
  // For values around 0.5, the error should be relatively small
  EXPECT_LT(mean_error, 0.5f);
}

// Test with scale that compensates for large input values
TEST_F(StaticScaledFP8QuantTest, LargeInputWithScaleTest) {
  const int64_t num_tokens = 32;
  const int64_t hidden_size = 64;

  auto fp32_options =
      torch::TensorOptions().device(device_).dtype(torch::kFloat32);
  auto fp8_options =
      torch::TensorOptions().device(device_).dtype(torch::kFloat8_e4m3fn);

  // Create input tensor with larger values
  torch::Tensor input =
      torch::randn({num_tokens, hidden_size}, fp32_options) * 10.0f;

  // Use a larger scale to bring values into FP8 range
  // The kernel divides by scale, so larger scale compresses the range
  float input_max = input.abs().max().item<float>();
  float scale_val = input_max / 448.0f;  // FP8 e4m3 max is ~448
  if (scale_val < 1.0f) scale_val = 1.0f;

  torch::Tensor scale = torch::tensor({scale_val}, fp32_options);
  torch::Tensor out = torch::empty({num_tokens, hidden_size}, fp8_options);

  ASSERT_NO_THROW(
      { xllm::kernel::cuda::static_scaled_fp8_quant(out, input, scale); });

  // Verify output is finite
  torch::Tensor out_fp32 = out.to(torch::kFloat32);
  EXPECT_TRUE(out_fp32.isfinite().all().item<bool>());

  // Verify scaled output is within FP8 representable range
  float out_max = out_fp32.abs().max().item<float>();
  LOG(INFO) << "Input max: " << input_max << ", Scale: " << scale_val
            << ", Output max: " << out_max;
  EXPECT_LE(out_max, 450.0f);  // Allow some tolerance
}

// Test that zero values remain zero after quantization
TEST_F(StaticScaledFP8QuantTest, ZeroValuesTest) {
  const int64_t num_tokens = 16;
  const int64_t hidden_size = 32;

  auto fp32_options =
      torch::TensorOptions().device(device_).dtype(torch::kFloat32);
  auto fp8_options =
      torch::TensorOptions().device(device_).dtype(torch::kFloat8_e4m3fn);

  // Create input tensor with zeros
  torch::Tensor input = torch::zeros({num_tokens, hidden_size}, fp32_options);

  torch::Tensor scale = torch::tensor({1.0f}, fp32_options);
  torch::Tensor out = torch::empty({num_tokens, hidden_size}, fp8_options);

  xllm::kernel::cuda::static_scaled_fp8_quant(out, input, scale);

  // Verify all outputs are zero
  torch::Tensor out_fp32 = out.to(torch::kFloat32);
  EXPECT_TRUE((out_fp32 == 0).all().item<bool>());
}

// Test contiguity requirements
TEST_F(StaticScaledFP8QuantTest, ContiguityTest) {
  const int64_t num_tokens = 32;
  const int64_t hidden_size = 64;

  auto fp32_options =
      torch::TensorOptions().device(device_).dtype(torch::kFloat32);
  auto fp8_options =
      torch::TensorOptions().device(device_).dtype(torch::kFloat8_e4m3fn);

  // Create contiguous input tensor
  torch::Tensor input =
      torch::randn({num_tokens, hidden_size}, fp32_options).contiguous();

  // Ensure input is contiguous in the last dimension
  EXPECT_EQ(input.stride(-1), 1);

  torch::Tensor scale = torch::tensor({1.0f}, fp32_options);
  torch::Tensor out =
      torch::empty({num_tokens, hidden_size}, fp8_options).contiguous();

  ASSERT_NO_THROW(
      { xllm::kernel::cuda::static_scaled_fp8_quant(out, input, scale); });

  // Verify output
  EXPECT_EQ(out.size(0), num_tokens);
  EXPECT_EQ(out.size(1), hidden_size);
}

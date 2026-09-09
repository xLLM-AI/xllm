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

/**
 * SM120 (Blackwell) FP8 GEMM Configuration Performance Test
 *
 * This test verifies configuration selection and performance of SM120 FP8 GEMM
 * for different shapes. Uses the public cutlass_scaled_mm API for testing.
 *
 * Test objectives:
 * 1. Verify correctness of configuration selection for different M ranges
 * 2. Verify performance continuity at configuration boundaries
 * 3. Verify effectiveness of swap_ab strategy
 */

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "cuda_ops_api.h"
#include "cutlass_extensions/common.hpp"
#include "cutlass_w8a8/c3x/sm120_fp8_dispatch_policy.hpp"

namespace {

// Benchmark configuration
constexpr int kWarmupIters = 10;
constexpr int kBenchIters = 100;

// Result structure
struct ConfigResult {
  std::string config_name;
  std::string tile_shape;
  bool swap_ab;
  double latency_ms;
  double throughput_tflops;
};

// Compute TFLOPS
double ComputeTflops(int64_t m, int64_t n, int64_t k, double latency_s) {
  double flops = 2.0 * m * n * k;
  return flops / latency_s / 1e12;
}

// Create test tensors
std::tuple<torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor>
CreateTestTensors(int64_t m, int64_t n, int64_t k, torch::Device device) {
  auto fp32_options =
      torch::TensorOptions().device(device).dtype(torch::kFloat32);
  auto fp16_options =
      torch::TensorOptions().device(device).dtype(torch::kFloat16);

  // Create FP32 inputs and scale
  torch::Tensor a_fp32 = torch::randn({m, k}, fp32_options) * 0.5f;
  torch::Tensor b_fp32 = torch::randn({k, n}, fp32_options) * 0.5f;

  // Convert to FP8
  torch::Tensor a = a_fp32.to(torch::kFloat8_e4m3fn);
  torch::Tensor b = b_fp32.to(torch::kFloat8_e4m3fn);

  // Transpose B to column-major (required by CUTLASS)
  b = b.t().contiguous().t();

  // Scales
  torch::Tensor a_scales = torch::ones({1}, fp32_options);
  torch::Tensor b_scales = torch::ones({1}, fp32_options);

  // Output
  torch::Tensor c = torch::zeros({m, n}, fp16_options);

  return {a, b, a_scales, b_scales, c};
}

// Run benchmark with default dispatch
ConfigResult BenchmarkWithDispatch(int64_t m,
                                   int64_t n,
                                   int64_t k,
                                   torch::Device device) {
  auto [a, b, a_scales, b_scales, c] = CreateTestTensors(m, n, k, device);

  // Warmup
  for (int i = 0; i < kWarmupIters; ++i) {
    xllm::kernel::cuda::cutlass_scaled_mm(
        c, a, b, a_scales, b_scales, std::nullopt);
  }
  torch::cuda::synchronize();

  // Benchmark
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < kBenchIters; ++i) {
    xllm::kernel::cuda::cutlass_scaled_mm(
        c, a, b, a_scales, b_scales, std::nullopt);
  }
  torch::cuda::synchronize();
  auto end = std::chrono::high_resolution_clock::now();

  double total_time_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  double latency_ms = total_time_ms / kBenchIters;

  auto dispatch = xllm::kernel::cuda::select_sm120_dispatch_for_device(
      m, n, k, device.index());
  std::string config_name =
      xllm::kernel::cuda::get_sm120_dispatch_kernel_name(dispatch.kernel);

  return {
      config_name,
      xllm::kernel::cuda::get_sm120_dispatch_tile_shape_desc(dispatch.kernel),
      dispatch.swap_ab,
      latency_ms,
      ComputeTflops(m, n, k, latency_ms / 1000.0)};
}

// Verify numerical correctness
// Note: FP8 has low precision, errors accumulate for large matrices, need
// looser tolerance
// Important: Reference computation should use FP8-converted values to ensure
// fair comparison
bool VerifyCorrectness(int64_t m,
                       int64_t n,
                       int64_t k,
                       torch::Device device,
                       float tolerance = 0.5f) {  // FP8 needs looser tolerance
  auto fp32_options =
      torch::TensorOptions().device(device).dtype(torch::kFloat32);
  auto fp16_options =
      torch::TensorOptions().device(device).dtype(torch::kFloat16);

  // Create input tensors
  torch::Tensor a_fp32 = torch::randn({m, k}, fp32_options) * 0.5f;
  torch::Tensor b_fp32 = torch::randn({k, n}, fp32_options) * 0.5f;

  // Convert to FP8
  torch::Tensor a = a_fp32.to(torch::kFloat8_e4m3fn);
  torch::Tensor b_fp8 = b_fp32.to(torch::kFloat8_e4m3fn);

  // Convert FP8 back to FP32 for reference computation
  // This ensures fair comparison since FP8 has very low precision (4-bit
  // mantissa)
  torch::Tensor a_f32_from_fp8 = a.to(torch::kFloat32);
  torch::Tensor b_f32_from_fp8 = b_fp8.to(torch::kFloat32);

  // Compute reference result using FP8-precision inputs
  torch::Tensor ref = torch::mm(a_f32_from_fp8, b_f32_from_fp8);

  // Prepare B matrix (column-major for CUTLASS)
  torch::Tensor b = b_fp8.t().contiguous().t();

  torch::Tensor a_scales = torch::ones({1}, fp32_options);
  torch::Tensor b_scales = torch::ones({1}, fp32_options);
  torch::Tensor c = torch::zeros({m, n}, fp16_options);

  xllm::kernel::cuda::cutlass_scaled_mm(
      c, a, b, a_scales, b_scales, std::nullopt);

  // Compare results
  torch::Tensor c_fp32 = c.to(torch::kFloat32);
  torch::Tensor diff = torch::abs(c_fp32 - ref);
  torch::Tensor rel_err = diff / (torch::abs(ref) + 1e-6f);  // relative error

  float max_rel_err = rel_err.max().item<float>();
  float mean_rel_err = rel_err.mean().item<float>();

  // FP8 has low precision, allow larger relative errors
  // The accumulation is done in FP32/FP16, but input quantization causes errors
  return mean_rel_err < tolerance && max_rel_err < tolerance * 10;
}

}  // namespace

class CutlassScaledMMSM120ConfigTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!torch::cuda::is_available()) {
      GTEST_SKIP() << "CUDA not available, skipping test";
    }
    int current_device = 0;
    cudaError_t err = cudaGetDevice(&current_device);
    ASSERT_EQ(err, cudaSuccess)
        << "Failed to get current CUDA device: " << cudaGetErrorString(err);
    device_ = torch::Device(torch::kCUDA, current_device);

    // Check compute capability, SM120 test only runs on SM120 architecture
    int compute_capability = xllm::kernel::cuda::get_sm_version_num();
    if (compute_capability < 120) {
      GTEST_SKIP() << "SM120 test requires compute capability >= 12.0 "
                      "(Blackwell), current: "
                   << compute_capability;
    }

    compute_capability_ = compute_capability;
  }

  void PrintHeader(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(70, '=') << "\n";
  }

  void PrintTableHeader() {
    std::cout << std::setw(6) << "M" << std::setw(15) << "Config"
              << std::setw(15) << "TileShape" << std::setw(10) << "SwapAB"
              << std::setw(12) << "Latency(ms)" << std::setw(12) << "TFLOPS"
              << "\n";
    std::cout << std::string(70, '-') << "\n";
  }

  void PrintResult(int64_t m, const ConfigResult& result) {
    std::cout << std::setw(6) << m << std::setw(15) << result.config_name
              << std::setw(15) << result.tile_shape << std::setw(10)
              << (result.swap_ab ? "Yes" : "No") << std::setw(12) << std::fixed
              << std::setprecision(3) << result.latency_ms << std::setw(12)
              << std::setprecision(2) << result.throughput_tflops << "\n";
  }

  torch::Device device_ = torch::kCPU;
  int compute_capability_ = 0;
};

// ============================================================================
// Configuration Selection Correctness Tests
// ============================================================================

TEST_F(CutlassScaledMMSM120ConfigTest, ConfigSelection_M16) {
  PrintHeader("Test M <= 16 configuration selection (expected: 128x32_swap)");
  PrintTableHeader();

  std::vector<int64_t> m_values = {1, 4, 8, 12, 16};
  const int64_t N = 2048;
  const int64_t K = 2048;

  for (int64_t m : m_values) {
    auto result = BenchmarkWithDispatch(m, N, K, device_);
    PrintResult(m, result);
    EXPECT_EQ(result.config_name, "128x32_swap")
        << "M=" << m << " should use 128x32_swap configuration";
  }
}

TEST_F(CutlassScaledMMSM120ConfigTest, ConfigSelection_M64_SmallK) {
  PrintHeader(
      "Test M in (16, 64], K < 4096, N < 8192 configuration selection "
      "(expected: 128x64)");
  PrintTableHeader();

  std::vector<int64_t> m_values = {17, 24, 32, 48, 64};
  const int64_t N = 2048;  // N < 8192
  const int64_t K = 2048;  // K < 4096

  for (int64_t m : m_values) {
    auto result = BenchmarkWithDispatch(m, N, K, device_);
    PrintResult(m, result);
    // With small N and K, all M in (16, 64] should use 128x64 (no swap)
    EXPECT_EQ(result.config_name, "128x64")
        << "M=" << m << ", N=" << N << ", K=" << K
        << " should use 128x64 configuration";
  }
}

TEST_F(CutlassScaledMMSM120ConfigTest, ConfigSelection_M64_LargeK) {
  PrintHeader(
      "Test M in (16, 64], K >= 4096 configuration selection (expected: "
      "128x64_swap)");
  PrintTableHeader();

  std::vector<int64_t> m_values = {17, 24, 32, 48, 64};
  const int64_t N = 2048;
  const int64_t K = 11008;  // K >= 4096

  for (int64_t m : m_values) {
    auto result = BenchmarkWithDispatch(m, N, K, device_);
    PrintResult(m, result);
    // With K >= 4096, all M in (16, 64] should use 128x64_swap
    EXPECT_EQ(result.config_name, "128x64_swap")
        << "M=" << m << ", K=" << K << " should use 128x64_swap configuration";
  }
}

TEST_F(CutlassScaledMMSM120ConfigTest, ConfigSelection_M128) {
  PrintHeader("Test M in (64, 128] configuration selection (expected: 128x64)");
  PrintTableHeader();

  std::vector<int64_t> m_values = {65, 80, 96, 112, 128};
  const int64_t N = 2048;
  const int64_t K = 2048;

  for (int64_t m : m_values) {
    auto result = BenchmarkWithDispatch(m, N, K, device_);
    PrintResult(m, result);
    EXPECT_EQ(result.config_name, "128x64")
        << "M=" << m << " should use 128x64 configuration";
  }
}

TEST_F(CutlassScaledMMSM120ConfigTest, ConfigSelection_M256) {
  PrintHeader(
      "Test M in (128, 256] configuration selection (expected: 128x64)");
  PrintTableHeader();

  std::vector<int64_t> m_values = {129, 160, 192, 224, 256};
  const int64_t N = 2048;
  const int64_t K = 2048;

  for (int64_t m : m_values) {
    auto result = BenchmarkWithDispatch(m, N, K, device_);
    PrintResult(m, result);
    EXPECT_EQ(result.config_name, "128x64")
        << "M=" << m << " should use 128x64 configuration";
  }
}

TEST_F(CutlassScaledMMSM120ConfigTest, ConfigSelection_LargeM) {
  PrintHeader(
      "Test M > 256 configuration selection (expected: 128x64 or 128x128, "
      "depending on wave efficiency)");
  PrintTableHeader();

  std::vector<int64_t> m_values = {257, 320, 384, 448, 512};
  const int64_t N = 2048;
  const int64_t K = 2048;

  for (int64_t m : m_values) {
    auto result = BenchmarkWithDispatch(m, N, K, device_);
    PrintResult(m, result);
    // Wave efficiency based selection - either 128x64 or 128x128
    EXPECT_TRUE(result.config_name == "128x64" ||
                result.config_name == "128x128")
        << "M=" << m << " should use 128x64 or 128x128 configuration";
  }
}

TEST_F(CutlassScaledMMSM120ConfigTest, ConfigSelection_VeryLargeM) {
  PrintHeader(
      "Test M > 512 configuration selection (expected: 128x64 or 128x128, "
      "depending on wave efficiency)");
  PrintTableHeader();

  std::vector<int64_t> m_values = {513, 1024, 2048, 4000};
  const int64_t N = 2048;
  const int64_t K = 2048;

  for (int64_t m : m_values) {
    auto result = BenchmarkWithDispatch(m, N, K, device_);
    PrintResult(m, result);
    // Wave efficiency based selection - either 128x64 or 128x128
    EXPECT_TRUE(result.config_name == "128x64" ||
                result.config_name == "128x128")
        << "M=" << m << " should use 128x64 or 128x128 configuration";
  }
}

// ============================================================================
// Numerical Correctness Tests
// ============================================================================

TEST_F(CutlassScaledMMSM120ConfigTest, Correctness_AllConfigs) {
  PrintHeader("Numerical correctness test (all configurations)");

  // Test correctness for each configuration
  // Test cases must match the dispatch logic in
  // scaled_mm_sm120_fp8_dispatch.cuh
  std::vector<std::tuple<int64_t, int64_t, int64_t, std::string>> test_cases = {
      {8, 2048, 2048, "128x32_swap"},    // M<=16
      {32, 2048, 2048, "128x64"},        // M in (16,32], small N and K
      {32, 2048, 11008, "128x64_swap"},  // M in (16,32], K>=4096
      {96, 2048, 2048, "128x64"},        // M in (64,128]
      {192, 2048, 2048, "128x64"},       // M in (128,256]
  };

  for (const auto& [m, n, k, expected_config] : test_cases) {
    auto dispatch = xllm::kernel::cuda::select_sm120_dispatch_for_device(
        m, n, k, device_.index());
    std::string config =
        xllm::kernel::cuda::get_sm120_dispatch_kernel_name(dispatch.kernel);
    EXPECT_EQ(config, expected_config);

    bool correct = VerifyCorrectness(m, n, k, device_);
    EXPECT_TRUE(correct) << "Configuration " << config << " (M=" << m
                         << ", N=" << n << ", K=" << k
                         << ") numerical correctness verification failed";

    std::cout << "  " << std::setw(15) << config << ": M=" << std::setw(5) << m
              << ", N=" << std::setw(5) << n << ", K=" << std::setw(5) << k
              << " - " << (correct ? "PASS" : "FAIL") << std::endl;
  }
}

// ============================================================================
// Performance Tests - Specific Shapes
// ============================================================================

TEST_F(CutlassScaledMMSM120ConfigTest, Performance_Shape_N2048_K11008) {
  const int64_t N = 2048;
  const int64_t K = 11008;

  PrintHeader("Performance test: N=" + std::to_string(N) +
              ", K=" + std::to_string(K));
  PrintTableHeader();

  std::vector<int64_t> m_values = {1,
                                   8,
                                   16,
                                   17,
                                   32,
                                   64,
                                   65,
                                   128,
                                   129,
                                   256,
                                   257,
                                   512,
                                   1024,
                                   2048,
                                   3072,
                                   4000};

  for (int64_t m : m_values) {
    auto result = BenchmarkWithDispatch(m, N, K, device_);
    PrintResult(m, result);
  }
}

TEST_F(CutlassScaledMMSM120ConfigTest, Performance_Shape_N22016_K2048) {
  const int64_t N = 22016;
  const int64_t K = 2048;

  PrintHeader("Performance test: N=" + std::to_string(N) +
              ", K=" + std::to_string(K));
  PrintTableHeader();

  std::vector<int64_t> m_values = {1,
                                   8,
                                   16,
                                   17,
                                   32,
                                   64,
                                   65,
                                   128,
                                   129,
                                   256,
                                   257,
                                   512,
                                   1024,
                                   2048,
                                   3072,
                                   4000};

  for (int64_t m : m_values) {
    auto result = BenchmarkWithDispatch(m, N, K, device_);
    PrintResult(m, result);
  }
}

TEST_F(CutlassScaledMMSM120ConfigTest, Performance_Shape_N2048_K2048) {
  const int64_t N = 2048;
  const int64_t K = 2048;

  PrintHeader("Performance test: N=" + std::to_string(N) +
              ", K=" + std::to_string(K));
  PrintTableHeader();

  std::vector<int64_t> m_values = {1,
                                   8,
                                   16,
                                   17,
                                   32,
                                   64,
                                   65,
                                   128,
                                   129,
                                   256,
                                   257,
                                   512,
                                   1024,
                                   2048,
                                   3072,
                                   4000};

  for (int64_t m : m_values) {
    auto result = BenchmarkWithDispatch(m, N, K, device_);
    PrintResult(m, result);
  }
}

TEST_F(CutlassScaledMMSM120ConfigTest, Performance_Shape_N2560_K2048) {
  const int64_t N = 2560;
  const int64_t K = 2048;

  PrintHeader("Performance test: N=" + std::to_string(N) +
              ", K=" + std::to_string(K));
  PrintTableHeader();

  std::vector<int64_t> m_values = {1,
                                   8,
                                   16,
                                   17,
                                   32,
                                   64,
                                   65,
                                   128,
                                   129,
                                   256,
                                   257,
                                   512,
                                   1024,
                                   2048,
                                   3072,
                                   4000};

  for (int64_t m : m_values) {
    auto result = BenchmarkWithDispatch(m, N, K, device_);
    PrintResult(m, result);
  }
}

// ============================================================================
// Configuration Boundary Performance Continuity Tests
// ============================================================================

TEST_F(CutlassScaledMMSM120ConfigTest, BoundaryPerformance_M16_M64) {
  PrintHeader(
      "Configuration boundary test: 128x32_swap -> 128x64 (M=16 -> M=17)");
  PrintTableHeader();

  const int64_t N = 2048;
  const int64_t K = 2048;  // K < 4096, M64 does not use swap_ab

  std::vector<int64_t> m_values = {14, 15, 16, 17, 18, 19, 20};

  double prev_tflops = 0;
  std::string prev_config;

  for (int64_t m : m_values) {
    auto result = BenchmarkWithDispatch(m, N, K, device_);
    PrintResult(m, result);

    if (!prev_config.empty() && result.config_name != prev_config) {
      double change_pct =
          (result.throughput_tflops - prev_tflops) / prev_tflops * 100.0;
      std::cout << "  >>> Configuration switch: " << prev_config << " -> "
                << result.config_name
                << ", performance change: " << std::showpos << std::fixed
                << std::setprecision(1) << change_pct << "%" << std::noshowpos
                << std::endl;

      // Performance drop at config boundary is expected behavior.
      // The swap_ab technique is very efficient for small M, but when
      // switching to non-swap_ab config, there's inherent efficiency loss.
      // Allow up to 50% drop at boundaries (to be optimized in future).
      EXPECT_GT(change_pct, -50.0)
          << "Performance drop too large at configuration boundary";
    }

    prev_tflops = result.throughput_tflops;
    prev_config = result.config_name;
  }
}

TEST_F(CutlassScaledMMSM120ConfigTest, BoundaryPerformance_M64_M128) {
  PrintHeader(
      "Configuration boundary test: 128x64 (M=64 -> M=65, same configuration)");
  PrintTableHeader();

  const int64_t N = 2048;
  const int64_t K = 2048;

  std::vector<int64_t> m_values = {62, 63, 64, 65, 66, 67, 68};

  double prev_tflops = 0;
  std::string prev_config;

  for (int64_t m : m_values) {
    auto result = BenchmarkWithDispatch(m, N, K, device_);
    PrintResult(m, result);

    if (!prev_config.empty() && result.config_name != prev_config) {
      double change_pct =
          (result.throughput_tflops - prev_tflops) / prev_tflops * 100.0;
      std::cout << "  >>> Configuration switch: " << prev_config << " -> "
                << result.config_name
                << ", performance change: " << std::showpos << std::fixed
                << std::setprecision(1) << change_pct << "%" << std::noshowpos
                << std::endl;

      EXPECT_GT(change_pct, -20.0)
          << "Performance drop too large at configuration boundary";
    }

    prev_tflops = result.throughput_tflops;
    prev_config = result.config_name;
  }
}

TEST_F(CutlassScaledMMSM120ConfigTest, BoundaryPerformance_M128_M256) {
  PrintHeader(
      "Configuration boundary test: 128x64 (M=128 -> M=129, same "
      "configuration)");
  PrintTableHeader();

  const int64_t N = 2048;
  const int64_t K = 2048;

  std::vector<int64_t> m_values = {126, 127, 128, 129, 130, 131, 132};

  double prev_tflops = 0;
  std::string prev_config;

  for (int64_t m : m_values) {
    auto result = BenchmarkWithDispatch(m, N, K, device_);
    PrintResult(m, result);

    if (!prev_config.empty() && result.config_name != prev_config) {
      double change_pct =
          (result.throughput_tflops - prev_tflops) / prev_tflops * 100.0;
      std::cout << "  >>> Configuration switch: " << prev_config << " -> "
                << result.config_name
                << ", performance change: " << std::showpos << std::fixed
                << std::setprecision(1) << change_pct << "%" << std::noshowpos
                << std::endl;

      EXPECT_GT(change_pct, -20.0)
          << "Performance drop too large at configuration boundary";
    }

    prev_tflops = result.throughput_tflops;
    prev_config = result.config_name;
  }
}

TEST_F(CutlassScaledMMSM120ConfigTest, BoundaryPerformance_M256_LargeM) {
  PrintHeader(
      "Configuration boundary test: 128x64 -> wave-based (M=256 -> M=257)");
  PrintTableHeader();

  const int64_t N = 2048;
  const int64_t K = 2048;

  std::vector<int64_t> m_values = {254, 255, 256, 257, 258, 259, 260};

  double prev_tflops = 0;
  std::string prev_config;

  for (int64_t m : m_values) {
    auto result = BenchmarkWithDispatch(m, N, K, device_);
    PrintResult(m, result);

    if (!prev_config.empty() && result.config_name != prev_config) {
      double change_pct =
          (result.throughput_tflops - prev_tflops) / prev_tflops * 100.0;
      std::cout << "  >>> Configuration switch: " << prev_config << " -> "
                << result.config_name
                << ", performance change: " << std::showpos << std::fixed
                << std::setprecision(1) << change_pct << "%" << std::noshowpos
                << std::endl;

      // Wave efficiency based selection may cause config switch here.
      EXPECT_GT(change_pct, -30.0)
          << "Performance drop too large at configuration boundary";
    }

    prev_tflops = result.throughput_tflops;
    prev_config = result.config_name;
  }
}

TEST_F(CutlassScaledMMSM120ConfigTest, BoundaryPerformance_WaveEfficiency) {
  PrintHeader("Configuration boundary test: wave efficiency based (M > 256)");
  PrintTableHeader();

  const int64_t N = 2048;
  const int64_t K = 2048;

  std::vector<int64_t> m_values = {510, 511, 512, 513, 514, 515, 516};

  double prev_tflops = 0;
  std::string prev_config;

  for (int64_t m : m_values) {
    auto result = BenchmarkWithDispatch(m, N, K, device_);
    PrintResult(m, result);

    if (!prev_config.empty() && result.config_name != prev_config) {
      double change_pct =
          (result.throughput_tflops - prev_tflops) / prev_tflops * 100.0;
      std::cout << "  >>> Configuration switch: " << prev_config << " -> "
                << result.config_name
                << ", performance change: " << std::showpos << std::fixed
                << std::setprecision(1) << change_pct << "%" << std::noshowpos
                << std::endl;

      // Performance change at wave-based boundary depends on wave efficiency.
      // Allow up to 50% drop at this boundary.
      EXPECT_GT(change_pct, -50.0)
          << "Performance drop too large at configuration boundary";
    }

    prev_tflops = result.throughput_tflops;
    prev_config = result.config_name;
  }
}

// ============================================================================
// Full Grid Search Test
// ============================================================================

TEST_F(CutlassScaledMMSM120ConfigTest, FullGridSearch) {
  // Test shapes (N, K)
  std::vector<std::pair<int64_t, int64_t>> test_shapes = {
      {2048, 11008},
      {22016, 2048},
      {2048, 2048},
      {2560, 2048},
  };

  // M values: fine-grained range
  std::vector<int64_t> m_values;
  // 1-128: step 8
  for (int64_t m = 1; m <= 128; m += 8) {
    m_values.push_back(m);
  }
  // Key boundary points
  for (int64_t m : {16, 17, 64, 65, 128, 129, 256, 257}) {
    m_values.push_back(m);
  }
  // 128-4000: step 128
  for (int64_t m = 256; m <= 4000; m += 128) {
    m_values.push_back(m);
  }
  // Sort and remove duplicates
  std::sort(m_values.begin(), m_values.end());
  m_values.erase(std::unique(m_values.begin(), m_values.end()), m_values.end());

  std::cout
      << "\n================================================================\n";
  std::cout << "SM120 Full Grid Search Test\n";
  std::cout
      << "================================================================\n";
  std::cout << "Number of M values: " << m_values.size() << "\n";
  std::cout << "M value range: " << m_values.front() << " - " << m_values.back()
            << "\n";

  // Store results for analysis
  std::map<std::pair<int64_t, int64_t>, std::vector<ConfigResult>>
      results_by_shape;

  for (const auto& [n, k] : test_shapes) {
    std::cout << "\n----------------------------------------\n";
    std::cout << "N=" << n << ", K=" << k << "\n";
    std::cout << "----------------------------------------\n";

    std::vector<ConfigResult> shape_results;

    for (int64_t m : m_values) {
      auto result = BenchmarkWithDispatch(m, n, k, device_);
      shape_results.push_back(result);

      std::cout << "M=" << std::setw(5) << m << " | " << std::setw(14)
                << result.config_name << " | " << std::setw(12)
                << result.tile_shape << " | " << std::fixed
                << std::setprecision(3) << std::setw(8) << result.latency_ms
                << "ms | " << std::setprecision(2) << std::setw(7)
                << result.throughput_tflops << " TFLOPS\n";
    }

    results_by_shape[{n, k}] = shape_results;
  }

  // Analyze results
  std::cout
      << "\n================================================================\n";
  std::cout << "SM120 Configuration Boundary Analysis\n";
  std::cout
      << "================================================================\n";

  for (const auto& [nk, results] : results_by_shape) {
    auto [n, k] = nk;
    std::cout << "\nN=" << n << ", K=" << k << ":\n";

    std::string prev_config;
    for (size_t i = 0; i < results.size(); ++i) {
      if (i > 0 && results[i].config_name != prev_config) {
        double perf_before = results[i - 1].throughput_tflops;
        double perf_after = results[i].throughput_tflops;
        double change = (perf_after - perf_before) / perf_before * 100.0;

        std::cout << "  M=" << m_values[i] << ": " << prev_config << " ("
                  << results[i - 1].tile_shape << ") -> "
                  << results[i].config_name << " (" << results[i].tile_shape
                  << "), performance change: " << std::showpos << std::fixed
                  << std::setprecision(1) << change << "%\n"
                  << std::noshowpos;
      }
      prev_config = results[i].config_name;
    }
  }
}

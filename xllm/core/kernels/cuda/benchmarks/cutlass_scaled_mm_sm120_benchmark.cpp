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
 * SM120 (Blackwell) FP8 GEMM Configuration Grid Search Benchmark
 *
 * This benchmark tests optimal CUTLASS GEMM configurations for different M, N,
 * K shapes.
 *
 * Test shapes:
 * - N=2048, K=11008 (typical LLM down_proj shape)
 * - N=22016, K=2048 (typical LLM up_proj/gate_proj shape)
 * - N=2048, K=2048  (square matrix)
 * - N=2560, K=2048  (typical attention out_proj)
 * - M range: 1 to 4000 (corresponding to batch_size * seq_len)
 *
 * SM120 (Blackwell) features:
 * - 4th generation Tensor Core
 * - FP8 (E4M3/E5M2) matrix multiplication support
 * - Larger shared memory (228KB/SM)
 * - TMA (Tensor Memory Accelerator)
 * - ClusterShape limitation: currently only supports 1x1x1
 *
 * Build:
 *   In xllm project, build with cmake and run:
 *   ./cutlass_scaled_mm_sm120_benchmark
 *
 * Usage:
 *   ./cutlass_scaled_mm_sm120_benchmark [--warmup N] [--iters N] [--quick]
 *   ./cutlass_scaled_mm_sm120_benchmark --help
 */

#include <ATen/cuda/CUDAContext.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "cuda_ops_api.h"
#include "cutlass_extensions/common.hpp"
#include "cutlass_w8a8/c3x/sm120_fp8_dispatch_policy.hpp"

namespace {

// Test shapes (N, K) - typical LLM shapes
const std::vector<std::pair<int64_t, int64_t>> kTestShapes = {
    {2048, 11008},  // down_proj: hidden_size -> intermediate_size
    {22016, 2048},  // up_proj/gate_proj: intermediate_size -> hidden_size
    {2048, 2048},   // square matrix
    {2560, 2048},   // attention out_proj (hidden_size=2560)
};

// M values for quick mode - covering key boundary points
const std::vector<int64_t> kQuickMValues = {
    1,    2,    4,    8,    12,   16,  // 128x32_swap range
    17,   24,   32,   48,   64,        // 128x64/128x64_swap range
    65,   80,   96,   112,  128,       // 128x64 range
    129,  160,  192,  224,  256,       // 128x64 range
    257,  384,  512,                   // 128x64/128x128 range (wave-based)
    513,  640,  768,  896,  1024,      // 128x64/128x128 range (wave-based)
    1025, 1280, 1536, 1792, 2048,      // 128x64/128x128 range (wave-based)
    2049, 2560, 3072, 3584, 4000       // 128x64/128x128 range (wave-based)
};

struct BenchmarkConfig {
  int warmup_iters = 10;
  int bench_iters = 100;
  bool quick_mode = false;
  int m_start = 1;
  int m_end = 4000;
  int m_step = 32;
  bool save_results = true;
  std::string output_dir = "./benchmark_results";
};

struct BenchmarkResult {
  int64_t m;
  int64_t n;
  int64_t k;
  xllm::kernel::cuda::SM120DispatchKernel kernel;
  std::string config_name;
  double latency_ms;
  double throughput_tflops;
  double memory_bandwidth_gb;
  float wave_efficiency;  // Wave efficiency (0-1)
};

// Compute TFLOPS
double ComputeTflops(int64_t m, int64_t n, int64_t k, double latency_s) {
  double flops = 2.0 * m * n * k;
  return flops / latency_s / 1e12;
}

// Compute memory bandwidth (GB/s)
double ComputeMemoryBandwidth(int64_t m,
                              int64_t n,
                              int64_t k,
                              double latency_s) {
  // Input: A (M*K) + B (K*N), FP8 = 1 byte
  // Output: C (M*N), FP16/BF16 = 2 bytes
  // Scales: a_scales (1) + b_scales (1), FP32 = 4 bytes
  double bytes_read = static_cast<double>(m * k + k * n) + 8;  // FP8 + scales
  double bytes_written = static_cast<double>(m * n * 2);       // FP16
  double total_bytes = bytes_read + bytes_written;
  return total_bytes / latency_s / 1e9;
}

// Run a single benchmark
BenchmarkResult RunBenchmark(int64_t m,
                             int64_t n,
                             int64_t k,
                             int warmup_iters,
                             int bench_iters,
                             torch::Device device,
                             int num_sms = 0) {
  BenchmarkResult result;
  result.m = m;
  result.n = n;
  result.k = k;

  // Get SM count if not provided
  if (num_sms == 0) {
    num_sms = static_cast<int>(
        xllm::kernel::cuda::get_sm120_num_sms_for_device(device.index()));
  }

  auto dispatch = xllm::kernel::cuda::select_sm120_dispatch(m, n, k, num_sms);
  result.kernel = dispatch.kernel;
  result.config_name =
      xllm::kernel::cuda::get_sm120_dispatch_kernel_name(dispatch.kernel);
  result.wave_efficiency = xllm::kernel::cuda::compute_sm120_wave_efficiency(
      m, n, dispatch.tile_m, dispatch.tile_n, num_sms);

  auto fp32_options =
      torch::TensorOptions().device(device).dtype(torch::kFloat32);
  auto fp16_options =
      torch::TensorOptions().device(device).dtype(torch::kFloat16);

  // Create input matrices (use smaller values to avoid FP8 overflow)
  torch::Tensor a_fp32 = torch::randn({m, k}, fp32_options) * 0.5f;
  torch::Tensor b_fp32 = torch::randn({k, n}, fp32_options) * 0.5f;

  // Convert to FP8
  torch::Tensor a = a_fp32.to(torch::kFloat8_e4m3fn);
  torch::Tensor b = b_fp32.to(torch::kFloat8_e4m3fn);

  // Transpose B to column-major (required by CUTLASS)
  b = b.t().contiguous().t();

  // Create scales
  torch::Tensor a_scales = torch::ones({1}, fp32_options);
  torch::Tensor b_scales = torch::ones({1}, fp32_options);

  // Create output
  torch::Tensor c = torch::zeros({m, n}, fp16_options);

  // Warmup
  for (int i = 0; i < warmup_iters; ++i) {
    xllm::kernel::cuda::cutlass_scaled_mm(
        c, a, b, a_scales, b_scales, std::nullopt);
  }
  torch::cuda::synchronize();

  // Benchmark
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < bench_iters; ++i) {
    xllm::kernel::cuda::cutlass_scaled_mm(
        c, a, b, a_scales, b_scales, std::nullopt);
  }
  torch::cuda::synchronize();
  auto end = std::chrono::high_resolution_clock::now();

  double total_time_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  result.latency_ms = total_time_ms / bench_iters;
  double latency_s = result.latency_ms / 1000.0;

  result.throughput_tflops = ComputeTflops(m, n, k, latency_s);
  result.memory_bandwidth_gb = ComputeMemoryBandwidth(m, n, k, latency_s);

  return result;
}

// Analyze configuration boundaries
void AnalyzeConfigBoundaries(const std::vector<BenchmarkResult>& results,
                             int64_t n,
                             int64_t k) {
  if (results.empty()) return;

  std::cout << "\nConfiguration boundary analysis (N=" << n << ", K=" << k
            << "):" << std::endl;

  std::string prev_config;
  for (size_t i = 0; i < results.size(); ++i) {
    if (i > 0 && results[i].config_name != prev_config) {
      double perf_before = results[i - 1].throughput_tflops;
      double perf_after = results[i].throughput_tflops;
      double change_pct = (perf_after - perf_before) / perf_before * 100.0;

      std::cout << "  M=" << results[i].m << ": " << prev_config << " ("
                << xllm::kernel::cuda::get_sm120_dispatch_tile_shape_desc(
                       results[i - 1].kernel)
                << ") -> " << results[i].config_name << " ("
                << xllm::kernel::cuda::get_sm120_dispatch_tile_shape_desc(
                       results[i].kernel)
                << ")"
                << ", performance change: " << std::showpos << std::fixed
                << std::setprecision(1) << change_pct << "%" << std::noshowpos
                << std::endl;
    }
    prev_config = results[i].config_name;
  }
}

// Generate summary report
void GenerateSummary(
    const std::map<std::pair<int64_t, int64_t>, std::vector<BenchmarkResult>>&
        all_results) {
  std::cout << "\n" << std::string(80, '=') << std::endl;
  std::cout << "SM120 FP8 GEMM Performance Summary" << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  for (const auto& [shape, results] : all_results) {
    if (results.empty()) continue;

    auto [n, k] = shape;
    std::cout << "\nN=" << n << ", K=" << k << ":" << std::endl;

    // Group by configuration
    std::map<std::string, std::vector<const BenchmarkResult*>> config_results;
    for (const auto& r : results) {
      config_results[r.config_name].push_back(&r);
    }

    for (const auto& [config_name, config_res] : config_results) {
      int64_t m_min = INT64_MAX, m_max = 0;
      double total_tflops = 0;
      double max_tflops = 0;
      double min_tflops = std::numeric_limits<double>::max();

      for (const auto* r : config_res) {
        m_min = std::min(m_min, r->m);
        m_max = std::max(m_max, r->m);
        total_tflops += r->throughput_tflops;
        max_tflops = std::max(max_tflops, r->throughput_tflops);
        min_tflops = std::min(min_tflops, r->throughput_tflops);
      }

      double avg_tflops = total_tflops / config_res.size();
      auto kernel = config_res.front()->kernel;
      std::string tile_desc =
          xllm::kernel::cuda::get_sm120_dispatch_tile_shape_desc(kernel);
      bool swap_ab = xllm::kernel::cuda::is_sm120_swap_ab_kernel(kernel);

      std::cout << "  " << std::setw(15) << std::left << config_name
                << " (tile=" << std::setw(12) << tile_desc
                << ", swap=" << (swap_ab ? "Y" : "N") << ")"
                << ": M=[" << std::setw(4) << std::right << m_min << ", "
                << std::setw(4) << m_max << "]"
                << ", TFLOPS: avg=" << std::fixed << std::setprecision(2)
                << std::setw(6) << avg_tflops << ", max=" << std::setw(6)
                << max_tflops << ", min=" << std::setw(6) << min_tflops
                << std::endl;
    }
  }
}

// Generate optimization recommendations
void GenerateRecommendations(
    const std::map<std::pair<int64_t, int64_t>, std::vector<BenchmarkResult>>&
        all_results) {
  std::cout << "\n" << std::string(80, '=') << std::endl;
  std::cout << "SM120 FP8 GEMM Optimization Recommendations" << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  std::vector<std::string> recommendations;
  std::vector<std::string> issues;

  for (const auto& [shape, results] : all_results) {
    if (results.size() < 2) continue;

    auto [n, k] = shape;

    // Analyze performance for each configuration range
    std::map<std::string, std::pair<double, double>> config_perf_range;
    for (const auto& r : results) {
      auto& range = config_perf_range[r.config_name];
      if (range.first == 0 || r.throughput_tflops < range.first) {
        range.first = r.throughput_tflops;  // min
      }
      if (r.throughput_tflops > range.second) {
        range.second = r.throughput_tflops;  // max
      }
    }

    // Check performance jumps at configuration boundaries
    for (size_t i = 1; i < results.size(); ++i) {
      const auto& prev = results[i - 1];
      const auto& curr = results[i];

      // Check for significant performance drop when M increases (>15%)
      if (curr.m > prev.m &&
          curr.throughput_tflops < prev.throughput_tflops * 0.85) {
        if (prev.config_name != curr.config_name) {
          std::ostringstream oss;
          oss << "N=" << n << ", K=" << k << ": At M=" << curr.m
              << ", configuration switch from " << prev.config_name << " to "
              << curr.config_name << " causes performance drop (" << std::fixed
              << std::setprecision(2) << prev.throughput_tflops << " -> "
              << curr.throughput_tflops << " TFLOPS, -" << std::setprecision(1)
              << ((prev.throughput_tflops - curr.throughput_tflops) /
                  prev.throughput_tflops * 100)
              << "%)";
          issues.push_back(oss.str());
        }
      }
    }

    // Check if swap_ab strategy is effective
    for (const auto& r : results) {
      if (xllm::kernel::cuda::is_sm120_swap_ab_kernel(r.kernel)) {
        // For swap_ab configurations, check if they are better than
        // non-swap_ab versions. This needs to be verified in actual tests.
      }
    }
  }

  // Output discovered issues
  if (!issues.empty()) {
    std::cout << "\nDiscovered performance issues:" << std::endl;
    for (size_t i = 0; i < issues.size(); ++i) {
      std::cout << "  " << (i + 1) << ". " << issues[i] << std::endl;
    }
  }

  // General optimization recommendations
  std::cout << "\nGeneral optimization recommendations:" << std::endl;
  std::cout << "  1. For very small batch scenarios (M <= 16), swap_ab "
               "technique is usually effective"
            << std::endl;
  std::cout << "  2. When K >= 4096, medium M (17-64) can also consider "
               "swap_ab"
            << std::endl;
  std::cout << "  3. Large M (>256) scenarios should use large TileShape "
               "(128x128x128)"
            << std::endl;
  std::cout << "  4. SM120's ClusterShape is limited to 1x1x1, cannot use "
               "cluster optimization"
            << std::endl;

  if (issues.empty()) {
    std::cout << "\nCurrent configuration boundaries look reasonable, no "
                 "obvious performance regression issues found."
              << std::endl;
  } else {
    std::cout << "\nRecommend adjusting configuration boundary thresholds "
                 "based on the above issues."
              << std::endl;
  }
}

// Save results to CSV
void SaveResultsToCSV(const std::map<std::pair<int64_t, int64_t>,
                                     std::vector<BenchmarkResult>>& all_results,
                      const std::string& output_dir) {
  for (const auto& [shape, results] : all_results) {
    if (results.empty()) continue;

    auto [n, k] = shape;
    std::ostringstream filename;
    filename << output_dir << "/sm120_fp8_gemm_N" << n << "_K" << k << ".csv";

    std::ofstream file(filename.str());
    if (!file.is_open()) {
      LOG(WARNING) << "Cannot open file: " << filename.str();
      continue;
    }

    // CSV header
    file << "M,N,K,Config,TileShape,SwapAB,Latency_ms,TFLOPS,BW_GB_s,WaveEff\n";

    for (const auto& r : results) {
      file << r.m << "," << r.n << "," << r.k << "," << r.config_name << ","
           << xllm::kernel::cuda::get_sm120_dispatch_tile_shape_desc(r.kernel)
           << ","
           << (xllm::kernel::cuda::is_sm120_swap_ab_kernel(r.kernel) ? "true"
                                                                     : "false")
           << "," << std::fixed << std::setprecision(4) << r.latency_ms << ","
           << std::setprecision(2) << r.throughput_tflops << ","
           << r.memory_bandwidth_gb << "," << std::setprecision(3)
           << r.wave_efficiency << "\n";
    }

    file.close();
    std::cout << "Results saved to: " << filename.str() << std::endl;
  }
}

// Generate JSON analysis report
void SaveAnalysisToJSON(
    const std::map<std::pair<int64_t, int64_t>, std::vector<BenchmarkResult>>&
        all_results,
    const std::string& output_dir,
    int compute_capability) {
  std::ostringstream filename;
  filename << output_dir << "/sm120_analysis.json";

  std::ofstream file(filename.str());
  if (!file.is_open()) {
    LOG(WARNING) << "Cannot open file: " << filename.str();
    return;
  }

  file << "{\n";
  file << "  \"architecture\": \"SM120 (Blackwell)\",\n";
  file << "  \"compute_capability\": " << compute_capability << ",\n";
  file << "  \"configurations\": {\n";

  bool first_config = true;
  for (const auto& kernel : xllm::kernel::cuda::kSM120DispatchKernels) {
    if (!first_config) file << ",\n";
    first_config = false;
    const char* config_name =
        xllm::kernel::cuda::get_sm120_dispatch_kernel_name(kernel);

    file << "    \"" << config_name << "\": {\n";
    file << "      \"tile_shape\": \""
         << xllm::kernel::cuda::get_sm120_dispatch_tile_shape_desc(kernel)
         << "\",\n";
    file << "      \"cluster_shape\": \"1x1x1\",\n";
    file << "      \"swap_ab\": "
         << (xllm::kernel::cuda::is_sm120_swap_ab_kernel(kernel) ? "true"
                                                                 : "false")
         << "\n";
    file << "    }";
  }

  file << "\n  },\n";
  file << "  \"test_shapes\": [\n";

  bool first_shape = true;
  for (const auto& [shape, results] : all_results) {
    if (results.empty()) continue;

    if (!first_shape) file << ",\n";
    first_shape = false;

    auto [n, k] = shape;
    file << "    {\n";
    file << "      \"N\": " << n << ",\n";
    file << "      \"K\": " << k << ",\n";
    file << "      \"best_performance\": {\n";

    // Find best performance
    const BenchmarkResult* best = nullptr;
    for (const auto& r : results) {
      if (!best || r.throughput_tflops > best->throughput_tflops) {
        best = &r;
      }
    }

    if (best) {
      file << "        \"M\": " << best->m << ",\n";
      file << "        \"config\": \"" << best->config_name << "\",\n";
      file << "        \"TFLOPS\": " << std::fixed << std::setprecision(2)
           << best->throughput_tflops << "\n";
    }

    file << "      },\n";
    file << "      \"config_boundaries\": [";

    // Record configuration boundaries
    std::string prev_config;
    bool first_boundary = true;
    for (const auto& r : results) {
      if (!prev_config.empty() && r.config_name != prev_config) {
        if (!first_boundary) file << ", ";
        first_boundary = false;
        file << r.m;
      }
      prev_config = r.config_name;
    }

    file << "]\n";
    file << "    }";
  }

  file << "\n  ]\n";
  file << "}\n";

  file.close();
  std::cout << "Analysis report saved to: " << filename.str() << std::endl;
}

void PrintUsage(const char* program_name) {
  std::cout << "SM120 (Blackwell) FP8 GEMM Grid Search Benchmark\n\n";
  std::cout
      << "Usage: " << program_name << " [options]\n"
      << "\nOptions:\n"
      << "  --warmup N     Number of warmup iterations (default: 10)\n"
      << "  --iters N      Number of benchmark iterations (default: 100)\n"
      << "  --quick        Quick mode: only test key boundary points\n"
      << "  --m-start N    M value start (default: 1)\n"
      << "  --m-end N      M value end (default: 4000)\n"
      << "  --m-step N     M value step (default: 32)\n"
      << "  --no-save      Do not save results to file\n"
      << "  --output DIR   Output directory (default: ./benchmark_results)\n"
      << "  --help         Show help information\n"
      << std::endl;
}

BenchmarkConfig ParseArgs(int argc, char* argv[]) {
  BenchmarkConfig config;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--warmup" && i + 1 < argc) {
      config.warmup_iters = std::stoi(argv[++i]);
    } else if (arg == "--iters" && i + 1 < argc) {
      config.bench_iters = std::stoi(argv[++i]);
    } else if (arg == "--quick") {
      config.quick_mode = true;
    } else if (arg == "--m-start" && i + 1 < argc) {
      config.m_start = std::stoi(argv[++i]);
    } else if (arg == "--m-end" && i + 1 < argc) {
      config.m_end = std::stoi(argv[++i]);
    } else if (arg == "--m-step" && i + 1 < argc) {
      config.m_step = std::stoi(argv[++i]);
    } else if (arg == "--no-save") {
      config.save_results = false;
    } else if (arg == "--output" && i + 1 < argc) {
      config.output_dir = argv[++i];
    } else if (arg == "--help") {
      PrintUsage(argv[0]);
      exit(0);
    }
  }

  return config;
}

}  // namespace

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;

  BenchmarkConfig config = ParseArgs(argc, argv);

  // Check CUDA availability
  if (!torch::cuda::is_available()) {
    LOG(ERROR) << "CUDA not available";
    return 1;
  }

  int current_device = 0;
  cudaError_t err = cudaGetDevice(&current_device);
  TORCH_CHECK(err == cudaSuccess,
              "Failed to get current CUDA device: ",
              cudaGetErrorString(err));
  torch::Device device(torch::kCUDA, current_device);

  // Check compute capability, SM120 benchmark only runs on SM120 architecture
  int compute_capability = xllm::kernel::cuda::get_sm_version_num();

  // SM120 requires compute capability >= 12.0 (Blackwell)
  if (compute_capability < 120) {
    LOG(ERROR) << "SM120 benchmark requires compute capability >= 12.0 "
                  "(Blackwell), current: "
               << compute_capability
               << ". This test can only run on SM120 architecture.";
    return 1;
  }

  // Prepare M value list
  std::vector<int64_t> m_values;
  if (config.quick_mode) {
    m_values = kQuickMValues;
  } else {
    for (int m = config.m_start; m <= config.m_end; m += config.m_step) {
      m_values.push_back(m);
    }
    // Add key boundary points
    std::vector<int64_t> key_points = {1,   8,   16,   17,   32,   64,   65,
                                       96,  128, 129,  192,  256,  257,  384,
                                       512, 768, 1024, 1536, 2048, 3072, 4000};
    for (int64_t kp : key_points) {
      if (kp >= config.m_start && kp <= config.m_end) {
        m_values.push_back(kp);
      }
    }
    // Sort and remove duplicates
    std::sort(m_values.begin(), m_values.end());
    m_values.erase(std::unique(m_values.begin(), m_values.end()),
                   m_values.end());
  }

  // Print configuration information
  std::cout << std::string(80, '=') << std::endl;
  std::cout << "SM120 (Blackwell) FP8 GEMM Grid Search Benchmark" << std::endl;
  std::cout << std::string(80, '=') << std::endl;
  std::cout << "GPU: " << at::cuda::getDeviceProperties(current_device)->name
            << std::endl;
  std::cout << "Compute Capability: " << compute_capability << std::endl;
  std::cout << "Warmup Iterations: " << config.warmup_iters << std::endl;
  std::cout << "Benchmark Iterations: " << config.bench_iters << std::endl;
  std::cout << "Number of M values: " << m_values.size() << std::endl;
  std::cout << "M value range: " << m_values.front() << " - " << m_values.back()
            << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  std::cout << "\nSM120 Configuration Strategy:" << std::endl;
  for (const auto& kernel : xllm::kernel::cuda::kSM120DispatchKernels) {
    const char* name =
        xllm::kernel::cuda::get_sm120_dispatch_kernel_name(kernel);
    std::cout << "  " << std::setw(15) << std::left << name
              << ": TileShape=" << std::setw(12)
              << xllm::kernel::cuda::get_sm120_dispatch_tile_shape_desc(kernel)
              << ", ClusterShape=1x1x1"
              << ", swap_ab="
              << (xllm::kernel::cuda::is_sm120_swap_ab_kernel(kernel) ? "true"
                                                                      : "false")
              << std::endl;
  }
  std::cout << std::string(80, '=') << std::endl;

  // Store all results
  std::map<std::pair<int64_t, int64_t>, std::vector<BenchmarkResult>>
      all_results;

  // Run tests
  for (const auto& [n, k] : kTestShapes) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test shape: N=" << n << ", K=" << k << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    std::vector<BenchmarkResult> shape_results;

    for (int64_t m : m_values) {
      if (m <= 0) continue;

      try {
        BenchmarkResult result = RunBenchmark(
            m, n, k, config.warmup_iters, config.bench_iters, device);
        shape_results.push_back(result);

        std::cout << "M=" << std::setw(5) << m << ", Config=" << std::setw(14)
                  << std::left << result.config_name << std::right << " ("
                  << std::setw(12)
                  << xllm::kernel::cuda::get_sm120_dispatch_tile_shape_desc(
                         result.kernel)
                  << ")"
                  << ", Latency=" << std::setw(8) << std::fixed
                  << std::setprecision(3) << result.latency_ms << "ms"
                  << ", TFLOPS=" << std::setw(7) << std::setprecision(2)
                  << result.throughput_tflops << ", BW=" << std::setw(8)
                  << result.memory_bandwidth_gb << "GB/s"
                  << ", WaveEff=" << std::setprecision(3)
                  << result.wave_efficiency << std::endl;
      } catch (const std::exception& e) {
        LOG(WARNING) << "M=" << m << ": Error - " << e.what();
      }
    }

    // Analyze configuration boundaries
    AnalyzeConfigBoundaries(shape_results, n, k);

    all_results[{n, k}] = std::move(shape_results);
  }

  // Generate summary and recommendations
  GenerateSummary(all_results);
  GenerateRecommendations(all_results);

  // Save results
  if (config.save_results) {
    // Create output directory (if it doesn't exist)
    std::error_code ec;
    std::filesystem::create_directories(config.output_dir, ec);
    if (ec) {
      LOG(WARNING) << "Failed to create output directory: " << config.output_dir
                   << ", error: " << ec.message();
    }

    SaveResultsToCSV(all_results, config.output_dir);
    SaveAnalysisToJSON(all_results, config.output_dir, compute_capability);
  }

  return 0;
}

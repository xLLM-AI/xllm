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

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include <string>
#include <utility>
#include <vector>

#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

namespace xllm::kernel::npu::tilelang {
namespace {

class TileLangFusedGdnGatingWrapperTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { torch_npu::init_npu("npu:0"); }

  static void TearDownTestSuite() { torch_npu::finalize_npu(); }
};

struct FusedGdnGatingTestCase {
  std::string name;
  int64_t num_batches;
  int64_t num_heads;
  int64_t seed;
  float beta = 1.0F;
  float threshold = 20.0F;
};

std::pair<torch::Tensor, torch::Tensor> torch_fused_gdn_gating(
    const torch::Tensor& A_log,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& dt_bias,
    float beta = 1.0F,
    float threshold = 20.0F) {
  namespace F = torch::nn::functional;

  auto softplus_out =
      F::softplus(a.to(torch::kFloat32) + dt_bias,
                  F::SoftplusFuncOptions().beta(beta).threshold(threshold));
  auto g = -A_log.exp() * softplus_out;
  auto beta_output = torch::sigmoid(b.to(torch::kFloat32)).to(torch::kBFloat16);
  return {g.unsqueeze(0), beta_output.unsqueeze(0)};
}

void run_fused_gdn_gating_case(const FusedGdnGatingTestCase& test_case) {
  ASSERT_GT(test_case.num_batches, 0);

  const auto device = torch::Device("npu:0");
  torch::manual_seed(test_case.seed);

  auto fp32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto bf16_opts =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);

  auto A_log = torch::randn({test_case.num_heads}, fp32_opts);
  auto a =
      torch::randn({test_case.num_batches, test_case.num_heads}, bf16_opts);
  auto b =
      torch::randn({test_case.num_batches, test_case.num_heads}, bf16_opts);
  auto dt_bias = torch::randn({test_case.num_heads}, fp32_opts);

  auto [g_ref, beta_ref] = torch_fused_gdn_gating(
      A_log, a, b, dt_bias, test_case.beta, test_case.threshold);
  auto [g_out, beta_out] = fused_gdn_gating(
      A_log, a, b, dt_bias, test_case.beta, test_case.threshold);

  auto g_max_diff = (g_out - g_ref).abs().max().item<float>();
  int64_t g_max_index = (g_out - g_ref).abs().argmax().item<int64_t>();
  const int64_t g_max_head = g_max_index % test_case.num_heads;
  const int64_t g_max_row =
      (g_max_index / test_case.num_heads) % test_case.num_batches;
  auto beta_max_diff =
      (beta_out.to(torch::kFloat32) - beta_ref.to(torch::kFloat32))
          .abs()
          .max()
          .item<float>();
  int64_t beta_max_index =
      (beta_out.to(torch::kFloat32) - beta_ref.to(torch::kFloat32))
          .abs()
          .argmax()
          .item<int64_t>();
  const int64_t beta_max_head = beta_max_index % test_case.num_heads;
  const int64_t beta_max_row =
      (beta_max_index / test_case.num_heads) % test_case.num_batches;

  EXPECT_TRUE(torch::allclose(g_out, g_ref, 1e-3, 1e-3))
      << "g mismatch, max_diff=" << g_max_diff << ", row=" << g_max_row
      << ", head=" << g_max_head;
  EXPECT_TRUE(torch::allclose(beta_out, beta_ref, 1e-2, 1e-2))
      << "beta mismatch, max_diff=" << beta_max_diff << ", row=" << beta_max_row
      << ", head=" << beta_max_head;
}

TEST_F(TileLangFusedGdnGatingWrapperTest, MatchesTorchReference) {
  const std::vector<FusedGdnGatingTestCase> cases = {
      {
          .name = "tiny_b1_h8",
          .num_batches = 1,
          .num_heads = 8,
          .seed = 101,
      },
      {
          .name = "tiny_b17_h8",
          .num_batches = 17,
          .num_heads = 8,
          .seed = 101,
      },
      {
          .name = "tiny_b1_h16",
          .num_batches = 1,
          .num_heads = 16,
          .seed = 101,
      },
      {
          .name = "qwen35_tp2_image_b275_h24",
          .num_batches = 275,
          .num_heads = 24,
          .seed = 201,
      },
      {
          .name = "qwen35_tp2_video_b3716_h24",
          .num_batches = 3716,
          .num_heads = 24,
          .seed = 202,
      },
      {
          .name = "qwen35_tp2_video_b4096_h24",
          .num_batches = 4096,
          .num_heads = 24,
          .seed = 203,
      },
      {
          .name = "qwen35_tp2_video_b8192_h24",
          .num_batches = 8192,
          .num_heads = 24,
          .seed = 204,
      },
      {
          .name = "small_b17_h32",
          .num_batches = 17,
          .num_heads = 32,
          .seed = 102,
      },
      {
          .name = "medium_b29_h48",
          .num_batches = 29,
          .num_heads = 48,
          .seed = 103,
      },
      {
          .name = "medium_b131_h64",
          .num_batches = 131,
          .num_heads = 64,
          .seed = 104,
      },
      {
          .name = "medium_b257_h128",
          .num_batches = 257,
          .num_heads = 128,
          .seed = 105,
      },
      {
          .name = "large_b4096_h32",
          .num_batches = 4096,
          .num_heads = 32,
          .seed = 106,
      },
      {
          .name = "chunked_b8192_h32",
          .num_batches = 8192,
          .num_heads = 32,
          .seed = 108,
      },
      {
          .name = "custom_beta2_threshold0p5_b33_h64",
          .num_batches = 33,
          .num_heads = 64,
          .seed = 107,
          .beta = 2.0F,
          .threshold = 0.5F,
      },
      {
          .name = "large_b65536_h32",
          .num_batches = 65536,
          .num_heads = 32,
          .seed = 109,
      },
      {
          .name = "large_b262144_h32",
          .num_batches = 262144,
          .num_heads = 32,
          .seed = 110,
      },
  };

  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    run_fused_gdn_gating_case(test_case);
  }
}

}  // namespace
}  // namespace xllm::kernel::npu::tilelang

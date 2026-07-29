/* Copyright 2025-2026 The xLLM Authors.

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
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/torch_npu.h>

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

namespace xllm::kernel::npu::tilelang {
namespace {

constexpr int64_t kHeadSize = 256;
constexpr double kRmsNormEps = 1e-6;

class TileLangSplitQkvRmsnormMRopeWrapperTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { torch_npu::init_npu("npu:0"); }

  static void TearDownTestSuite() {
    torch_npu::finalize_npu();
    aclrtResetDevice(0);
    aclFinalize();
  }
};

struct SplitQkvTestCase {
  std::string name;
  int64_t num_tokens;
  int64_t num_q_heads;
  int64_t num_kv_heads;
  int64_t seed;
  bool is_interleaved;
};

torch::Tensor rms_norm_ref(const torch::Tensor& x,
                           const torch::Tensor& weight,
                           double eps) {
  torch::Tensor x_fp32 = x.to(torch::kFloat32);
  torch::Tensor weight_fp32 = weight.to(torch::kFloat32);
  torch::Tensor reciprocal_std =
      torch::rsqrt(torch::mean(x_fp32 * x_fp32, /*dim=*/-1, true) + eps);
  return x_fp32 * reciprocal_std * weight_fp32;
}

std::pair<torch::Tensor, torch::Tensor> assemble_non_interleaved_mrope_rows_ref(
    const torch::Tensor& cos_sin,
    const std::vector<int64_t>& mrope_section) {
  const int64_t half_rope_dim = cos_sin.size(2) / 2;
  const int64_t t_len = mrope_section[0];
  const int64_t h_len = mrope_section[1];
  const int64_t w_len = mrope_section[2];
  const int64_t h_end = t_len + h_len;
  const int64_t w_end = h_end + w_len;

  torch::Tensor cos_axes =
      cos_sin.slice(/*dim=*/2, /*start=*/0, /*end=*/half_rope_dim)
          .to(torch::kFloat32);
  torch::Tensor sin_axes =
      cos_sin.slice(/*dim=*/2, /*start=*/half_rope_dim, /*end=*/cos_sin.size(2))
          .to(torch::kFloat32);

  torch::Tensor cos_rows =
      torch::zeros({cos_sin.size(1), half_rope_dim},
                   cos_sin.options().dtype(torch::kFloat32));
  torch::Tensor sin_rows =
      torch::zeros({cos_sin.size(1), half_rope_dim},
                   cos_sin.options().dtype(torch::kFloat32));

  if (t_len > 0) {
    cos_rows.slice(/*dim=*/1, /*start=*/0, /*end=*/t_len)
        .copy_(cos_axes.select(/*dim=*/0, /*index=*/0)
                   .slice(/*dim=*/1, /*start=*/0, /*end=*/t_len));
    sin_rows.slice(/*dim=*/1, /*start=*/0, /*end=*/t_len)
        .copy_(sin_axes.select(/*dim=*/0, /*index=*/0)
                   .slice(/*dim=*/1, /*start=*/0, /*end=*/t_len));
  }
  if (h_len > 0) {
    cos_rows.slice(/*dim=*/1, /*start=*/t_len, /*end=*/h_end)
        .copy_(cos_axes.select(/*dim=*/0, /*index=*/1)
                   .slice(/*dim=*/1, /*start=*/t_len, /*end=*/h_end));
    sin_rows.slice(/*dim=*/1, /*start=*/t_len, /*end=*/h_end)
        .copy_(sin_axes.select(/*dim=*/0, /*index=*/1)
                   .slice(/*dim=*/1, /*start=*/t_len, /*end=*/h_end));
  }
  if (w_len > 0) {
    cos_rows.slice(/*dim=*/1, /*start=*/h_end, /*end=*/w_end)
        .copy_(cos_axes.select(/*dim=*/0, /*index=*/2)
                   .slice(/*dim=*/1, /*start=*/h_end, /*end=*/w_end));
    sin_rows.slice(/*dim=*/1, /*start=*/h_end, /*end=*/w_end)
        .copy_(sin_axes.select(/*dim=*/0, /*index=*/2)
                   .slice(/*dim=*/1, /*start=*/h_end, /*end=*/w_end));
  }

  return {cos_rows, sin_rows};
}

std::pair<torch::Tensor, torch::Tensor> assemble_interleaved_mrope_rows_ref(
    const torch::Tensor& cos_sin,
    const std::vector<int64_t>& mrope_section) {
  const int64_t half_rope_dim = cos_sin.size(2) / 2;
  const int64_t h_len = mrope_section[1];
  const int64_t w_len = mrope_section[2];

  torch::Tensor cos_axes =
      cos_sin.slice(/*dim=*/2, /*start=*/0, /*end=*/half_rope_dim)
          .to(torch::kFloat32);
  torch::Tensor sin_axes =
      cos_sin.slice(/*dim=*/2, /*start=*/half_rope_dim, /*end=*/cos_sin.size(2))
          .to(torch::kFloat32);

  torch::Tensor cos_rows = cos_axes.select(/*dim=*/0, /*index=*/0).clone();
  torch::Tensor sin_rows = sin_axes.select(/*dim=*/0, /*index=*/0).clone();

  for (int64_t i = 0; i < half_rope_dim; ++i) {
    int64_t axis_id = 0;
    if ((i % 3) == 1 && i < h_len * 3) {
      axis_id = 1;
    } else if ((i % 3) == 2 && i < w_len * 3) {
      axis_id = 2;
    }
    cos_rows.slice(/*dim=*/1, /*start=*/i, /*end=*/i + 1)
        .copy_(cos_axes.select(/*dim=*/0, /*index=*/axis_id)
                   .slice(/*dim=*/1, /*start=*/i, /*end=*/i + 1));
    sin_rows.slice(/*dim=*/1, /*start=*/i, /*end=*/i + 1)
        .copy_(sin_axes.select(/*dim=*/0, /*index=*/axis_id)
                   .slice(/*dim=*/1, /*start=*/i, /*end=*/i + 1));
  }

  return {cos_rows, sin_rows};
}

torch::Tensor merge_cos_sin_for_wrapper(const torch::Tensor& cos_sin) {
  return cos_sin.permute({1, 0, 2}).contiguous().view(
      {cos_sin.size(1), cos_sin.size(0) * cos_sin.size(2)});
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
torch_split_qkv_rmsnorm_mrope(const torch::Tensor& qkvg,
                              const torch::Tensor& q_weight,
                              const torch::Tensor& k_weight,
                              const torch::Tensor& cos_sin,
                              int64_t num_q_heads,
                              int64_t num_kv_heads,
                              int64_t head_size,
                              double eps,
                              const std::vector<int64_t>& mrope_section,
                              bool is_interleaved) {
  CHECK_EQ(mrope_section.size(), 3) << "mrope_section must contain [t, h, w]";

  const int64_t q_size = num_q_heads * head_size;
  const int64_t kv_size = num_kv_heads * head_size;
  torch::Tensor q = qkvg.slice(/*dim=*/1, /*start=*/0, /*end=*/q_size)
                        .view({-1, num_q_heads, head_size});
  torch::Tensor gate = qkvg.slice(/*dim=*/1,
                                  /*start=*/q_size,
                                  /*end=*/q_size * 2)
                           .view({-1, num_q_heads, head_size});
  torch::Tensor k =
      qkvg.slice(/*dim=*/1, /*start=*/q_size * 2, /*end=*/q_size * 2 + kv_size)
          .view({-1, num_kv_heads, head_size});
  torch::Tensor v = qkvg.slice(/*dim=*/1,
                               /*start=*/q_size * 2 + kv_size,
                               /*end=*/q_size * 2 + kv_size * 2)
                        .view({-1, num_kv_heads, head_size});

  torch::Tensor q_norm = rms_norm_ref(q, q_weight, eps);
  torch::Tensor k_norm = rms_norm_ref(k, k_weight, eps);
  const int64_t half_rope_dim = cos_sin.size(2) / 2;
  auto [cos_row, sin_row] =
      is_interleaved
          ? assemble_interleaved_mrope_rows_ref(cos_sin, mrope_section)
          : assemble_non_interleaved_mrope_rows_ref(cos_sin, mrope_section);

  auto apply_half_mrope = [&](const torch::Tensor& x) {
    torch::Tensor out = x.clone();
    torch::Tensor x1 = out.slice(/*dim=*/2, /*start=*/0, /*end=*/half_rope_dim);
    torch::Tensor x2 = out.slice(/*dim=*/2,
                                 /*start=*/half_rope_dim,
                                 /*end=*/half_rope_dim * 2);
    torch::Tensor cos_expand = cos_row.unsqueeze(/*dim=*/1);
    torch::Tensor sin_expand = sin_row.unsqueeze(/*dim=*/1);
    torch::Tensor out_first = x1 * cos_expand - x2 * sin_expand;
    torch::Tensor out_second = x2 * cos_expand + x1 * sin_expand;
    out.slice(/*dim=*/2, /*start=*/0, /*end=*/half_rope_dim).copy_(out_first);
    out.slice(/*dim=*/2,
              /*start=*/half_rope_dim,
              /*end=*/half_rope_dim * 2)
        .copy_(out_second);
    return out;
  };

  torch::Tensor q_out = apply_half_mrope(q_norm).to(qkvg.dtype());
  torch::Tensor k_out = apply_half_mrope(k_norm).to(qkvg.dtype());
  return {q_out, k_out, v.contiguous(), gate.contiguous()};
}

float max_abs_diff(const torch::Tensor& lhs, const torch::Tensor& rhs) {
  return (lhs.to(torch::kFloat32) - rhs.to(torch::kFloat32))
      .abs()
      .max()
      .item<float>();
}

void run_case(const SplitQkvTestCase& test_case) {
  ASSERT_GT(test_case.num_tokens, 0);
  ASSERT_GT(test_case.num_q_heads, 0);
  ASSERT_GT(test_case.num_kv_heads, 0);

  const torch::Device device("npu:0");
  const auto opts =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const std::vector<int64_t> mrope_section = {11, 11, 10};
  const int64_t rope_dim =
      2 * (mrope_section[0] + mrope_section[1] + mrope_section[2]);
  const int64_t q_size = test_case.num_q_heads * kHeadSize;
  const int64_t kv_size = test_case.num_kv_heads * kHeadSize;

  torch::manual_seed(test_case.seed);
  torch::Tensor qkv =
      torch::randn({test_case.num_tokens, q_size * 2 + kv_size * 2}, opts);
  torch::Tensor q_weight = torch::randn({kHeadSize}, opts);
  torch::Tensor k_weight = torch::randn({kHeadSize}, opts);
  torch::Tensor phase = torch::randn(
      {3, test_case.num_tokens, rope_dim / 2},
      torch::TensorOptions().dtype(torch::kFloat32).device(device));
  torch::Tensor cos_sin =
      torch::cat({torch::cos(phase), torch::sin(phase)}, /*dim=*/2)
          .to(torch::kBFloat16);
  torch::Tensor cos_sin_merged = merge_cos_sin_for_wrapper(cos_sin);
  torch::Tensor gather_pattern = build_split_qkv_rmsnorm_mrope_gather_pattern(
      rope_dim, mrope_section, test_case.is_interleaved, device);

  auto [q_ref, k_ref, v_ref, gate_ref] =
      torch_split_qkv_rmsnorm_mrope(qkv,
                                    q_weight,
                                    k_weight,
                                    cos_sin,
                                    test_case.num_q_heads,
                                    test_case.num_kv_heads,
                                    kHeadSize,
                                    kRmsNormEps,
                                    mrope_section,
                                    test_case.is_interleaved);
  auto [q_out, k_out, v_out, gate_out] =
      split_qkv_rmsnorm_mrope(qkv,
                              q_weight,
                              k_weight,
                              cos_sin_merged,
                              gather_pattern,
                              static_cast<float>(kRmsNormEps),
                              test_case.num_q_heads,
                              test_case.num_kv_heads,
                              kHeadSize);

  EXPECT_EQ(q_out.dim(), 3);
  EXPECT_EQ(k_out.dim(), 3);
  EXPECT_EQ(v_out.dim(), 3);
  EXPECT_EQ(gate_out.dim(), 3);
  EXPECT_EQ(q_out.size(0), test_case.num_tokens);
  EXPECT_EQ(q_out.size(1), test_case.num_q_heads);
  EXPECT_EQ(q_out.size(2), kHeadSize);
  EXPECT_EQ(k_out.size(0), test_case.num_tokens);
  EXPECT_EQ(k_out.size(1), test_case.num_kv_heads);
  EXPECT_EQ(k_out.size(2), kHeadSize);
  EXPECT_EQ(v_out.size(0), test_case.num_tokens);
  EXPECT_EQ(v_out.size(1), test_case.num_kv_heads);
  EXPECT_EQ(v_out.size(2), kHeadSize);
  EXPECT_EQ(gate_out.size(0), test_case.num_tokens);
  EXPECT_EQ(gate_out.size(1), test_case.num_q_heads);
  EXPECT_EQ(gate_out.size(2), kHeadSize);

  aclrtStream stream = c10_npu::getCurrentNPUStream(device.index()).stream();
  aclrtSynchronizeStream(stream);

  EXPECT_TRUE(torch::allclose(q_out, q_ref, /*rtol=*/1e-2, /*atol=*/1e-2))
      << "q mismatch, max_diff=" << max_abs_diff(q_out, q_ref);
  EXPECT_TRUE(torch::allclose(k_out, k_ref, /*rtol=*/1e-2, /*atol=*/1e-2))
      << "k mismatch, max_diff=" << max_abs_diff(k_out, k_ref);
  EXPECT_TRUE(torch::allclose(v_out, v_ref, /*rtol=*/0, /*atol=*/0))
      << "v mismatch, max_diff=" << max_abs_diff(v_out, v_ref);
  EXPECT_TRUE(torch::allclose(gate_out, gate_ref, /*rtol=*/0, /*atol=*/0))
      << "gate mismatch, max_diff=" << max_abs_diff(gate_out, gate_ref);
}

TEST_F(TileLangSplitQkvRmsnormMRopeWrapperTest,
       BuildGatherPatternMatchesReference) {
  const torch::Device device("npu:0");
  const std::vector<int64_t> mrope_section = {11, 11, 10};
  const int64_t rope_dim = 64;

  torch::Tensor non_interleaved =
      build_split_qkv_rmsnorm_mrope_gather_pattern(
          rope_dim, mrope_section, /*is_interleaved=*/false, device)
          .cpu()
          .view(torch::kInt32);
  EXPECT_EQ(non_interleaved.dtype(), torch::kInt32);
  EXPECT_EQ(non_interleaved.size(0), 256);
  EXPECT_EQ(non_interleaved[0].item<int32_t>(), 0);
  EXPECT_EQ(non_interleaved[10].item<int32_t>(), 10);
  EXPECT_EQ(non_interleaved[11].item<int32_t>(), 75);
  EXPECT_EQ(non_interleaved[21].item<int32_t>(), 85);
  EXPECT_EQ(non_interleaved[22].item<int32_t>(), 150);
  EXPECT_EQ(non_interleaved[31].item<int32_t>(), 159);
  EXPECT_EQ(non_interleaved[32].item<int32_t>(), 32);
  EXPECT_EQ(non_interleaved[42].item<int32_t>(), 42);
  EXPECT_EQ(non_interleaved[43].item<int32_t>(), 107);
  EXPECT_EQ(non_interleaved[53].item<int32_t>(), 117);
  EXPECT_EQ(non_interleaved[54].item<int32_t>(), 182);
  EXPECT_EQ(non_interleaved[63].item<int32_t>(), 191);
  EXPECT_EQ(non_interleaved[64].item<int32_t>(), 0);
  EXPECT_EQ(non_interleaved[255].item<int32_t>(), 0);

  torch::Tensor interleaved =
      build_split_qkv_rmsnorm_mrope_gather_pattern(
          rope_dim, mrope_section, /*is_interleaved=*/true, device)
          .cpu()
          .view(torch::kInt32);
  EXPECT_EQ(interleaved.dtype(), torch::kInt32);
  EXPECT_EQ(interleaved.size(0), 256);
  EXPECT_EQ(interleaved[0].item<int32_t>(), 0);
  EXPECT_EQ(interleaved[1].item<int32_t>(), 65);
  EXPECT_EQ(interleaved[2].item<int32_t>(), 130);
  EXPECT_EQ(interleaved[3].item<int32_t>(), 3);
  EXPECT_EQ(interleaved[4].item<int32_t>(), 68);
  EXPECT_EQ(interleaved[5].item<int32_t>(), 133);
  EXPECT_EQ(interleaved[30].item<int32_t>(), 30);
  EXPECT_EQ(interleaved[31].item<int32_t>(), 95);
  EXPECT_EQ(interleaved[32].item<int32_t>(), 32);
  EXPECT_EQ(interleaved[33].item<int32_t>(), 97);
  EXPECT_EQ(interleaved[34].item<int32_t>(), 162);
  EXPECT_EQ(interleaved[35].item<int32_t>(), 35);
  EXPECT_EQ(interleaved[63].item<int32_t>(), 127);
  EXPECT_EQ(interleaved[64].item<int32_t>(), 0);
  EXPECT_EQ(interleaved[255].item<int32_t>(), 0);
}

TEST_F(TileLangSplitQkvRmsnormMRopeWrapperTest, MatchesTorchReference) {
  const std::vector<SplitQkvTestCase> cases = {
      {.name = "tiny_t1_q16_kv4",
       .num_tokens = 1,
       .num_q_heads = 16,
       .num_kv_heads = 4,
       .seed = 101,
       .is_interleaved = false},
      {.name = "medium_t17_q16_kv4",
       .num_tokens = 17,
       .num_q_heads = 16,
       .num_kv_heads = 4,
       .seed = 102,
       .is_interleaved = false},
      {.name = "chunked_t4097_q16_kv4",
       .num_tokens = 4097,
       .num_q_heads = 16,
       .num_kv_heads = 4,
       .seed = 103,
       .is_interleaved = false},
      {.name = "tiny_t1_q16_kv2",
       .num_tokens = 1,
       .num_q_heads = 16,
       .num_kv_heads = 2,
       .seed = 201,
       .is_interleaved = false},
      {.name = "medium_t17_q16_kv2",
       .num_tokens = 17,
       .num_q_heads = 16,
       .num_kv_heads = 2,
       .seed = 202,
       .is_interleaved = false},
      {.name = "chunked_t4097_q16_kv2",
       .num_tokens = 4097,
       .num_q_heads = 16,
       .num_kv_heads = 2,
       .seed = 203,
       .is_interleaved = false},
      {.name = "tiny_t1_q8_kv1",
       .num_tokens = 1,
       .num_q_heads = 8,
       .num_kv_heads = 1,
       .seed = 301,
       .is_interleaved = false},
      {.name = "medium_t17_q8_kv1",
       .num_tokens = 17,
       .num_q_heads = 8,
       .num_kv_heads = 1,
       .seed = 302,
       .is_interleaved = false},
      {.name = "chunked_t4097_q8_kv1",
       .num_tokens = 4097,
       .num_q_heads = 8,
       .num_kv_heads = 1,
       .seed = 303,
       .is_interleaved = false},
      {.name = "tiny_t1_q16_kv4_interleaved",
       .num_tokens = 1,
       .num_q_heads = 16,
       .num_kv_heads = 4,
       .seed = 401,
       .is_interleaved = true},
      {.name = "medium_t17_q16_kv4_interleaved",
       .num_tokens = 17,
       .num_q_heads = 16,
       .num_kv_heads = 4,
       .seed = 402,
       .is_interleaved = true},
      {.name = "chunked_t4097_q16_kv4_interleaved",
       .num_tokens = 4097,
       .num_q_heads = 16,
       .num_kv_heads = 4,
       .seed = 403,
       .is_interleaved = true},
      {.name = "tiny_t1_q16_kv2_interleaved",
       .num_tokens = 1,
       .num_q_heads = 16,
       .num_kv_heads = 2,
       .seed = 501,
       .is_interleaved = true},
      {.name = "medium_t17_q16_kv2_interleaved",
       .num_tokens = 17,
       .num_q_heads = 16,
       .num_kv_heads = 2,
       .seed = 502,
       .is_interleaved = true},
      {.name = "chunked_t4097_q16_kv2_interleaved",
       .num_tokens = 4097,
       .num_q_heads = 16,
       .num_kv_heads = 2,
       .seed = 503,
       .is_interleaved = true},
      {.name = "tiny_t1_q8_kv1_interleaved",
       .num_tokens = 1,
       .num_q_heads = 8,
       .num_kv_heads = 1,
       .seed = 601,
       .is_interleaved = true},
      {.name = "medium_t17_q8_kv1_interleaved",
       .num_tokens = 17,
       .num_q_heads = 8,
       .num_kv_heads = 1,
       .seed = 602,
       .is_interleaved = true},
      {.name = "chunked_t4097_q8_kv1_interleaved",
       .num_tokens = 4097,
       .num_q_heads = 8,
       .num_kv_heads = 1,
       .seed = 603,
       .is_interleaved = true},
      // Odd num_tokens
      {.name = "odd_t3_q16_kv4",
       .num_tokens = 3,
       .num_q_heads = 16,
       .num_kv_heads = 4,
       .seed = 701,
       .is_interleaved = false},
      {.name = "odd_t7_q16_kv2_interleaved",
       .num_tokens = 7,
       .num_q_heads = 16,
       .num_kv_heads = 2,
       .seed = 702,
       .is_interleaved = true},
      {.name = "odd_t15_q8_kv1",
       .num_tokens = 15,
       .num_q_heads = 8,
       .num_kv_heads = 1,
       .seed = 703,
       .is_interleaved = false},
      {.name = "odd_t33_q16_kv4_interleaved",
       .num_tokens = 33,
       .num_q_heads = 16,
       .num_kv_heads = 4,
       .seed = 704,
       .is_interleaved = true},
      // Boundary: min specialization bucket
      {.name = "boundary_t2_q16_kv2",
       .num_tokens = 2,
       .num_q_heads = 16,
       .num_kv_heads = 2,
       .seed = 801,
       .is_interleaved = false},
      // Boundary: vec_core_num (48) and neighbors
      {.name = "boundary_t48_q8_kv1_interleaved",
       .num_tokens = 48,
       .num_q_heads = 8,
       .num_kv_heads = 1,
       .seed = 802,
       .is_interleaved = true},
      {.name = "boundary_t49_q16_kv4",
       .num_tokens = 49,
       .num_q_heads = 16,
       .num_kv_heads = 4,
       .seed = 803,
       .is_interleaved = false},
      // Large: old 4096 boundary
      {.name = "large_t4096_q16_kv2",
       .num_tokens = 4096,
       .num_q_heads = 16,
       .num_kv_heads = 2,
       .seed = 901,
       .is_interleaved = false},
      // Large: odd, exceeding old chunking limit
      {.name = "large_t4099_q8_kv1_interleaved",
       .num_tokens = 4099,
       .num_q_heads = 8,
       .num_kv_heads = 1,
       .seed = 902,
       .is_interleaved = true},
      // Large batch
      {.name = "large_t8192_q16_kv4",
       .num_tokens = 8192,
       .num_q_heads = 16,
       .num_kv_heads = 4,
       .seed = 903,
       .is_interleaved = false},
      {.name = "large_t8193_q16_kv4_interleaved",
       .num_tokens = 8193,
       .num_q_heads = 16,
       .num_kv_heads = 4,
       .seed = 904,
       .is_interleaved = true},
      // Qwen3.5-0.8B/2B tp=1, Qwen3.5-4B/9B tp=2
      {.name = "tiny_t1_q8_kv2",
       .num_tokens = 1,
       .num_q_heads = 8,
       .num_kv_heads = 2,
       .seed = 1001,
       .is_interleaved = false},
      {.name = "large_t4097_q8_kv2",
       .num_tokens = 4097,
       .num_q_heads = 8,
       .num_kv_heads = 2,
       .seed = 1002,
       .is_interleaved = false},
      // Qwen3.5-27B tp=1
      {.name = "tiny_t1_q24_kv4",
       .num_tokens = 1,
       .num_q_heads = 24,
       .num_kv_heads = 4,
       .seed = 1101,
       .is_interleaved = false},
      {.name = "large_t4097_q24_kv4",
       .num_tokens = 4097,
       .num_q_heads = 24,
       .num_kv_heads = 4,
       .seed = 1102,
       .is_interleaved = true},
      // Qwen3.5-27B tp=2
      {.name = "tiny_t1_q12_kv2",
       .num_tokens = 1,
       .num_q_heads = 12,
       .num_kv_heads = 2,
       .seed = 1201,
       .is_interleaved = false},
      {.name = "large_t4097_q12_kv2",
       .num_tokens = 4097,
       .num_q_heads = 12,
       .num_kv_heads = 2,
       .seed = 1202,
       .is_interleaved = true},
      // Qwen3.5-27B tp=4
      {.name = "tiny_t1_q6_kv1",
       .num_tokens = 1,
       .num_q_heads = 6,
       .num_kv_heads = 1,
       .seed = 1301,
       .is_interleaved = false},
      {.name = "large_t4097_q6_kv1",
       .num_tokens = 4097,
       .num_q_heads = 6,
       .num_kv_heads = 1,
       .seed = 1302,
       .is_interleaved = true},
      // Qwen3.5-27B tp=8
      {.name = "tiny_t1_q3_kv1",
       .num_tokens = 1,
       .num_q_heads = 3,
       .num_kv_heads = 1,
       .seed = 1401,
       .is_interleaved = false},
      {.name = "large_t4097_q3_kv1",
       .num_tokens = 4097,
       .num_q_heads = 3,
       .num_kv_heads = 1,
       .seed = 1402,
       .is_interleaved = true},
      // Qwen3.5-35B tp=4, Qwen3.5-4B tp=4
      {.name = "tiny_t1_q4_kv1",
       .num_tokens = 1,
       .num_q_heads = 4,
       .num_kv_heads = 1,
       .seed = 1501,
       .is_interleaved = false},
      {.name = "large_t4097_q4_kv1",
       .num_tokens = 4097,
       .num_q_heads = 4,
       .num_kv_heads = 1,
       .seed = 1502,
       .is_interleaved = true},
      // Qwen3.5-122B/397B tp=1
      {.name = "tiny_t1_q32_kv2",
       .num_tokens = 1,
       .num_q_heads = 32,
       .num_kv_heads = 2,
       .seed = 1601,
       .is_interleaved = false},
      {.name = "large_t4097_q32_kv2",
       .num_tokens = 4097,
       .num_q_heads = 32,
       .num_kv_heads = 2,
       .seed = 1602,
       .is_interleaved = true},
      // Qwen3.5-122B/397B tp=2
      {.name = "tiny_t1_q16_kv1",
       .num_tokens = 1,
       .num_q_heads = 16,
       .num_kv_heads = 1,
       .seed = 1701,
       .is_interleaved = false},
      {.name = "large_t4097_q16_kv1",
       .num_tokens = 4097,
       .num_q_heads = 16,
       .num_kv_heads = 1,
       .seed = 1702,
       .is_interleaved = true},
      // Qwen3.5-0.8B/2B tp=4, various tp=8/16
      {.name = "tiny_t1_q2_kv1",
       .num_tokens = 1,
       .num_q_heads = 2,
       .num_kv_heads = 1,
       .seed = 1801,
       .is_interleaved = false},
      {.name = "large_t4097_q2_kv1",
       .num_tokens = 4097,
       .num_q_heads = 2,
       .num_kv_heads = 1,
       .seed = 1802,
       .is_interleaved = true},
  };

  for (const SplitQkvTestCase& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    run_case(test_case);
  }
}

}  // namespace
}  // namespace xllm::kernel::npu::tilelang

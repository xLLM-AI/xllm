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
#include <torch_npu/torch_npu.h>

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

namespace xllm::kernel::npu::tilelang {
namespace {

class TileLangFusedSigmoidGatingDeltaRuleWrapperTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { torch_npu::init_npu("npu:0"); }

  static void TearDownTestSuite() { torch_npu::finalize_npu(); }
};

struct FusedSigmoidGatingDeltaRuleTestCase {
  std::string name;
  std::vector<int64_t> seqlens;
  int64_t nk;
  int64_t nv;
  int64_t dk;
  int64_t dv;
  int64_t seed;
  float softplus_beta = 1.0F;
  torch::ScalarType init_state_dtype = torch::kFloat32;
};

std::tuple<torch::Tensor, torch::Tensor> torch_fused_sigmoid_gating_delta_rule(
    const torch::Tensor& A_log,
    const torch::Tensor& a,
    const torch::Tensor& dt_bias,
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& beta,
    const torch::Tensor& init_state,
    const torch::Tensor& ssm_state_indices,
    const torch::Tensor& cu_seqlens,
    float softplus_beta = 1.0F) {
  const int64_t total_tokens = query.size(1);
  const int64_t nk = query.size(2);
  const int64_t dk = query.size(3);
  const int64_t nv = value.size(2);
  const int64_t dv = value.size(3);
  const int64_t num_seqs = cu_seqlens.size(0) - 1;
  const float scale = 1.0F / std::sqrt(static_cast<float>(dk));
  const float l2_norm_eps = 1e-6F;
  const int64_t v_per_k = nv / nk;
  const auto fp32_opts = query.options().dtype(torch::kFloat32);

  auto state = torch::zeros({num_seqs, nv, dk, dv}, fp32_opts);
  for (int64_t i = 0; i < num_seqs; ++i) {
    int64_t state_idx = ssm_state_indices[i].item<int64_t>();
    if (state_idx >= 0) {
      state[i] = init_state[state_idx].to(torch::kFloat32);
    }
  }

  auto A_log_f = A_log.to(torch::kFloat32);
  auto dt_bias_f = dt_bias.to(torch::kFloat32);
  auto a_f = a.to(torch::kFloat32);
  auto beta_f = beta.to(torch::kFloat32);
  auto query_f = query.to(torch::kFloat32);
  auto key_f = key.to(torch::kFloat32);
  auto value_f = value.to(torch::kFloat32);

  auto exp_A = torch::exp(A_log_f);
  auto out = torch::empty({1, total_tokens, nv, dv}, query.options());

  for (int64_t seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
    int64_t seq_start = cu_seqlens[seq_idx].item<int64_t>();
    int64_t seq_end = cu_seqlens[seq_idx + 1].item<int64_t>();
    for (int64_t v_head_idx = 0; v_head_idx < nv; ++v_head_idx) {
      auto h = state[seq_idx][v_head_idx];
      int64_t k_head_idx = v_head_idx / v_per_k;
      for (int64_t token = seq_start; token < seq_end; ++token) {
        auto q_t = query_f[0][token][k_head_idx];
        auto k_t = key_f[0][token][k_head_idx];
        auto v_t = value_f[0][token][v_head_idx];

        {
          auto q_norm = q_t / torch::sqrt((q_t * q_t).sum() + l2_norm_eps);
          auto k_norm = k_t / torch::sqrt((k_t * k_t).sum() + l2_norm_eps);
          q_t = q_norm;
          k_t = k_norm;
        }

        auto x = a_f[token][v_head_idx] + dt_bias_f[v_head_idx];
        auto beta_x = softplus_beta * x;
        auto sp =
            torch::where(beta_x > 20.0F,
                         x,
                         torch::log1p(torch::exp(beta_x)) / softplus_beta);

        h = h * torch::exp(-exp_A[v_head_idx] * sp);
        auto pred = torch::matmul(k_t.unsqueeze(0), h).squeeze(0);
        h = h +
            torch::outer(
                k_t, (v_t - pred) * torch::sigmoid(beta_f[token][v_head_idx]));
        out[0][token][v_head_idx] =
            torch::matmul((q_t * scale).unsqueeze(0), h).squeeze(0);
      }
      state[seq_idx][v_head_idx] = h;
    }
  }

  return {out.to(query.scalar_type()), state};
}

void run_fused_sigmoid_gating_delta_rule_case(
    const FusedSigmoidGatingDeltaRuleTestCase& test_case) {
  const auto device = torch::Device("npu:0");
  torch::manual_seed(test_case.seed);

  const int64_t num_seqs = static_cast<int64_t>(test_case.seqlens.size());
  ASSERT_GT(num_seqs, 0);

  int64_t total_tokens = 0;
  std::vector<int32_t> cu_seqlens_vec;
  cu_seqlens_vec.push_back(0);
  for (int64_t len : test_case.seqlens) {
    total_tokens += len;
    cu_seqlens_vec.push_back(static_cast<int32_t>(total_tokens));
  }

  const auto bf16_opts =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const auto fp32_opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  const auto i32_opts =
      torch::TensorOptions().dtype(torch::kInt32).device(device);

  const int64_t nk = test_case.nk;
  const int64_t nv = test_case.nv;
  const int64_t dk = test_case.dk;
  const int64_t dv = test_case.dv;

  auto A_log = torch::randn({nv}, fp32_opts);
  auto a = torch::randn({total_tokens, nv}, bf16_opts);
  auto dt_bias = torch::randn({nv}, fp32_opts);
  const int64_t q_width = nk * dk;
  const int64_t k_width = nk * dk;
  const int64_t v_width = nv * dv;
  auto qkv =
      torch::randn({total_tokens, q_width + k_width + v_width}, bf16_opts);
  auto query = qkv.narrow(1, 0, q_width).view({total_tokens, nk, dk});
  auto key = qkv.narrow(1, q_width, k_width).view({total_tokens, nk, dk});
  auto value =
      qkv.narrow(1, q_width + k_width, v_width).view({total_tokens, nv, dv});
  auto beta = torch::randn({total_tokens, nv}, bf16_opts);
  ASSERT_GT(query.stride(0), nk * dk);
  ASSERT_GT(key.stride(0), nk * dk);
  ASSERT_GT(value.stride(0), nv * dv);

  int64_t num_cache_slots = num_seqs * 2;
  auto init_state = torch::randn(
      {num_cache_slots, nv, dk, dv},
      torch::TensorOptions().dtype(test_case.init_state_dtype).device(device));
  auto ssm_state_indices = torch::arange(num_seqs, i32_opts);
  auto cu_seqlens = torch::tensor(cu_seqlens_vec, i32_opts);

  auto [out_ref, final_state_ref] =
      torch_fused_sigmoid_gating_delta_rule(A_log,
                                            a,
                                            dt_bias,
                                            query.unsqueeze(0),
                                            key.unsqueeze(0),
                                            value.unsqueeze(0),
                                            beta,
                                            init_state,
                                            ssm_state_indices,
                                            cu_seqlens,
                                            test_case.softplus_beta);

  // TileLang kernel.
  const float scale_val = 1.0F / std::sqrt(static_cast<float>(dk));
  auto [out_out, final_state_out] =
      fused_sigmoid_gating_delta_rule(A_log,
                                      a,
                                      dt_bias,
                                      query,
                                      key,
                                      value,
                                      beta,
                                      init_state,
                                      ssm_state_indices,
                                      cu_seqlens,
                                      scale_val,
                                      /*use_qk_l2norm_in_kernel=*/true,
                                      test_case.softplus_beta,
                                      /*softplus_threshold=*/20.0F);

  auto out_sliced = out_out.unsqueeze(0);
  auto final_state_sliced = final_state_out.slice(0, 0, num_seqs);

  EXPECT_TRUE(
      torch::allclose(out_sliced, out_ref, /*rtol=*/2e-2, /*atol=*/2e-2))
      << "out mismatch for case " << test_case.name;
  EXPECT_TRUE(torch::allclose(final_state_sliced,
                              final_state_ref,
                              /*rtol=*/2e-2,
                              /*atol=*/2e-2))
      << "final_state mismatch for case " << test_case.name;
  EXPECT_TRUE(
      torch::allclose(init_state.narrow(0, 0, num_seqs).to(torch::kFloat32),
                      final_state_ref,
                      /*rtol=*/2e-2,
                      /*atol=*/2e-2))
      << "in-place cache update mismatch for case " << test_case.name;
}

TEST_F(TileLangFusedSigmoidGatingDeltaRuleWrapperTest, MatchesTorchReference) {
  const std::vector<FusedSigmoidGatingDeltaRuleTestCase> cases = {
      {
          .name = "small_4x8_d128",
          .seqlens = {4, 8, 6, 3},
          .nk = 4,
          .nv = 8,
          .dk = 128,
          .dv = 128,
          .seed = 20260421,
      },
      {
          .name = "medium_16x32_d128",
          .seqlens = {4, 8, 12, 6, 3, 7, 5, 9},
          .nk = 16,
          .nv = 32,
          .dk = 128,
          .dv = 128,
          .seed = 20260422,
      },
      {
          .name = "oversized_40x32_decode_d128",
          .seqlens = std::vector<int64_t>(40, 1),
          .nk = 16,
          .nv = 32,
          .dk = 128,
          .dv = 128,
          .seed = 20260703,
      },
      {
          .name = "bf16_cache_4x8_decode_d128",
          .seqlens = std::vector<int64_t>(4, 1),
          .nk = 4,
          .nv = 8,
          .dk = 128,
          .dv = 128,
          .seed = 20260704,
          .init_state_dtype = torch::kBFloat16,
      },
  };

  for (const auto& test_case : cases) {
    SCOPED_TRACE(::testing::Message() << "case=" << test_case.name);
    run_fused_sigmoid_gating_delta_rule_case(test_case);
  }
}

}  // namespace
}  // namespace xllm::kernel::npu::tilelang

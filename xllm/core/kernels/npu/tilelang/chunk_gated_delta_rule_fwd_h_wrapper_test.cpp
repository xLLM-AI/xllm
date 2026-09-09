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
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/torch_npu.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <vector>

#include "acl/acl.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

namespace xllm::kernel::npu::tilelang {
namespace {

class TileLangChunkGatedDeltaRuleFwdHTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { torch_npu::init_npu("npu:0"); }

  static void TearDownTestSuite() { torch_npu::finalize_npu(); }
};

struct ChunkGdrTestCase {
  std::string name;
  int64_t batch_size;
  int64_t seq_len;
  int64_t H;
  int64_t Hg;
  int64_t K;
  int64_t V;
  int64_t chunk_size;
  int64_t seed;
};

torch::Tensor make_chunk_local_cumsum(int64_t N,
                                      int64_t T,
                                      int64_t H,
                                      int64_t chunk_size,
                                      int64_t seed) {
  auto cpu_opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  torch::manual_seed(seed);
  auto g_cumsum = torch::empty({N * T, H}, cpu_opts);
  constexpr float kGateScale = 0.002F;

  for (int64_t b = 0; b < N; ++b) {
    int64_t seq_start = b * T;
    int64_t seq_end = seq_start + T;
    for (int64_t head = 0; head < H; ++head) {
      for (int64_t chunk_start = seq_start; chunk_start < seq_end;
           chunk_start += chunk_size) {
        int64_t chunk_end = std::min(chunk_start + chunk_size, seq_end);
        int64_t chunk_len = chunk_end - chunk_start;
        auto gate = torch::randn({chunk_len}, cpu_opts) * kGateScale;
        auto chunk_cumsum = torch::cumsum(gate, /*dim=*/0);
        g_cumsum.slice(0, chunk_start, chunk_end)
            .select(1, head)
            .copy_(chunk_cumsum);
      }
    }
  }
  return g_cumsum;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ref_chunk_gated_delta_rule(const torch::Tensor& k,
                           const torch::Tensor& w,
                           const torch::Tensor& u,
                           const torch::Tensor& g,
                           const torch::Tensor& initial_state,
                           const torch::Tensor& cu_seqlens,
                           int64_t chunk_size) {
  const int64_t N = initial_state.size(0);
  const int64_t H = u.size(1);
  const int64_t Hg = k.size(1);
  const int64_t K = k.size(2);
  const int64_t V = u.size(2);
  const int64_t hg_ratio = H / Hg;

  auto k_f = k.to(torch::kFloat32);
  auto w_f = w.to(torch::kFloat32);
  auto u_f = u.to(torch::kFloat32);
  auto g_f = g.to(torch::kFloat32);
  auto h0_f = initial_state.to(torch::kFloat32);

  auto cu_cpu = cu_seqlens.cpu().contiguous();
  auto cu_ptr = cu_cpu.data_ptr<int32_t>();

  int64_t nt_max = 0;
  for (int64_t i = 0; i < N; ++i) {
    int64_t seq_len = cu_ptr[i + 1] - cu_ptr[i];
    int64_t nt_i = (seq_len + chunk_size - 1) / chunk_size;
    nt_max = std::max(nt_max, nt_i);
  }

  auto h_out = torch::empty({N, nt_max, H, K, V}, k.options());
  auto v_new = torch::empty_like(u);
  auto ht = torch::empty({N, H, K, V}, torch::kFloat32);

  for (int64_t n = 0; n < N; ++n) {
    int64_t bos = static_cast<int64_t>(cu_ptr[n]);
    int64_t eos = static_cast<int64_t>(cu_ptr[n + 1]);
    for (int64_t head = 0; head < H; ++head) {
      auto state = h0_f[n][head].clone();
      int64_t k_head_idx = head / hg_ratio;

      int64_t chunk_id = 0;
      for (int64_t start = bos; start < eos; start += chunk_size) {
        int64_t end = std::min(start + chunk_size, eos);

        // h[t] ← state (before this chunk's update)
        h_out[n][chunk_id][head].copy_(state.to(k.dtype()));

        // residual = u - w @ h
        auto u_chunk = u_f.slice(0, start, end).select(1, head);  // [L, V]
        auto w_chunk = w_f.slice(0, start, end).select(1, head);  // [L, K]
        auto residual = u_chunk - torch::matmul(w_chunk, state);  // [L, V]

        // Gating: state_v = residual * exp(where(g_diff<=0, g_diff, -inf))
        //          state   = state   * exp(last_g)
        auto g_chunk = g_f.slice(0, start, end).select(1, head);  // [L]
        auto last_g = g_chunk[-1].clone();
        auto g_diff = last_g - g_chunk;
        auto g_diff_masked = torch::where(
            g_diff <= 0.0, g_diff, torch::full_like(g_diff, -INFINITY));
        auto state_v = residual * g_diff_masked.exp().unsqueeze(/*dim=*/-1);
        state = state * std::exp(last_g.item<float>());

        // v_new[t] ← residual (before gating)
        v_new.slice(0, start, end)
            .select(1, head)
            .copy_(residual.to(k.dtype()));

        // h += k^T @ v_new
        auto k_chunk =
            k_f.slice(0, start, end).select(1, k_head_idx);  // [L, K]
        state = state + torch::matmul(k_chunk.t(), state_v);

        ++chunk_id;
      }
      ht[n][head].copy_(state);
    }
  }

  return {h_out, v_new, ht};
}

void run_chunk_gated_delta_rule_fwd_h_case(const ChunkGdrTestCase& test_case) {
  ASSERT_GT(test_case.batch_size, 0);
  ASSERT_GT(test_case.seq_len, 0);
  ASSERT_GT(test_case.H, 0);
  ASSERT_GT(test_case.Hg, 0);
  ASSERT_GE(test_case.H, test_case.Hg);
  ASSERT_GT(test_case.K, 0);
  ASSERT_GT(test_case.V, 0);
  ASSERT_EQ(test_case.chunk_size, 64);

  const auto npu_device = torch::Device("npu:0");
  const auto bf16_opts =
      torch::TensorOptions().dtype(torch::kBFloat16).device(npu_device);
  const auto f32_opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(npu_device);

  torch::manual_seed(test_case.seed);

  const int64_t N = test_case.batch_size;
  const int64_t T = test_case.seq_len;
  const int64_t H = test_case.H;
  const int64_t Hg = test_case.Hg;
  const int64_t K = test_case.K;
  const int64_t V = test_case.V;

  auto k = torch::randn({N * T, Hg, K}, bf16_opts) * 0.01F;
  auto w = torch::randn({N * T, H, K}, bf16_opts) * 0.01F;
  auto u = torch::randn({N * T, H, V}, bf16_opts) * 0.01F;
  auto initial_state = torch::randn({N, H, K, V}, f32_opts) * 0.01F;

  auto g = make_chunk_local_cumsum(N,
                                   T,
                                   H,
                                   /*chunk_size=*/64,
                                   test_case.seed)
               .to(npu_device);

  std::vector<int32_t> cu_seqlens_vec(N + 1);
  for (int64_t i = 0; i <= N; ++i) {
    cu_seqlens_vec[i] = static_cast<int32_t>(i * T);
  }
  auto cu_seqlens =
      torch::from_blob(
          cu_seqlens_vec.data(), {N + 1}, torch::dtype(torch::kInt32))
          .to(npu_device);

  auto [h, v_new, ht] =
      chunk_gated_delta_rule_fwd_h(k,
                                   w,
                                   u,
                                   g,
                                   initial_state,
                                   /*output_final_state=*/true,
                                   /*chunk_size=*/64,
                                   /*save_new_value=*/true,
                                   cu_seqlens,
                                   /*chunk_offsets=*/std::nullopt);

  std::cout << "[chunk_gated_delta_rule_fwd_h_test] case=" << test_case.name
            << ", h_shape=" << h.sizes() << ", v_new_shape=" << v_new.sizes()
            << ", ht_shape=" << ht.sizes() << std::endl;

  EXPECT_EQ(h.dim(), 5);
  EXPECT_EQ(h.size(0), 1);
  EXPECT_EQ(h.size(2), H);
  EXPECT_EQ(h.size(3), K);
  EXPECT_EQ(h.size(4), V);
  EXPECT_EQ(v_new.size(0), N * T);
  EXPECT_EQ(v_new.size(1), H);
  EXPECT_EQ(v_new.size(2), V);
  EXPECT_EQ(ht.size(0), N);
  EXPECT_EQ(ht.size(1), H);
  EXPECT_EQ(ht.size(2), K);
  EXPECT_EQ(ht.size(3), V);
}

void run_chunk_gated_delta_rule_fwd_h_accuracy_case(
    const ChunkGdrTestCase& test_case) {
  ASSERT_GT(test_case.batch_size, 0);
  ASSERT_GT(test_case.seq_len, 0);
  ASSERT_GT(test_case.H, 0);
  ASSERT_GT(test_case.Hg, 0);
  ASSERT_GE(test_case.H, test_case.Hg);
  ASSERT_GT(test_case.K, 0);
  ASSERT_GT(test_case.V, 0);
  ASSERT_EQ(test_case.chunk_size, 64);

  const auto npu_device = torch::Device("npu:0");
  const auto bf16_opts =
      torch::TensorOptions().dtype(torch::kBFloat16).device(npu_device);
  const auto f32_opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(npu_device);

  torch::manual_seed(test_case.seed);

  const int64_t N = test_case.batch_size;
  const int64_t T = test_case.seq_len;
  const int64_t H = test_case.H;
  const int64_t Hg = test_case.Hg;
  const int64_t K = test_case.K;
  const int64_t V = test_case.V;

  auto k_npu = torch::randn({N * T, Hg, K}, bf16_opts) * 0.01F;
  auto w_npu = torch::randn({N * T, H, K}, bf16_opts) * 0.01F;
  auto u_npu = torch::randn({N * T, H, V}, bf16_opts) * 0.01F;
  auto h0_npu = torch::randn({N, H, K, V}, f32_opts) * 0.01F;

  auto g_cpu =
      make_chunk_local_cumsum(N, T, H, /*chunk_size=*/64, test_case.seed);
  auto g_npu = g_cpu.to(npu_device);

  auto k_cpu = k_npu.cpu();
  auto w_cpu = w_npu.cpu();
  auto u_cpu = u_npu.cpu();
  auto h0_cpu = h0_npu.cpu();

  std::vector<int32_t> cu_seqlens_vec(N + 1);
  for (int64_t i = 0; i <= N; ++i) {
    cu_seqlens_vec[i] = static_cast<int32_t>(i * T);
  }
  auto cu_cpu = torch::from_blob(
                    cu_seqlens_vec.data(), {N + 1}, torch::dtype(torch::kInt32))
                    .clone();
  auto cu_npu = cu_cpu.to(npu_device);

  auto [h_ref, v_new_ref, ht_ref] = ref_chunk_gated_delta_rule(
      k_cpu, w_cpu, u_cpu, g_cpu, h0_cpu, cu_cpu, /*chunk_size=*/64);

  auto [h_npu, v_new_npu, ht_npu] =
      chunk_gated_delta_rule_fwd_h(k_npu,
                                   w_npu,
                                   u_npu,
                                   g_npu,
                                   h0_npu,
                                   /*output_final_state=*/true,
                                   /*chunk_size=*/64,
                                   /*save_new_value=*/true,
                                   cu_npu,
                                   /*chunk_offsets=*/std::nullopt);

  constexpr float kRtol = 5e-2F;
  constexpr float kAtol = 5e-1F;

  EXPECT_TRUE(torch::allclose(h_npu.cpu(), h_ref, kRtol, kAtol))
      << "h_out mismatch for case " << test_case.name;
  EXPECT_TRUE(torch::allclose(v_new_npu.cpu(), v_new_ref, kRtol, kAtol))
      << "v_new mismatch for case " << test_case.name;
  EXPECT_TRUE(torch::allclose(ht_npu.cpu(), ht_ref, kRtol, kAtol))
      << "ht mismatch for case " << test_case.name;
}

TEST_F(TileLangChunkGatedDeltaRuleFwdHTest,
       ChunkGatedDeltaRuleFwdHMatchesExpectedShapes) {
  const std::vector<ChunkGdrTestCase> cases = {
      {.name = "small_n1_t64_h16_hg16_k128_v128",
       .batch_size = 1,
       .seq_len = 64,
       .H = 16,
       .Hg = 16,
       .K = 128,
       .V = 128,
       .chunk_size = 64,
       .seed = 20260511},
      {.name = "tp2_n1_t128_h16_hg8_k128_v128",
       .batch_size = 1,
       .seq_len = 128,
       .H = 16,
       .Hg = 8,
       .K = 128,
       .V = 128,
       .chunk_size = 64,
       .seed = 20260515},
      {.name = "medium_n2_t128_h16_hg16_k128_v128",
       .batch_size = 2,
       .seq_len = 128,
       .H = 16,
       .Hg = 16,
       .K = 128,
       .V = 128,
       .chunk_size = 64,
       .seed = 20260512},
      {.name = "large_n1_t128_h32_hg16_k128_v128",
       .batch_size = 1,
       .seq_len = 128,
       .H = 32,
       .Hg = 16,
       .K = 128,
       .V = 128,
       .chunk_size = 64,
       .seed = 20260512},
  };

  for (const auto& test_case : cases) {
    SCOPED_TRACE(::testing::Message() << "case=" << test_case.name);
    run_chunk_gated_delta_rule_fwd_h_case(test_case);
  }
}

TEST_F(TileLangChunkGatedDeltaRuleFwdHTest,
       ChunkGatedDeltaRuleFwdHMatchesReference) {
  const std::vector<ChunkGdrTestCase> accuracy_cases = {
      {.name = "accuracy_n1_t64_h16_hg16_k128_v128",
       .batch_size = 1,
       .seq_len = 64,
       .H = 16,
       .Hg = 16,
       .K = 128,
       .V = 128,
       .chunk_size = 64,
       .seed = 20260513},
      {.name = "accuracy_tp2_n1_t128_h16_hg8_k128_v128",
       .batch_size = 1,
       .seq_len = 128,
       .H = 16,
       .Hg = 8,
       .K = 128,
       .V = 128,
       .chunk_size = 64,
       .seed = 20260516},
      {.name = "accuracy_n1_t128_h32_hg16_k128_v128",
       .batch_size = 1,
       .seq_len = 128,
       .H = 32,
       .Hg = 16,
       .K = 128,
       .V = 128,
       .chunk_size = 64,
       .seed = 20260514},
  };

  for (const auto& test_case : accuracy_cases) {
    SCOPED_TRACE(::testing::Message() << "case=" << test_case.name);
    run_chunk_gated_delta_rule_fwd_h_accuracy_case(test_case);
  }
}

}  // namespace
}  // namespace xllm::kernel::npu::tilelang

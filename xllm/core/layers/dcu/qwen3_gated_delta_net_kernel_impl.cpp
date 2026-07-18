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

// Optimized HIP-kernel backend for Qwen3 gated-delta-net.
//   * causal_conv1d prefill/decode  -> hipfied Tri Dao kernel via adapter.
//   * gated_delta_rule prefill      -> chunked WY decomposition + aiter FLA
//     h-recurrence + chunk_fwd_o.

#include <ATen/ATen.h>
#include <glog/logging.h>

#include <cmath>
#include <tuple>

#include "fla_api.h"
#include "kernels/dcu/causal_conv1d_adapter.h"
#include "qwen3_gated_delta_net_helpers.h"
#include "qwen3_gated_delta_net_impls.h"

namespace xllm {
namespace layer {
namespace qwen3_gdn {
namespace kernel_impl {

namespace {

constexpr int64_t kChunkSize = 64;

// Build [NT_total, 2] i64 tensor of (seq_id, local_chunk_id) rows.
torch::Tensor build_chunk_indices(const torch::Tensor& cu_seqlens_i64,
                                  int64_t chunk_size) {
  const auto opts_long = cu_seqlens_i64.options().dtype(torch::kInt64);
  auto lengths = cu_seqlens_i64.narrow(0, 1, cu_seqlens_i64.size(0) - 1) -
                 cu_seqlens_i64.narrow(0, 0, cu_seqlens_i64.size(0) - 1);
  auto num_chunks = ((lengths + chunk_size - 1) / chunk_size).to(torch::kInt64);
  auto cumsum = num_chunks.cumsum(0);
  int64_t total = cumsum[-1].item<int64_t>();
  auto arange_total = torch::arange(total, opts_long);
  auto zeros = torch::zeros({1}, cumsum.options());
  auto prefix = torch::cat({zeros, cumsum.slice(0, 0, -1)});
  auto repeats_prefix = torch::repeat_interleave(prefix, num_chunks);
  auto local_idx = arange_total - repeats_prefix;
  auto is_start = (local_idx == 0).to(torch::kInt64);
  auto seq_id = is_start.cumsum(0) - 1;
  return torch::stack({seq_id, local_idx}, 1).contiguous();
}

// Cumulative sum of g within each chunk per sequence per head. Returns fp32.
torch::Tensor chunk_local_cumsum_g(const torch::Tensor& g_in,
                                   const std::vector<int64_t>& cu_seqlens_vec,
                                   int64_t chunk_size) {
  auto g_f32 = g_in.to(torch::kFloat32).contiguous();
  auto out = torch::zeros_like(g_f32);
  const int64_t N = static_cast<int64_t>(cu_seqlens_vec.size()) - 1;
  for (int64_t s = 0; s < N; ++s) {
    int64_t bos = cu_seqlens_vec[s];
    int64_t eos = cu_seqlens_vec[s + 1];
    int64_t len = eos - bos;
    if (len == 0) continue;
    auto seq = g_f32.select(0, 0).slice(0, bos, eos);
    int64_t nchunks = (len + chunk_size - 1) / chunk_size;
    int64_t full_end = nchunks * chunk_size;
    if (full_end == len) {
      auto view = seq.view({nchunks, chunk_size, seq.size(-1)});
      out.select(0, 0)
          .slice(0, bos, eos)
          .copy_(view.cumsum(1).reshape({len, seq.size(-1)}));
    } else {
      if (nchunks > 1) {
        int64_t full_len = (nchunks - 1) * chunk_size;
        auto full = seq.slice(0, 0, full_len)
                        .view({nchunks - 1, chunk_size, seq.size(-1)});
        out.select(0, 0)
            .slice(0, bos, bos + full_len)
            .copy_(full.cumsum(1).reshape({full_len, seq.size(-1)}));
      }
      int64_t tail_start = (nchunks - 1) * chunk_size;
      auto tail = seq.slice(0, tail_start, len).cumsum(0);
      out.select(0, 0).slice(0, bos + tail_start, eos).copy_(tail);
    }
  }
  return out;
}

// Per-chunk A[i, j] = beta_i * (k_i . k_j) * exp(g_i - g_j) for i > j.
torch::Tensor chunk_scaled_dot_kkt(const torch::Tensor& k,
                                   const torch::Tensor& beta,
                                   const torch::Tensor& g_cum,
                                   const std::vector<int64_t>& cu_seqlens_vec,
                                   int64_t chunk_size) {
  const int64_t T = k.size(1);
  const int64_t Hv = beta.size(-1);
  auto opts_f32 = k.options().dtype(torch::kFloat32);
  auto A = torch::zeros({1, T, Hv, chunk_size}, opts_f32);
  auto k_v_full = repeat_tensor_heads(k, Hv, 2);
  const int64_t N = static_cast<int64_t>(cu_seqlens_vec.size()) - 1;
  // strict lower-triangular [BT, BT] mask shared across chunks.
  auto tri_full = torch::tril(torch::ones({chunk_size, chunk_size}, opts_f32),
                              /*diagonal=*/-1);
  for (int64_t s = 0; s < N; ++s) {
    int64_t bos = cu_seqlens_vec[s];
    int64_t eos = cu_seqlens_vec[s + 1];
    int64_t len = eos - bos;
    if (len == 0) continue;
    for (int64_t t0 = 0; t0 < len; t0 += chunk_size) {
      int64_t rows = std::min(chunk_size, len - t0);
      auto k_c = k_v_full.select(0, 0).slice(0, bos + t0, bos + t0 + rows);
      auto beta_c = beta.select(0, 0).slice(0, bos + t0, bos + t0 + rows);
      auto g_c = g_cum.select(0, 0).slice(0, bos + t0, bos + t0 + rows);
      auto k_th = k_c.permute({1, 0, 2}).to(torch::kFloat32);
      auto kkT = torch::matmul(k_th, k_th.transpose(1, 2))
                     .permute({1, 0, 2})
                     .contiguous();
      auto g_diff =
          (g_c.unsqueeze(1) - g_c.unsqueeze(0)).permute({0, 2, 1}).contiguous();
      auto mask = tri_full.slice(0, 0, rows).slice(1, 0, rows).unsqueeze(1);
      // safe_exp guards masked-out upper triangle from fp32 overflow.
      auto safe_g = torch::minimum(g_diff, torch::zeros_like(g_diff));
      auto a_chunk = beta_c.unsqueeze(-1) * kkT * torch::exp(safe_g) * mask;
      A.select(0, 0)
          .slice(0, bos + t0, bos + t0 + rows)
          .slice(-1, 0, rows)
          .copy_(a_chunk);
    }
  }
  return A;
}

// Per-chunk (I + A_strict_lower)^{-1} via triangular solve.
torch::Tensor solve_tril_batched(const torch::Tensor& A,
                                 const std::vector<int64_t>& cu_seqlens_vec,
                                 torch::ScalarType out_dtype) {
  const int64_t T = A.size(1);
  const int64_t Hv = A.size(2);
  const int64_t BT = A.size(-1);
  auto opts_f32 = A.options();
  auto out = torch::empty({1, T, Hv, BT}, opts_f32);
  auto eye = torch::eye(BT, opts_f32);
  const int64_t N = static_cast<int64_t>(cu_seqlens_vec.size()) - 1;
  for (int64_t s = 0; s < N; ++s) {
    int64_t bos = cu_seqlens_vec[s];
    int64_t eos = cu_seqlens_vec[s + 1];
    int64_t len = eos - bos;
    if (len == 0) continue;
    for (int64_t t0 = 0; t0 < len; t0 += BT) {
      int64_t rows = std::min(BT, len - t0);
      auto A_full = torch::zeros({BT, Hv, BT}, opts_f32);
      A_full.slice(0, 0, rows)
          .copy_(A.select(0, 0).slice(0, bos + t0, bos + t0 + rows));
      auto M = eye.unsqueeze(0) + A_full.permute({1, 0, 2}).contiguous();
      auto X =
          at::linalg_solve_triangular(M,
                                      eye.unsqueeze(0).expand({Hv, BT, BT}),
                                      /*upper=*/false,
                                      /*left=*/true,
                                      /*unitriangular=*/true);
      out.select(0, 0)
          .slice(0, bos + t0, bos + t0 + rows)
          .copy_(X.permute({1, 0, 2}).contiguous().slice(0, 0, rows));
    }
  }
  return out.to(out_dtype);
}

// w = A_tril @ (beta * exp(g_cum) * k_repeated),   u = A_tril @ (beta * v)
std::pair<torch::Tensor, torch::Tensor> recompute_w_u(
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& beta,
    const torch::Tensor& A_tril,
    const torch::Tensor& g_cum,
    const std::vector<int64_t>& cu_seqlens_vec,
    int64_t chunk_size) {
  const int64_t T = k.size(1);
  const int64_t Hv = v.size(-2);
  const int64_t Kd = k.size(-1);
  const int64_t Vd = v.size(-1);
  auto w = torch::empty({1, T, Hv, Kd}, k.options());
  auto u = torch::empty({1, T, Hv, Vd}, v.options());
  auto k_v_full = repeat_tensor_heads(k, Hv, 2);
  const int64_t N = static_cast<int64_t>(cu_seqlens_vec.size()) - 1;
  for (int64_t s = 0; s < N; ++s) {
    int64_t bos = cu_seqlens_vec[s];
    int64_t eos = cu_seqlens_vec[s + 1];
    int64_t len = eos - bos;
    if (len == 0) continue;
    for (int64_t t0 = 0; t0 < len; t0 += chunk_size) {
      int64_t rows = std::min(chunk_size, len - t0);
      auto A_c = A_tril.select(0, 0)
                     .slice(0, bos + t0, bos + t0 + rows)
                     .slice(-1, 0, rows);
      auto beta_c = beta.select(0, 0).slice(0, bos + t0, bos + t0 + rows);
      auto k_c = k_v_full.select(0, 0).slice(0, bos + t0, bos + t0 + rows);
      auto v_c = v.select(0, 0).slice(0, bos + t0, bos + t0 + rows);
      auto g_c = g_cum.select(0, 0).slice(0, bos + t0, bos + t0 + rows);
      auto A_hrs = A_c.permute({1, 0, 2}).contiguous();
      auto bv = (beta_c.unsqueeze(-1) * v_c)
                    .to(torch::kFloat32)
                    .permute({1, 0, 2})
                    .contiguous();
      auto u_hv = torch::matmul(A_hrs, bv);
      auto weight = beta_c.to(torch::kFloat32) * torch::exp(g_c);
      auto bk = (weight.unsqueeze(-1) * k_c.to(torch::kFloat32))
                    .permute({1, 0, 2})
                    .contiguous();
      auto w_hv = torch::matmul(A_hrs, bk);
      w.select(0, 0)
          .slice(0, bos + t0, bos + t0 + rows)
          .copy_(w_hv.permute({1, 0, 2}).contiguous());
      u.select(0, 0)
          .slice(0, bos + t0, bos + t0 + rows)
          .copy_(u_hv.permute({1, 0, 2}).contiguous());
    }
  }
  return {w, u};
}

// Per-chunk cross-chunk + intra-chunk attention output.
torch::Tensor chunk_fwd_o(const torch::Tensor& q,
                          const torch::Tensor& k,
                          const torch::Tensor& v_new,
                          const torch::Tensor& h,
                          const torch::Tensor& g_cum,
                          double scale,
                          const std::vector<int64_t>& cu_seqlens_vec,
                          int64_t chunk_size) {
  const int64_t T = q.size(1);
  const int64_t Hv = v_new.size(-2);
  const int64_t Vd = v_new.size(-1);
  auto o = torch::empty({1, T, Hv, Vd}, v_new.options());
  auto q_v_full = repeat_tensor_heads(q, Hv, 2);
  auto k_v_full = repeat_tensor_heads(k, Hv, 2);
  auto opts_f32 = q.options().dtype(torch::kFloat32);
  auto tri_full = torch::tril(torch::ones({chunk_size, chunk_size}, opts_f32),
                              /*diagonal=*/0);
  const int64_t N = static_cast<int64_t>(cu_seqlens_vec.size()) - 1;
  int64_t chunk_offset = 0;
  for (int64_t s = 0; s < N; ++s) {
    int64_t bos = cu_seqlens_vec[s];
    int64_t eos = cu_seqlens_vec[s + 1];
    int64_t len = eos - bos;
    if (len == 0) continue;
    int64_t local_chunk = 0;
    for (int64_t t0 = 0; t0 < len; t0 += chunk_size, ++local_chunk) {
      int64_t rows = std::min(chunk_size, len - t0);
      auto q_c = q_v_full.select(0, 0)
                     .slice(0, bos + t0, bos + t0 + rows)
                     .to(torch::kFloat32);
      auto k_c = k_v_full.select(0, 0)
                     .slice(0, bos + t0, bos + t0 + rows)
                     .to(torch::kFloat32);
      auto v_c = v_new.select(0, 0)
                     .slice(0, bos + t0, bos + t0 + rows)
                     .to(torch::kFloat32);
      auto g_c = g_cum.select(0, 0).slice(0, bos + t0, bos + t0 + rows);
      // h_chunk: aiter stores [Hv, V, K] under transpose_state_layout=true;
      // transpose to [Hv, K, V] for q @ h.
      auto h_chunk = h.select(0, 0)
                         .select(0, chunk_offset + local_chunk)
                         .transpose(-1, -2)
                         .to(torch::kFloat32)
                         .contiguous();
      auto q_hv = q_c.permute({1, 0, 2}).contiguous();
      auto oh = torch::matmul(q_hv, h_chunk).permute({1, 0, 2}).contiguous();
      oh = oh * torch::exp(g_c).unsqueeze(-1);
      auto k_hv = k_c.permute({1, 0, 2}).contiguous();
      auto A_qk = torch::matmul(q_hv, k_hv.transpose(1, 2))
                      .permute({1, 0, 2})
                      .contiguous();
      auto g_diff =
          (g_c.unsqueeze(1) - g_c.unsqueeze(0)).permute({0, 2, 1}).contiguous();
      auto mask = tri_full.slice(0, 0, rows).slice(1, 0, rows).unsqueeze(1);
      // safe_exp: clamp g_diff <= 0 to avoid overflow on masked upper triangle.
      auto safe_g = torch::minimum(g_diff, torch::zeros_like(g_diff));
      A_qk = A_qk * torch::exp(safe_g) * mask;
      auto A_hv = A_qk.permute({1, 0, 2}).contiguous();
      auto v_hv = v_c.permute({1, 0, 2}).contiguous();
      auto ov = torch::matmul(A_hv, v_hv).permute({1, 0, 2}).contiguous();
      auto out_c = ((oh + ov) * static_cast<float>(scale)).to(v_new.dtype());
      o.select(0, 0).slice(0, bos + t0, bos + t0 + rows).copy_(out_c);
    }
    chunk_offset += (len + chunk_size - 1) / chunk_size;
  }
  return o;
}

}  // namespace

torch::Tensor causal_conv1d_prefill(const torch::Tensor& flat_input,
                                    const torch::Tensor& conv_weight_2d,
                                    torch::Tensor& conv_cache,
                                    const std::vector<int64_t>& cu_seqlens,
                                    const std::vector<int64_t>& state_indices,
                                    int32_t /*kernel_size*/,
                                    bool activation) {
  // Layer stores conv_weight as [K, C]; DCU kernel expects [C, K].
  auto weight_ck = conv_weight_2d.transpose(0, 1).contiguous();
  return xllm::kernel::dcu::causal_conv1d_varlen_fwd(
      flat_input, weight_ck, cu_seqlens, state_indices, conv_cache, activation);
}

torch::Tensor causal_conv1d_decode(const torch::Tensor& flat_input,
                                   const torch::Tensor& conv_weight_2d,
                                   torch::Tensor& conv_cache,
                                   const torch::Tensor& state_indices,
                                   int32_t /*kernel_size*/,
                                   bool activation) {
  auto weight_ck = conv_weight_2d.transpose(0, 1).contiguous();
  return xllm::kernel::dcu::causal_conv1d_update(
      flat_input,
      weight_ck,
      state_indices.to(torch::kInt32),
      conv_cache,
      activation);
}

std::tuple<torch::Tensor, torch::Tensor> gated_delta_rule_prefill(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor g_in,
    torch::Tensor beta_in,
    std::optional<torch::Tensor> initial_state,
    const std::vector<int64_t>& cu_seqlens_vec,
    bool output_final_state,
    bool use_qk_l2norm_in_kernel) {
  const auto initial_dtype = q.dtype();
  if (use_qk_l2norm_in_kernel) {
    q = l2norm(q, -1, 1e-6);
    k = l2norm(k, -1, 1e-6);
  }
  q = q.contiguous();
  k = k.contiguous();
  v = v.contiguous();
  beta_in = beta_in.contiguous();
  const double scale = 1.0 / std::sqrt(static_cast<double>(k.size(-1)));

  auto cu_seqlens_i64 = torch::tensor(
      cu_seqlens_vec,
      torch::TensorOptions().dtype(torch::kInt64).device(q.device()));
  auto chunk_indices = build_chunk_indices(cu_seqlens_i64, kChunkSize);

  auto g_cum = chunk_local_cumsum_g(g_in, cu_seqlens_vec, kChunkSize);
  auto A = chunk_scaled_dot_kkt(k, beta_in, g_cum, cu_seqlens_vec, kChunkSize);
  auto A_tril = solve_tril_batched(A, cu_seqlens_vec, torch::kFloat32);
  auto [w, u] =
      recompute_w_u(k, v, beta_in, A_tril, g_cum, cu_seqlens_vec, kChunkSize);
  w = w.contiguous();
  u = u.contiguous();
  g_cum = g_cum.contiguous();

  const int64_t N = static_cast<int64_t>(cu_seqlens_vec.size()) - 1;
  std::optional<torch::Tensor> state_idx_opt;
  torch::Tensor sglang_state;
  if (initial_state.has_value()) {
    state_idx_opt = torch::arange(
        N, torch::TensorOptions().dtype(torch::kInt32).device(q.device()));
    // aiter sglang expects fp32 state in [N, H, V, K] layout when
    // transpose_state_layout=true; caller passes fla layout [N, H, K, V].
    sglang_state = initial_state.value()
                       .to(torch::kFloat32)
                       .transpose(-1, -2)
                       .contiguous();
  } else {
    sglang_state = torch::zeros(
        {N, u.size(2), u.size(3), k.size(3)},
        torch::TensorOptions().dtype(torch::kFloat32).device(k.device()));
    state_idx_opt = torch::arange(
        N, torch::TensorOptions().dtype(torch::kInt32).device(q.device()));
  }

  // The sglang variant is chosen because it updates `initial_state` in place;
  // the vllm variant leaves untouched positions of a freshly torch::empty'd
  // final_state uninitialized, poisoning ssm_cache later.
  auto out = aiter::native::chunk_gated_delta_rule_fwd_sglang(
      k,
      w,
      u,
      /*g=*/std::optional<torch::Tensor>(g_cum),
      /*gk=*/std::nullopt,
      std::optional<torch::Tensor>(sglang_state),
      state_idx_opt,
      output_final_state,
      /*chunk_size=*/static_cast<int>(kChunkSize),
      /*save_new_value=*/true,
      /*cu_seqlens=*/std::optional<torch::Tensor>(cu_seqlens_i64),
      /*chunk_indices=*/std::nullopt,
      /*use_exp2=*/false,
      /*transpose_state_layout=*/true);
  CHECK_EQ(out.size(), 2u)
      << "aiter chunk_gated_delta_rule_fwd_sglang returned "
         "unexpected number of tensors";
  auto& h = out[0];
  auto& v_new = out[1];
  // sglang wrote final state in-place into sglang_state; convert layout back
  // to [N, H, K, V] to match the caller's fla convention.
  auto final_state = sglang_state.transpose(-1, -2).contiguous();

  auto o =
      chunk_fwd_o(q, k, v_new, h, g_cum, scale, cu_seqlens_vec, kChunkSize);
  return std::make_tuple(o.to(initial_dtype), final_state);
}

}  // namespace kernel_impl
}  // namespace qwen3_gdn
}  // namespace layer
}  // namespace xllm

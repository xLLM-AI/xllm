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

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <glog/logging.h>

#include <cmath>
#include <cstdint>
#include <sstream>
#include <vector>

#include "core/common/macros.h"
#include "core/kernels/musa/musa_ops_api.h"
#include "core/kernels/param.h"

namespace xllm {
namespace kernel {
namespace cuda {

namespace {

torch::Tensor as_long_indices(const torch::Tensor& indices) {
  if (indices.scalar_type() == torch::kLong) {
    return indices.is_contiguous() ? indices : indices.contiguous();
  }
  return indices.to(torch::kLong).contiguous();
}

inline torch::Tensor l2norm_last(const torch::Tensor& x, double eps) {
  return x / (x.pow(2).sum(-1, /*keepdim=*/true) + eps).sqrt();
}

torch::Tensor gated_layer_norm_ref(GatedLayerNormParams& params) {
  const auto x_shape_og = params.x.sizes();
  const int64_t last_dim = params.x.size(-1);
  auto x_2d = params.x.reshape({-1, last_dim});
  const int64_t M = x_2d.size(0);
  const int64_t N = x_2d.size(1);
  const int64_t group_size_val =
      params.group_size > 0 ? params.group_size : last_dim;
  CHECK(N % group_size_val == 0)
      << "gated_layer_norm: N must be divisible by group_size";
  const int64_t ngroups = N / group_size_val;

  torch::Tensor z_2d;
  if (params.z.has_value() && params.z.value().defined()) {
    z_2d = params.z.value().reshape({-1, last_dim});
  }

  torch::Tensor x_input = x_2d;
  if (z_2d.defined() && !params.norm_before_gate) {
    x_input = x_2d * (z_2d * torch::sigmoid(z_2d));
  }

  auto x_grouped = x_input.unfold(1, group_size_val, group_size_val);
  auto x_grouped_flat = x_grouped.reshape({-1, group_size_val});

  torch::Tensor x_norm_flat;
  if (!params.is_rms_norm) {
    x_norm_flat = torch::layer_norm(x_grouped_flat,
                                    {group_size_val},
                                    torch::Tensor(),
                                    torch::Tensor(),
                                    params.eps);
  } else {
    auto mean_sq = x_grouped_flat.pow(2).mean(-1, /*keepdim=*/true);
    x_norm_flat = x_grouped_flat * torch::rsqrt(mean_sq + params.eps);
  }

  auto x_norm = x_norm_flat.reshape({M, ngroups, group_size_val})
                    .contiguous()
                    .view({M, N});
  auto y = x_norm * params.weight.to(x_norm.dtype());
  if (params.bias.defined()) {
    y = y + params.bias.to(y.dtype());
  }
  if (z_2d.defined() && params.norm_before_gate) {
    y = y * (z_2d.to(y.dtype()) * torch::sigmoid(z_2d.to(y.dtype())));
  }
  return y.reshape(x_shape_og);
}

inline torch::Tensor recurrent_gdn_step(torch::Tensor& state,
                                        const torch::Tensor& q_t,
                                        const torch::Tensor& k_t,
                                        const torch::Tensor& v_t,
                                        const torch::Tensor& g_t,
                                        const torch::Tensor& beta_t) {
  auto g_exp = g_t.exp().unsqueeze(-1).unsqueeze(-1);
  state.mul_(g_exp);
  auto kv_mem = (state * k_t.unsqueeze(-1)).sum(-2);
  auto delta = (v_t - kv_mem) * beta_t.unsqueeze(-1);
  state.add_(k_t.unsqueeze(-1) * delta.unsqueeze(-2));
  return (state * q_t.unsqueeze(-1)).sum(-2);
}

}  // namespace

torch::Tensor l2_norm(torch::Tensor& x, double eps) {
  return x / (x.pow(2).sum(-1, /*keepdim=*/true) + eps).sqrt();
}

std::pair<torch::Tensor, torch::Tensor> fused_gdn_gating(
    FusedGdnGatingParams& params) {
  const auto& A_log = params.A_log;
  const auto& a = params.a;
  const auto& b = params.b;
  const auto& dt_bias = params.dt_bias;
  const auto orig_dtype = a.scalar_type();

  if (!a.is_cpu() &&
      (orig_dtype == torch::kFloat32 || orig_dtype == torch::kBFloat16) &&
      b.scalar_type() == orig_dtype) {
    return gdn_gating(a, b, A_log, dt_bias, params.beta, params.threshold);
  }

  auto a_f32 = a.to(torch::kFloat32);
  auto b_f32 = b.to(torch::kFloat32);
  auto A_log_f32 = A_log.to(torch::kFloat32);
  auto dt_bias_f32 = dt_bias.to(torch::kFloat32);

  auto pre = a_f32 + dt_bias_f32.unsqueeze(0);
  auto sp = torch::nn::functional::softplus(
      pre,
      torch::nn::functional::SoftplusFuncOptions()
          .beta(params.beta)
          .threshold(params.threshold));
  auto g_f32 = -torch::exp(A_log_f32).unsqueeze(0) * sp;
  auto beta_f32 = torch::sigmoid(b_f32);

  return {g_f32.to(orig_dtype), beta_f32.to(orig_dtype)};
}

torch::Tensor causal_conv1d_update(CausalConv1dUpdateParams& params) {
  auto x = params.x;
  auto weight = params.weight;
  if (weight.dim() == 3) {
    CHECK(weight.size(1) == 1)
        << "causal_conv1d_update: expected weight [dim, 1, width]";
    weight = weight.squeeze(1);
  }
  CHECK(weight.dim() == 2)
      << "causal_conv1d_update: expected weight [dim, width]";
  CHECK(params.conv_state.dim() == 3)
      << "causal_conv1d_update: expected conv_state "
         "[num_cache_lines, dim, state_len]";
  CHECK(params.conv_state_indices.has_value())
      << "causal_conv1d_update: conv_state_indices is required";
  CHECK(params.query_start_loc.has_value())
      << "causal_conv1d_update: query_start_loc is required";

  const int64_t dim = weight.size(0);
  const int64_t width = weight.size(1);
  const int64_t state_len = width - 1;
  CHECK(params.conv_state.size(1) == dim)
      << "causal_conv1d_update: conv_state dim mismatch";
  CHECK(params.conv_state.size(2) == state_len)
      << "causal_conv1d_update: conv_state state_len mismatch";

  const auto& cache_indices_raw = params.conv_state_indices.value();
  const int64_t batch = cache_indices_raw.size(0);
  const int64_t conv_num_tokens_pre = x.size(0);

  if (params.output_buf.has_value() && params.output_buf->defined() &&
      conv_num_tokens_pre == batch && state_len > 0 && width >= 2 &&
      width <= 5 && cache_indices_raw.scalar_type() == torch::kInt32 &&
      cache_indices_raw.is_contiguous()) {
    causal_conv1d_decode_fused(x,
                               weight,
                               params.bias,
                               params.conv_state,
                               cache_indices_raw,
                               *params.output_buf,
                               static_cast<int>(params.pad_slot_id),
                               params.activation);
    return *params.output_buf;
  }

  auto weight_f32 = weight.to(torch::kFloat32);
  auto x_f32 = x.to(torch::kFloat32);
  auto out = torch::empty_like(x_f32);
  std::optional<torch::Tensor> bias_f32;
  if (params.bias.has_value() && params.bias.value().defined()) {
    bias_f32 = params.bias.value().to(torch::kFloat32);
  }

  const auto& cache_indices = cache_indices_raw.contiguous();
  const auto& query_start_loc = params.query_start_loc.value().contiguous();

  const int64_t conv_num_tokens = x_f32.size(0);
  if (conv_num_tokens == batch && state_len > 0) {
    auto idx = as_long_indices(cache_indices);
    auto history = params.conv_state.index_select(0, idx).to(torch::kFloat32);
    auto window = torch::cat({history, x_f32.unsqueeze(-1)}, /*dim=*/-1);
    auto token_out = (window * weight_f32.unsqueeze(0)).sum(-1);
    if (bias_f32.has_value()) {
      token_out = token_out + bias_f32.value();
    }
    if (params.activation) {
      token_out = torch::silu(token_out);
    }
    auto new_state = window.narrow(/*dim=*/-1, 1, state_len).contiguous();
    params.conv_state.index_copy_(
        0, idx, new_state.to(params.conv_state.scalar_type()));
    return token_out.to(x.scalar_type());
  }

  auto cache_indices_cpu =
      cache_indices.to(torch::kCPU, torch::kLong).contiguous();
  auto query_start_loc_cpu =
      query_start_loc.to(torch::kCPU, torch::kLong).contiguous();
  auto cache_indices_acc = cache_indices_cpu.accessor<int64_t, 1>();
  auto query_start_loc_acc = query_start_loc_cpu.accessor<int64_t, 1>();

  for (int64_t seq = 0; seq < batch; ++seq) {
    const int64_t cache_idx = cache_indices_acc[seq];
    if (cache_idx == params.pad_slot_id) {
      continue;
    }
    const int64_t start = query_start_loc_acc[seq];
    const int64_t end = query_start_loc_acc[seq + 1];
    auto history = params.conv_state[cache_idx].to(torch::kFloat32).clone();

    for (int64_t token_idx = start; token_idx < end; ++token_idx) {
      auto x_t = x_f32[token_idx];
      torch::Tensor token_out;
      if (state_len == 0) {
        token_out = weight_f32.select(1, 0) * x_t;
      } else {
        auto window = torch::cat({history, x_t.unsqueeze(-1)}, /*dim=*/-1);
        token_out = (window * weight_f32).sum(-1);
      }
      if (bias_f32.has_value()) {
        token_out = token_out + bias_f32.value();
      }
      if (params.activation) {
        token_out = torch::silu(token_out);
      }
      out[token_idx] = token_out;
      if (state_len > 0) {
        if (state_len == 1) {
          history = x_t.unsqueeze(-1);
        } else {
          history = torch::cat(
              {history.slice(/*dim=*/-1, /*start=*/1, /*end=*/state_len),
               x_t.unsqueeze(-1)},
              /*dim=*/-1);
        }
      }
    }
    params.conv_state[cache_idx] = history.to(params.conv_state.scalar_type());
  }

  return out.to(x.scalar_type());
}

torch::Tensor gated_layer_norm(GatedLayerNormParams& params) {
  if (params.output_buf.has_value() && params.output_buf->defined() &&
      params.is_rms_norm && params.norm_before_gate && params.z.has_value() &&
      params.z.value().defined() && !params.bias.defined()) {
    const int64_t last_dim = params.x.size(-1);
    const int64_t group_size_val =
        params.group_size > 0 ? params.group_size : last_dim;
    const bool dtype_ok = params.x.scalar_type() == torch::kBFloat16 ||
                          params.x.scalar_type() == torch::kHalf ||
                          params.x.scalar_type() == torch::kFloat32;
    const torch::Tensor& z = params.z.value();
    if (group_size_val == last_dim && dtype_ok &&
        params.x.scalar_type() == z.scalar_type() &&
        params.x.scalar_type() == params.weight.scalar_type() &&
        params.x.scalar_type() == params.output_buf->scalar_type() &&
        params.x.is_contiguous() && z.is_contiguous() &&
        params.output_buf->is_contiguous() && params.weight.is_contiguous() &&
        params.weight.dim() == 1) {
      auto x_2d = params.x.reshape({-1, last_dim});
      auto z_2d = z.reshape({-1, last_dim});
      auto out_2d = params.output_buf->reshape({-1, last_dim});
      gated_rms_norm_fused(x_2d, params.weight, z_2d, out_2d, params.eps);
      return params.output_buf->reshape(params.x.sizes());
    }
  }
  return gated_layer_norm_ref(params);
}

std::pair<torch::Tensor, torch::Tensor> partial_rotary_embedding(
    PartialRotaryEmbeddingParams& params) {
  partial_rotary_embedding_inplace(params.positions,
                                   params.query,
                                   params.key,
                                   params.cos_sin_cache,
                                   params.head_size,
                                   params.rotary_dim,
                                   params.is_neox_style);
  return {params.query, params.key};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fused_qkvzba_split_reshape_cat(FusedQkvzbaSplitReshapeParams& params) {
  const int64_t nk = static_cast<int64_t>(params.num_heads_qk);
  const int64_t nv = static_cast<int64_t>(params.num_heads_v);
  const int64_t hk = static_cast<int64_t>(params.head_qk);
  const int64_t hv = static_cast<int64_t>(params.head_v);
  CHECK(nk > 0 && nv > 0 && nv % nk == 0)
      << "fused_qkvzba_split_reshape_cat: invalid head counts nk=" << nk
      << " nv=" << nv;
  const int64_t vpk = nv / nk;
  const int64_t per_group = 2 * hk + 2 * vpk * hv;

  const auto& qkvz = params.mixed_qkvz;
  const int64_t n = qkvz.numel() / qkvz.size(-1);
  CHECK(qkvz.size(-1) == nk * per_group)
      << "fused_qkvzba_split_reshape_cat: mixed_qkvz last dim mismatch, got "
      << qkvz.size(-1) << " expected " << nk * per_group;

  auto qkvz_g = qkvz.reshape({n, nk, per_group});
  auto q = qkvz_g.slice(-1, 0, hk);
  auto k = qkvz_g.slice(-1, hk, 2 * hk);
  auto v = qkvz_g.slice(-1, 2 * hk, 2 * hk + vpk * hv);
  auto z = qkvz_g.slice(-1, 2 * hk + vpk * hv, per_group);

  const auto& ba = params.mixed_ba;
  CHECK(ba.size(-1) == 2 * nv)
      << "fused_qkvzba_split_reshape_cat: mixed_ba last dim mismatch, got "
      << ba.size(-1) << " expected " << 2 * nv;
  auto ba_g = ba.reshape({n, nk, 2 * vpk});
  auto b_slice = ba_g.slice(-1, 0, vpk);
  auto a_slice = ba_g.slice(-1, vpk, 2 * vpk);

  const int64_t qkv_dim = 2 * nk * hk + nv * hv;
  const int64_t z_dim = nv * hv;

  if (params.mixed_qkv_out_buf.defined() && params.z_out_buf.defined() &&
      params.b_out_buf.defined() && params.a_out_buf.defined() &&
      params.mixed_qkv_out_buf.size(0) >= n &&
      params.mixed_qkv_out_buf.size(1) == qkv_dim &&
      params.z_out_buf.size(0) >= n && params.z_out_buf.size(1) == z_dim &&
      params.b_out_buf.size(0) >= n && params.b_out_buf.size(1) == nv &&
      params.a_out_buf.size(0) >= n && params.a_out_buf.size(1) == nv) {
    auto mixed_qkv_buf =
        params.mixed_qkv_out_buf.narrow(/*dim=*/0, /*start=*/0, /*length=*/n);
    auto z_buf = params.z_out_buf.narrow(/*dim=*/0, /*start=*/0, /*length=*/n);
    auto b_buf = params.b_out_buf.narrow(/*dim=*/0, /*start=*/0, /*length=*/n);
    auto a_buf = params.a_out_buf.narrow(/*dim=*/0, /*start=*/0, /*length=*/n);

    mixed_qkv_buf.narrow(/*dim=*/1, /*start=*/0, /*length=*/nk * hk)
        .view({n, nk, hk})
        .copy_(q);
    mixed_qkv_buf.narrow(/*dim=*/1, /*start=*/nk * hk, /*length=*/nk * hk)
        .view({n, nk, hk})
        .copy_(k);
    mixed_qkv_buf
        .narrow(/*dim=*/1,
                /*start=*/2 * nk * hk,
                /*length=*/nv * hv)
        .view({n, nk, vpk * hv})
        .copy_(v);
    z_buf.view({n, nk, vpk * hv}).copy_(z);
    b_buf.view({n, nk, vpk}).copy_(b_slice);
    a_buf.view({n, nk, vpk}).copy_(a_slice);

    return {mixed_qkv_buf, z_buf, b_buf, a_buf};
  }

  auto q_flat = q.reshape({n, nk * hk}).contiguous();
  auto k_flat = k.reshape({n, nk * hk}).contiguous();
  auto v_flat = v.reshape({n, nv * hv}).contiguous();
  auto z_flat = z.reshape({n, nv * hv}).contiguous();

  auto mixed_qkv = torch::cat({q_flat, k_flat, v_flat}, -1).contiguous();

  auto b = b_slice.reshape({n, nv}).contiguous();
  auto a = a_slice.reshape({n, nv}).contiguous();

  return {mixed_qkv, z_flat, b, a};
}

torch::Tensor recurrent_gated_delta_rule(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    torch::Tensor& state,
    const std::optional<torch::Tensor>& beta,
    const std::optional<double> scale,
    const std::optional<torch::Tensor>& actual_seq_lengths,
    const std::optional<torch::Tensor>& ssm_state_indices,
    const std::optional<torch::Tensor>& /*num_accepted_tokens*/,
    const std::optional<torch::Tensor>& g,
    const std::optional<torch::Tensor>& /*gk*/) {
  CHECK(scale.has_value()) << "recurrent_gated_delta_rule requires scale";
  CHECK(g.has_value()) << "recurrent_gated_delta_rule requires g";
  CHECK(beta.has_value()) << "recurrent_gated_delta_rule requires beta";

  const auto orig_dtype = value.scalar_type();
  auto q = query.to(torch::kFloat32);
  auto k = key.to(torch::kFloat32);
  auto v = value.to(torch::kFloat32);

  const int64_t Hqk = q.size(1);
  const int64_t Hv = v.size(1);
  if (Hqk != Hv) {
    CHECK(Hv % Hqk == 0) << "recurrent_gated_delta_rule: Hv (" << Hv
                         << ") must be a multiple of Hqk (" << Hqk
                         << ") for GQA expansion";
    const int64_t repeat = Hv / Hqk;
    q = q.repeat_interleave(repeat, /*dim=*/1);
    k = k.repeat_interleave(repeat, /*dim=*/1);
  }
  auto g_in = g.value().to(torch::kFloat32);
  auto beta_in = beta.value().to(torch::kFloat32);
  const double sc = scale.value();
  const double l2_eps = 1e-6;

  const int64_t num_tokens = q.size(0);
  auto out = torch::empty({num_tokens, q.size(1), v.size(-1)},
                          q.options().dtype(torch::kFloat32));

  int64_t batch_size = num_tokens;
  if (ssm_state_indices.has_value()) {
    batch_size = ssm_state_indices.value().size(0);
  }

  q = l2_norm(q, l2_eps) * static_cast<float>(sc);
  k = l2_norm(k, l2_eps);

  if (ssm_state_indices.has_value() && num_tokens == batch_size) {
    auto idx = as_long_indices(ssm_state_indices.value());
    auto st = state.index_select(0, idx)
                  .to(torch::kFloat32)
                  .transpose(-1, -2)
                  .contiguous();
    auto g_exp = g_in.exp().unsqueeze(-1).unsqueeze(-1);
    st = st * g_exp;
    auto kv_mem = (st * k.unsqueeze(-1)).sum(-2);
    auto delta = (v - kv_mem) * beta_in.unsqueeze(-1);
    st = st + k.unsqueeze(-1) * delta.unsqueeze(-2);
    auto out_v = (st * q.unsqueeze(-1)).sum(-2);
    state.index_copy_(
        0, idx, st.transpose(-1, -2).contiguous().to(state.scalar_type()));
    return out_v.to(orig_dtype);
  }

  std::optional<torch::Tensor> actual_seq_lengths_cpu;
  if (actual_seq_lengths.has_value()) {
    actual_seq_lengths_cpu =
        actual_seq_lengths.value().to(torch::kCPU, torch::kLong).contiguous();
  }
  std::optional<torch::Tensor> ssm_state_indices_cpu;
  if (ssm_state_indices.has_value()) {
    ssm_state_indices_cpu =
        ssm_state_indices.value().to(torch::kCPU, torch::kLong).contiguous();
  }
  const int64_t* actual_seq_lengths_data =
      actual_seq_lengths_cpu.has_value()
          ? actual_seq_lengths_cpu->data_ptr<int64_t>()
          : nullptr;
  const int64_t* ssm_state_indices_data =
      ssm_state_indices_cpu.has_value()
          ? ssm_state_indices_cpu->data_ptr<int64_t>()
          : nullptr;

  for (int64_t b = 0; b < batch_size; ++b) {
    int64_t start = b;
    int64_t end = b + 1;
    if (actual_seq_lengths_data != nullptr) {
      start = actual_seq_lengths_data[b];
      end = actual_seq_lengths_data[b + 1];
    }

    int64_t slot = b;
    if (ssm_state_indices_data != nullptr) {
      slot = ssm_state_indices_data[b];
    }

    auto st = state[slot].to(torch::kFloat32).transpose(-1, -2).contiguous();

    for (int64_t t = start; t < end; ++t) {
      out[t] = recurrent_gdn_step(st, q[t], k[t], v[t], g_in[t], beta_in[t]);
    }

    state[slot] = st.transpose(-1, -2).to(state.scalar_type());
  }

  return out.to(orig_dtype);
}

namespace {

std::string mate_gdn_dtype_suffix(torch::ScalarType dtype) {
  if (dtype == torch::kBFloat16) {
    return "bf16";
  }
  if (dtype == torch::kFloat16) {
    return "f16";
  }
  LOG(FATAL) << "mate GDN decode expects bfloat16 or float16 q/k/v";
}

}  // namespace

std::string get_mate_gdn_decode_uri(int64_t num_q_heads,
                                    int64_t num_v_heads,
                                    torch::ScalarType dtype) {
  std::ostringstream oss;
  oss << "mate_gdn_decode_hq" << num_q_heads << "_hv" << num_v_heads << "_"
      << mate_gdn_dtype_suffix(dtype);
  return oss.str();
}

torch::Tensor mate_gated_delta_rule_decode(
    MateGatedDeltaRuleDecodeParams& params) {
  auto mixed_qkv = params.mixed_qkv.contiguous();
  CHECK(mixed_qkv.dim() == 2) << "mate GDN decode expects mixed_qkv [B, D]";
  const int64_t batch_size = mixed_qkv.size(0);
  const int64_t num_k_heads = params.num_k_heads;
  const int64_t num_v_heads = params.num_v_heads;
  const int64_t head_k_dim = params.head_k_dim;
  const int64_t head_v_dim = params.head_v_dim;
  const int64_t qk_cols = num_k_heads * head_k_dim;
  const int64_t v_cols = num_v_heads * head_v_dim;
  CHECK(mixed_qkv.size(1) == 2 * qk_cols + v_cols)
      << "mate GDN decode mixed_qkv dim mismatch";

  auto query = mixed_qkv.slice(/*dim=*/1, /*start=*/0, /*end=*/qk_cols)
                   .reshape({batch_size, num_k_heads, head_k_dim})
                   .contiguous();
  auto key = mixed_qkv.slice(/*dim=*/1, /*start=*/qk_cols, /*end=*/2 * qk_cols)
                 .reshape({batch_size, num_k_heads, head_k_dim})
                 .contiguous();
  auto value =
      mixed_qkv
          .slice(/*dim=*/1, /*start=*/2 * qk_cols, /*end=*/2 * qk_cols + v_cols)
          .reshape({batch_size, num_v_heads, head_v_dim})
          .contiguous();

  auto a = params.a.contiguous();
  auto b = params.b.contiguous();
  if (a.dim() == 1) {
    a = a.unsqueeze(0);
  }
  if (b.dim() == 1) {
    b = b.unsqueeze(0);
  }
  CHECK(a.dim() == 2 && b.dim() == 2)
      << "mate GDN decode expects a/b shaped [B, Hv]";

  auto state_f32 = params.state.to(torch::kFloat32).contiguous();
  auto state_indices = params.state_indices.to(torch::kInt32).contiguous();
  auto output =
      params.decode_output.has_value() && params.decode_output.value().defined()
          ? params.decode_output.value()
          : torch::empty({batch_size, num_v_heads, head_v_dim},
                         value.options());

  const std::string uri =
      get_mate_gdn_decode_uri(num_k_heads, num_v_heads, query.scalar_type());
  bind_tvmffi_stream_to_current_torch_stream(query.device());
  auto run = get_function(uri, "run");

  run(to_ffi_tensor(query),
      to_ffi_tensor(key),
      to_ffi_tensor(value),
      to_ffi_tensor(params.A_log.contiguous()),
      to_ffi_tensor(a),
      to_ffi_tensor(params.dt_bias.contiguous()),
      to_ffi_tensor(b),
      to_ffi_tensor(state_indices),
      to_ffi_tensor(state_f32),
      to_ffi_tensor(output));

  auto updated_state =
      state_f32.index_select(/*dim=*/0, state_indices.to(torch::kLong));
  params.state.index_copy_(
      /*dim=*/0,
      state_indices.to(torch::kLong),
      updated_state.to(params.state.scalar_type()));

  return output;
}

namespace {

torch::Tensor l2norm_dim(const torch::Tensor& x, int64_t dim, double eps) {
  auto norm = torch::sqrt(torch::sum(torch::square(x), dim, true) + eps);
  return x / norm;
}

torch::Tensor repeat_heads(const torch::Tensor& t,
                           int64_t target_heads,
                           int64_t head_dim) {
  const int64_t cur = t.size(head_dim);
  if (cur == target_heads) {
    return t;
  }
  const int64_t repeats = target_heads / cur;
  std::vector<int64_t> view_shape = t.sizes().vec();
  view_shape.insert(view_shape.begin() + head_dim + 1, 1);
  std::vector<int64_t> expand_shape = view_shape;
  expand_shape[head_dim + 1] = repeats;
  std::vector<int64_t> out_shape = t.sizes().vec();
  out_shape[head_dim] = target_heads;
  return t.unsqueeze(head_dim + 1)
      .expand(expand_shape)
      .reshape(out_shape)
      .contiguous();
}

std::tuple<torch::Tensor, torch::Tensor> recurrent_one(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor g,
    torch::Tensor beta,
    std::optional<torch::Tensor> initial_state,
    bool use_qk_l2norm_in_kernel) {
  auto initial_dtype = query.dtype();
  if (use_qk_l2norm_in_kernel) {
    query = l2norm_dim(query, -1, 1e-6);
    key = l2norm_dim(key, -1, 1e-6);
  }
  auto tf = [](torch::Tensor x) {
    return x.transpose(1, 2).contiguous().to(torch::kFloat32);
  };
  query = tf(query);
  key = tf(key);
  value = tf(value);
  beta = tf(beta);
  g = tf(g);
  const int64_t vh = value.size(1);
  query = repeat_heads(query, vh, 1);
  key = repeat_heads(key, vh, 1);
  const int64_t bsz = key.size(0);
  const int64_t nh = key.size(1);
  const int64_t sl = key.size(2);
  const int64_t kd = key.size(3);
  const int64_t vd = value.size(3);
  const float scale_val = 1.0f / std::sqrt(static_cast<float>(query.size(-1)));
  query = query * scale_val;
  auto core = torch::zeros(
      {bsz, nh, sl, vd},
      torch::TensorOptions().dtype(torch::kFloat32).device(value.device()));
  torch::Tensor state;
  if (!initial_state.has_value()) {
    state = torch::zeros(
        {bsz, nh, kd, vd},
        torch::TensorOptions().dtype(torch::kFloat32).device(value.device()));
  } else {
    state = initial_state.value().to(value.device(), torch::kFloat32);
  }
  for (int64_t i = 0; i < sl; ++i) {
    auto q_t = query.select(2, i);
    auto k_t = key.select(2, i);
    auto v_t = value.select(2, i);
    auto g_t = g.select(2, i).exp().unsqueeze(-1).unsqueeze(-1);
    auto beta_t = beta.select(2, i).unsqueeze(-1);
    state = state * g_t;
    auto kv_mem = torch::sum(state * k_t.unsqueeze(-1), -2);
    auto delta = (v_t - kv_mem) * beta_t;
    state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2);
    core.select(2, i) = torch::sum(state * q_t.unsqueeze(-1), -2);
  }
  core = core.transpose(1, 2).contiguous().to(initial_dtype);
  return std::make_tuple(core, state);
}

}  // namespace

torch::Tensor fused_sigmoid_gating_delta_rule_update(
    FusedSigmoidGatingDeltaRuleUpdateParams& params) {
  auto a_f = params.a.to(torch::kFloat32);
  auto b_f = params.b.to(torch::kFloat32);
  auto dt_bias_f = params.dt_bias.to(torch::kFloat32);
  auto A_log_f = params.A_log.to(torch::kFloat32);
  auto beta = torch::sigmoid(b_f);
  auto sp = torch::nn::functional::softplus(
      a_f + dt_bias_f,
      torch::nn::functional::SoftplusFuncOptions()
          .beta(params.softplus_beta)
          .threshold(params.softplus_threshold));
  auto g = (-(A_log_f.exp()) * sp).to(params.q.dtype());
  beta = beta.to(params.q.dtype());

  auto idx_cpu =
      params.initial_state_indices.to(torch::kCPU, torch::kLong).contiguous();
  auto idx_acc = idx_cpu.accessor<int64_t, 1>();
  const int64_t bsz = params.q.size(0);
  auto& ssm = params.initial_state_source;
  std::vector<torch::Tensor> cores;
  cores.reserve(static_cast<size_t>(bsz));
  for (int64_t i = 0; i < bsz; ++i) {
    const int64_t slot = idx_acc[i];
    auto state = ssm.narrow(0, slot, 1).to(torch::kFloat32);
    auto q_i = params.q.narrow(0, i, 1);
    auto k_i = params.k.narrow(0, i, 1);
    auto v_i = params.v.narrow(0, i, 1);
    auto g_i = g.narrow(0, i, 1);
    auto beta_i = beta.narrow(0, i, 1);
    auto [core_i, state_i] = recurrent_one(
        q_i, k_i, v_i, g_i, beta_i, state, params.use_qk_l2norm_in_kernel);
    ssm.narrow(0, slot, 1).copy_(state_i.to(ssm.dtype()));
    cores.push_back(core_i);
  }
  return torch::cat(cores, 0).contiguous();
}

namespace {

template <typename T>
__device__ __forceinline__ float to_f32(T v) {
  if constexpr (std::is_same_v<T, __half>) {
    return __half2float(v);
  } else if constexpr (std::is_same_v<T, __nv_bfloat16> ||
                       std::is_same_v<T, __mt_bfloat16>) {
    return __bfloat162float(v);
  } else {
    return static_cast<float>(v);
  }
}

template <typename T>
__device__ __forceinline__ T from_f32(float v) {
  if constexpr (std::is_same_v<T, __half>) {
    return __float2half_rn(v);
  } else if constexpr (std::is_same_v<T, __nv_bfloat16> ||
                       std::is_same_v<T, __mt_bfloat16>) {
    return __float2bfloat16_rn(v);
  } else {
    return static_cast<T>(v);
  }
}

__device__ __forceinline__ float silu_f32(float x) {
  return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ float fast_sigmoid_f32(float z) {
  constexpr float kLog2e = 1.4426950408889634f;
  return 1.0f / (1.0f + exp2f(-z * kLog2e));
}

}  // namespace

namespace {

template <typename scalar_t>
__global__ void gdn_gating_kernel(const scalar_t* __restrict__ a,
                                  const scalar_t* __restrict__ b,
                                  const float* __restrict__ A_log,
                                  const float* __restrict__ dt_bias,
                                  scalar_t* __restrict__ g_out,
                                  scalar_t* __restrict__ beta_out,
                                  int64_t n_elem,
                                  int H,
                                  float sp_beta,
                                  float threshold) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n_elem) {
    return;
  }
  const int h = static_cast<int>(idx % H);

  const float pre = static_cast<float>(a[idx]) + dt_bias[h];
  const float bx = sp_beta * pre;
  const float sp = (bx > threshold) ? pre : (log1pf(expf(bx)) / sp_beta);
  const float g = -expf(A_log[h]) * sp;
  const float beta = 1.f / (1.f + expf(-static_cast<float>(b[idx])));

  g_out[idx] = static_cast<scalar_t>(g);
  beta_out[idx] = static_cast<scalar_t>(beta);
}

template <typename scalar_t>
void launch(const torch::Tensor& a,
            const torch::Tensor& b,
            const torch::Tensor& A_log_f32,
            const torch::Tensor& dt_bias_f32,
            torch::Tensor& g,
            torch::Tensor& beta,
            int64_t n_elem,
            int H,
            float sp_beta,
            float threshold,
            cudaStream_t stream) {
  const int threads = 256;
  const int blocks = static_cast<int>((n_elem + threads - 1) / threads);
  gdn_gating_kernel<scalar_t>
      <<<blocks, threads, 0, stream>>>(a.data_ptr<scalar_t>(),
                                       b.data_ptr<scalar_t>(),
                                       A_log_f32.data_ptr<float>(),
                                       dt_bias_f32.data_ptr<float>(),
                                       g.data_ptr<scalar_t>(),
                                       beta.data_ptr<scalar_t>(),
                                       n_elem,
                                       H,
                                       sp_beta,
                                       threshold);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

std::pair<torch::Tensor, torch::Tensor> gdn_gating(const torch::Tensor& a,
                                                   const torch::Tensor& b,
                                                   const torch::Tensor& A_log,
                                                   const torch::Tensor& dt_bias,
                                                   double sp_beta,
                                                   double threshold) {
  const int64_t H = a.size(-1);
  const int64_t n_elem = a.numel();
  auto a_c = a.contiguous();
  auto b_c = b.contiguous();
  auto A_log_f32 = A_log.to(torch::kFloat32).contiguous();
  auto dt_bias_f32 = dt_bias.to(torch::kFloat32).contiguous();
  auto g = torch::empty_like(a_c);
  auto beta = torch::empty_like(a_c);

  const at::cuda::OptionalCUDAGuard guard(device_of(a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (a_c.scalar_type() == torch::kFloat32) {
    launch<float>(a_c,
                  b_c,
                  A_log_f32,
                  dt_bias_f32,
                  g,
                  beta,
                  n_elem,
                  static_cast<int>(H),
                  static_cast<float>(sp_beta),
                  static_cast<float>(threshold),
                  stream);
  } else if (a_c.scalar_type() == torch::kBFloat16) {
    launch<at::BFloat16>(a_c,
                         b_c,
                         A_log_f32,
                         dt_bias_f32,
                         g,
                         beta,
                         n_elem,
                         static_cast<int>(H),
                         static_cast<float>(sp_beta),
                         static_cast<float>(threshold),
                         stream);
  } else {
    LOG(FATAL) << "gdn_gating: unsupported dtype " << a_c.scalar_type();
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {g, beta};
}

namespace {

template <typename T, int BLOCK_THREADS>
__global__ void __launch_bounds__(BLOCK_THREADS, 1)
    gated_rms_norm_kernel(const T* __restrict__ x,
                          const T* __restrict__ w,
                          const T* __restrict__ z,
                          T* __restrict__ y,
                          int64_t x_stride_row,
                          int64_t z_stride_row,
                          int64_t y_stride_row,
                          int N,
                          float eps,
                          float inv_N) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const T* __restrict__ x_row = x + static_cast<int64_t>(row) * x_stride_row;
  const T* __restrict__ z_row = z + static_cast<int64_t>(row) * z_stride_row;
  T* __restrict__ y_row = y + static_cast<int64_t>(row) * y_stride_row;

  float local_sum = 0.0f;
  for (int n = tid; n < N; n += BLOCK_THREADS) {
    const float x_val = to_f32<T>(x_row[n]);
    local_sum += x_val * x_val;
  }

  __shared__ float reduce[BLOCK_THREADS];
  reduce[tid] = local_sum;
  __syncthreads();
#pragma unroll
  for (int stride = BLOCK_THREADS / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }

  __shared__ float inv_rms_shared;
  if (tid == 0) {
    inv_rms_shared = rsqrtf(reduce[0] * inv_N + eps);
  }
  __syncthreads();
  const float inv_rms = inv_rms_shared;

  for (int n = tid; n < N; n += BLOCK_THREADS) {
    const float x_val = to_f32<T>(x_row[n]);
    const float w_val = to_f32<T>(w[n]);
    const float z_val = to_f32<T>(z_row[n]);
    const float normed = x_val * inv_rms * w_val;
    const float gate = z_val * fast_sigmoid_f32(z_val);
    y_row[n] = from_f32<T>(normed * gate);
  }
}

inline int pick_block_threads(int N) {
  int bt = 32;
  while (bt < N && bt < 1024) {
    bt <<= 1;
  }
  return bt;
}

#define GATED_RMSNORM_DISPATCH_BLOCK(T_TYPE, BT)                \
  do {                                                          \
    gated_rms_norm_kernel<T_TYPE, BT><<<rows, BT, 0, stream>>>( \
        reinterpret_cast<const T_TYPE*>(x.data_ptr()),          \
        reinterpret_cast<const T_TYPE*>(weight.data_ptr()),     \
        reinterpret_cast<const T_TYPE*>(z.data_ptr()),          \
        reinterpret_cast<T_TYPE*>(output.data_ptr()),           \
        x.stride(0),                                            \
        z.stride(0),                                            \
        output.stride(0),                                       \
        N,                                                      \
        static_cast<float>(eps),                                \
        inv_N);                                                 \
  } while (0)

template <typename T>
void launch_gated_rms_norm(const torch::Tensor& x,
                           const torch::Tensor& weight,
                           const torch::Tensor& z,
                           torch::Tensor& output,
                           int N,
                           double eps,
                           cudaStream_t stream) {
  const int rows = static_cast<int>(x.size(0));
  const int block_threads = pick_block_threads(N);
  const float inv_N = 1.0f / static_cast<float>(N);
  switch (block_threads) {
    case 32:
      GATED_RMSNORM_DISPATCH_BLOCK(T, 32);
      break;
    case 64:
      GATED_RMSNORM_DISPATCH_BLOCK(T, 64);
      break;
    case 128:
      GATED_RMSNORM_DISPATCH_BLOCK(T, 128);
      break;
    case 256:
      GATED_RMSNORM_DISPATCH_BLOCK(T, 256);
      break;
    case 512:
      GATED_RMSNORM_DISPATCH_BLOCK(T, 512);
      break;
    default:
      GATED_RMSNORM_DISPATCH_BLOCK(T, 1024);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#undef GATED_RMSNORM_DISPATCH_BLOCK

}  // namespace

void gated_rms_norm_fused(const torch::Tensor& x,
                          const torch::Tensor& weight,
                          const torch::Tensor& z,
                          torch::Tensor output,
                          double eps) {
  CHECK(x.dim() == 2) << "gated_rms_norm_fused: x must be 2D [M, N]";
  CHECK(z.dim() == 2) << "gated_rms_norm_fused: z must be 2D [M, N]";
  CHECK(output.dim() == 2) << "gated_rms_norm_fused: output must be 2D [M, N]";
  CHECK(weight.dim() == 1) << "gated_rms_norm_fused: weight must be 1D [N]";
  CHECK(x.size(0) == z.size(0) && x.size(0) == output.size(0))
      << "gated_rms_norm_fused: row count mismatch";
  CHECK(x.size(1) == z.size(1) && x.size(1) == output.size(1))
      << "gated_rms_norm_fused: column count mismatch";
  CHECK(weight.size(0) == x.size(1))
      << "gated_rms_norm_fused: weight size mismatch";
  CHECK(x.scalar_type() == z.scalar_type() &&
        x.scalar_type() == output.scalar_type() &&
        x.scalar_type() == weight.scalar_type())
      << "gated_rms_norm_fused: dtype mismatch";
  CHECK(x.stride(-1) == 1 && z.stride(-1) == 1 && output.stride(-1) == 1 &&
        weight.stride(0) == 1)
      << "gated_rms_norm_fused: last dim must be contiguous (stride==1)";

  const int N = static_cast<int>(x.size(1));
  CHECK(N > 0) << "gated_rms_norm_fused: hidden_size must be > 0 (got ", N, ")";

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (x.scalar_type()) {
    case torch::kBFloat16:
      launch_gated_rms_norm<__nv_bfloat16>(
          x, weight, z, output, N, eps, stream);
      break;
    case torch::kHalf:
      launch_gated_rms_norm<__half>(x, weight, z, output, N, eps, stream);
      break;
    case torch::kFloat32:
      launch_gated_rms_norm<float>(x, weight, z, output, N, eps, stream);
      break;
    default:
      LOG(FATAL) << "gated_rms_norm_fused: unsupported dtype "
                 << x.scalar_type();
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {

template <typename T>
__global__ void __launch_bounds__(256, 1)
    conv1d_decode_kernel(const T* __restrict__ x,
                         const T* __restrict__ weight,
                         const T* __restrict__ bias,
                         T* __restrict__ conv_state,
                         const int32_t* __restrict__ cache_indices,
                         T* __restrict__ out,
                         int64_t x_stride_token,
                         int64_t x_stride_dim,
                         int64_t w_stride_dim,
                         int64_t w_stride_width,
                         int64_t state_stride_seq,
                         int64_t state_stride_dim,
                         int64_t state_stride_token,
                         int64_t out_stride_token,
                         int64_t out_stride_dim,
                         int batch,
                         int dim,
                         int num_cache_lines,
                         int pad_slot_id,
                         int width,
                         bool has_bias,
                         bool silu_activation) {
  const int batch_idx = blockIdx.y;
  const int feat = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch_idx >= batch || feat >= dim) {
    return;
  }
  const int32_t cache_idx = cache_indices[batch_idx];
  const int64_t out_base = static_cast<int64_t>(batch_idx) * out_stride_token +
                           static_cast<int64_t>(feat) * out_stride_dim;
  if (cache_idx == pad_slot_id || cache_idx < 0 ||
      cache_idx >= num_cache_lines) {
    out[out_base] = from_f32<T>(0.f);
    return;
  }

  const int64_t x_base = static_cast<int64_t>(batch_idx) * x_stride_token +
                         static_cast<int64_t>(feat) * x_stride_dim;
  const int64_t state_base =
      static_cast<int64_t>(cache_idx) * state_stride_seq +
      static_cast<int64_t>(feat) * state_stride_dim;
  const int64_t w_base = static_cast<int64_t>(feat) * w_stride_dim;
  const int state_len = width - 1;

  float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
  if (state_len >= 1) {
    s0 = to_f32<T>(conv_state[state_base + 0 * state_stride_token]);
  }
  if (state_len >= 2) {
    s1 = to_f32<T>(conv_state[state_base + 1 * state_stride_token]);
  }
  if (state_len >= 3) {
    s2 = to_f32<T>(conv_state[state_base + 2 * state_stride_token]);
  }
  if (state_len >= 4) {
    s3 = to_f32<T>(conv_state[state_base + 3 * state_stride_token]);
  }
  const float x_cur = to_f32<T>(x[x_base]);

  float w0 = 0.0f, w1 = 0.0f, w2 = 0.0f, w3 = 0.0f, w4 = 0.0f;
  w0 = to_f32<T>(weight[w_base + 0 * w_stride_width]);
  if (width >= 2) {
    w1 = to_f32<T>(weight[w_base + 1 * w_stride_width]);
  }
  if (width >= 3) {
    w2 = to_f32<T>(weight[w_base + 2 * w_stride_width]);
  }
  if (width >= 4) {
    w3 = to_f32<T>(weight[w_base + 3 * w_stride_width]);
  }
  if (width >= 5) {
    w4 = to_f32<T>(weight[w_base + 4 * w_stride_width]);
  }

  float acc = has_bias ? to_f32<T>(bias[feat]) : 0.0f;
  if (width == 2) {
    acc += s0 * w0 + x_cur * w1;
  } else if (width == 3) {
    acc += s0 * w0 + s1 * w1 + x_cur * w2;
  } else if (width == 4) {
    acc += s0 * w0 + s1 * w1 + s2 * w2 + x_cur * w3;
  } else {
    acc += s0 * w0 + s1 * w1 + s2 * w2 + s3 * w3 + x_cur * w4;
  }
  if (silu_activation) {
    acc = silu_f32(acc);
  }
  out[out_base] = from_f32<T>(acc);

  if (state_len >= 1) {
    if (state_len >= 2) {
      conv_state[state_base + 0 * state_stride_token] = from_f32<T>(s1);
    } else {
      conv_state[state_base + 0 * state_stride_token] = from_f32<T>(x_cur);
    }
  }
  if (state_len >= 2) {
    if (state_len >= 3) {
      conv_state[state_base + 1 * state_stride_token] = from_f32<T>(s2);
    } else {
      conv_state[state_base + 1 * state_stride_token] = from_f32<T>(x_cur);
    }
  }
  if (state_len >= 3) {
    if (state_len >= 4) {
      conv_state[state_base + 2 * state_stride_token] = from_f32<T>(s3);
    } else {
      conv_state[state_base + 2 * state_stride_token] = from_f32<T>(x_cur);
    }
  }
  if (state_len >= 4) {
    conv_state[state_base + 3 * state_stride_token] = from_f32<T>(x_cur);
  }
}

template <typename T>
void launch_conv1d_decode(const torch::Tensor& x,
                          const torch::Tensor& weight,
                          const torch::Tensor* bias_or_null,
                          torch::Tensor& conv_state,
                          const torch::Tensor& cache_indices,
                          torch::Tensor& out,
                          int batch,
                          int dim,
                          int num_cache_lines,
                          int pad_slot_id,
                          int width,
                          bool silu_activation,
                          cudaStream_t stream) {
  const int threads = (dim >= 256) ? 256 : ((dim + 31) / 32) * 32;
  const int blocks_x = (dim + threads - 1) / threads;
  dim3 grid(blocks_x, batch);
  dim3 block(threads);
  conv1d_decode_kernel<T><<<grid, block, 0, stream>>>(
      reinterpret_cast<const T*>(x.data_ptr()),
      reinterpret_cast<const T*>(weight.data_ptr()),
      bias_or_null ? reinterpret_cast<const T*>(bias_or_null->data_ptr())
                   : nullptr,
      reinterpret_cast<T*>(conv_state.data_ptr()),
      cache_indices.data_ptr<int32_t>(),
      reinterpret_cast<T*>(out.data_ptr()),
      x.stride(0),
      x.stride(1),
      weight.stride(0),
      weight.stride(1),
      conv_state.stride(0),
      conv_state.stride(1),
      conv_state.stride(2),
      out.stride(0),
      out.stride(1),
      batch,
      dim,
      num_cache_lines,
      pad_slot_id,
      width,
      bias_or_null != nullptr,
      silu_activation);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void causal_conv1d_decode_fused(const torch::Tensor& x,
                                const torch::Tensor& weight,
                                const std::optional<torch::Tensor>& bias,
                                torch::Tensor conv_state,
                                const torch::Tensor& cache_indices,
                                torch::Tensor output_buf,
                                int pad_slot_id,
                                bool silu_activation) {
  CHECK(x.dim() == 2) << "causal_conv1d_decode_fused: x must be 2D";
  CHECK(weight.dim() == 2)
      << "causal_conv1d_decode_fused: weight must be 2D [dim, width]";
  CHECK(conv_state.dim() == 3)
      << "causal_conv1d_decode_fused: conv_state must be 3D";
  CHECK(cache_indices.dim() == 1)
      << "causal_conv1d_decode_fused: cache_indices must be 1D";
  CHECK(cache_indices.scalar_type() == torch::kInt32)
      << "causal_conv1d_decode_fused: cache_indices must be int32";
  CHECK(output_buf.dim() == 2)
      << "causal_conv1d_decode_fused: output_buf must be 2D";
  CHECK(x.scalar_type() == output_buf.scalar_type())
      << "causal_conv1d_decode_fused: output_buf dtype must match x";
  CHECK(x.scalar_type() == conv_state.scalar_type())
      << "causal_conv1d_decode_fused: conv_state dtype must match x";
  CHECK(x.scalar_type() == weight.scalar_type())
      << "causal_conv1d_decode_fused: weight dtype must match x";

  const int batch = static_cast<int>(x.size(0));
  const int dim = static_cast<int>(x.size(1));
  const int width = static_cast<int>(weight.size(1));
  const int num_cache_lines = static_cast<int>(conv_state.size(0));
  CHECK(width >= 2 && width <= 5)
      << "causal_conv1d_decode_fused: width must be in [2,5], got " << width;
  CHECK(weight.size(0) == dim)
      << "causal_conv1d_decode_fused: weight dim mismatch";
  CHECK(conv_state.size(1) == dim)
      << "causal_conv1d_decode_fused: conv_state dim mismatch";
  CHECK(conv_state.size(2) == width - 1)
      << "causal_conv1d_decode_fused: conv_state.state_len must be "
         "width - 1";
  CHECK(cache_indices.size(0) == batch)
      << "causal_conv1d_decode_fused: cache_indices length must match "
         "batch (=x.size(0))";
  CHECK(output_buf.size(0) == batch && output_buf.size(1) == dim)
      << "causal_conv1d_decode_fused: output_buf shape mismatch";
  CHECK(output_buf.stride(1) == 1)
      << "causal_conv1d_decode_fused: output_buf last dim must be "
         "contiguous (stride==1)";

  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const torch::Tensor* bias_ptr = nullptr;
  if (bias.has_value() && bias.value().defined()) {
    bias_ptr = &bias.value();
    CHECK(bias_ptr->scalar_type() == x.scalar_type())
        << "causal_conv1d_decode_fused: bias dtype must match x";
    CHECK(bias_ptr->numel() == dim)
        << "causal_conv1d_decode_fused: bias must have shape [dim]";
  }

  switch (x.scalar_type()) {
    case torch::kBFloat16:
      launch_conv1d_decode<__nv_bfloat16>(x,
                                          weight,
                                          bias_ptr,
                                          conv_state,
                                          cache_indices,
                                          output_buf,
                                          batch,
                                          dim,
                                          num_cache_lines,
                                          pad_slot_id,
                                          width,
                                          silu_activation,
                                          stream);
      break;
    case torch::kHalf:
      launch_conv1d_decode<__half>(x,
                                   weight,
                                   bias_ptr,
                                   conv_state,
                                   cache_indices,
                                   output_buf,
                                   batch,
                                   dim,
                                   num_cache_lines,
                                   pad_slot_id,
                                   width,
                                   silu_activation,
                                   stream);
      break;
    case torch::kFloat32:
      launch_conv1d_decode<float>(x,
                                  weight,
                                  bias_ptr,
                                  conv_state,
                                  cache_indices,
                                  output_buf,
                                  batch,
                                  dim,
                                  num_cache_lines,
                                  pad_slot_id,
                                  width,
                                  silu_activation,
                                  stream);
      break;
    default:
      LOG(FATAL) << "causal_conv1d_decode_fused: unsupported dtype "
                 << x.scalar_type();
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {

int next_power_of_two(int n) {
  CHECK(n > 0);
  int p = 1;
  while (p < n) {
    p <<= 1;
  }
  return p;
}

constexpr int kFusedGdnDecodeMaxKV = 256;

template <typename scalar_t>
__global__ void fused_gdn_decode_kernel(
    const scalar_t* __restrict__ mixed_qkv,
    int64_t mixed_qkv_row_stride,
    float* __restrict__ state,
    const float* __restrict__ A_log_f32,
    const scalar_t* __restrict__ a,
    const float* __restrict__ dt_bias_f32,
    const scalar_t* __restrict__ b,
    const int32_t* __restrict__ state_indices,
    scalar_t* __restrict__ output,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    int64_t qk_cols,
    int64_t v_cols,
    float scale,
    float softplus_beta,
    float softplus_threshold) {
  const int batch = blockIdx.x;
  const int hv = blockIdx.y;
  const int tid = threadIdx.x;
  const int block_threads = blockDim.x;

  const int64_t slot = static_cast<int64_t>(state_indices[batch]);
  const int K = static_cast<int>(head_k_dim);
  const int V = static_cast<int>(head_v_dim);
  if (slot < 0) {
    for (int v = tid; v < V; v += block_threads) {
      const int64_t out_off =
          (static_cast<int64_t>(batch) * num_v_heads + hv) * head_v_dim +
          static_cast<int64_t>(v);
      output[out_off] = static_cast<scalar_t>(0);
    }
    return;
  }

  const int group = static_cast<int>(num_v_heads / num_k_heads);
  const int hk = hv / group;

  __shared__ float q_sh[kFusedGdnDecodeMaxKV];
  __shared__ float k_sh[kFusedGdnDecodeMaxKV];
  __shared__ float v_sh[kFusedGdnDecodeMaxKV];
  __shared__ float reduce_buf[kFusedGdnDecodeMaxKV];
  __shared__ float g_exp_sh;
  __shared__ float beta_sh;
  __shared__ float dot_qk_sh;

  const scalar_t* qkv_row =
      mixed_qkv + static_cast<int64_t>(batch) * mixed_qkv_row_stride;
  const int64_t q_base = static_cast<int64_t>(hk) * K;
  const int64_t k_base = qk_cols + static_cast<int64_t>(hk) * K;
  const int64_t v_base = 2 * qk_cols + static_cast<int64_t>(hv) * V;

  for (int k = tid; k < K; k += block_threads) {
    q_sh[k] = static_cast<float>(qkv_row[q_base + k]);
    k_sh[k] = static_cast<float>(qkv_row[k_base + k]);
  }
  for (int v = tid; v < V; v += block_threads) {
    v_sh[v] = static_cast<float>(qkv_row[v_base + v]);
  }
  __syncthreads();

  float local_q2 = 0.f;
  float local_k2 = 0.f;
  for (int k = tid; k < K; k += block_threads) {
    local_q2 += q_sh[k] * q_sh[k];
    local_k2 += k_sh[k] * k_sh[k];
  }
  reduce_buf[tid] = local_q2;
  __syncthreads();
  for (int s = block_threads >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      reduce_buf[tid] += reduce_buf[tid + s];
    }
    __syncthreads();
  }
  const float q_norm_inv = rsqrtf(reduce_buf[0] + 1e-6f);
  __syncthreads();
  reduce_buf[tid] = local_k2;
  __syncthreads();
  for (int s = block_threads >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      reduce_buf[tid] += reduce_buf[tid + s];
    }
    __syncthreads();
  }
  const float k_norm_inv = rsqrtf(reduce_buf[0] + 1e-6f);
  __syncthreads();

  for (int k = tid; k < K; k += block_threads) {
    q_sh[k] = q_sh[k] * q_norm_inv * scale;
    k_sh[k] = k_sh[k] * k_norm_inv;
  }

  if (tid == 0) {
    const float a_val =
        static_cast<float>(a[static_cast<int64_t>(batch) * num_v_heads + hv]);
    const float b_val =
        static_cast<float>(b[static_cast<int64_t>(batch) * num_v_heads + hv]);
    const float pre = a_val + dt_bias_f32[hv];
    const float bx = softplus_beta * pre;
    const float sp =
        (bx > softplus_threshold) ? pre : (log1pf(expf(bx)) / softplus_beta);
    const float g = -expf(A_log_f32[hv]) * sp;
    g_exp_sh = expf(g);
    beta_sh = 1.f / (1.f + expf(-b_val));
  }
  __syncthreads();

  float local_qk = 0.f;
  for (int k = tid; k < K; k += block_threads) {
    local_qk += q_sh[k] * k_sh[k];
  }
  reduce_buf[tid] = local_qk;
  __syncthreads();
  for (int s = block_threads >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      reduce_buf[tid] += reduce_buf[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    dot_qk_sh = reduce_buf[0];
  }
  __syncthreads();

  const float g_exp = g_exp_sh;
  const float beta_val = beta_sh;
  const float dot_qk = dot_qk_sh;

  if (tid < V) {
    const int64_t state_base =
        ((slot * num_v_heads + hv) * static_cast<int64_t>(V) +
         static_cast<int64_t>(tid)) *
        static_cast<int64_t>(K);

    float kv = 0.f;
    float dot_qH_v = 0.f;
    for (int k = 0; k < K; ++k) {
      const float s = state[state_base + static_cast<int64_t>(k)];
      kv += s * k_sh[k];
      dot_qH_v += s * q_sh[k];
    }

    const float delta = (v_sh[tid] - g_exp * kv) * beta_val;
    const float o_v = g_exp * dot_qH_v + delta * dot_qk;

    const int64_t out_off =
        (static_cast<int64_t>(batch) * num_v_heads + hv) * V +
        static_cast<int64_t>(tid);
    output[out_off] = static_cast<scalar_t>(o_v);

    for (int k = 0; k < K; ++k) {
      const int64_t off = state_base + static_cast<int64_t>(k);
      const float s_old = state[off];
      state[off] = g_exp * s_old + k_sh[k] * delta;
    }
  }
}

template <typename scalar_t>
void launch(const torch::Tensor& mixed_qkv,
            torch::Tensor& state,
            const torch::Tensor& A_log_f32,
            const torch::Tensor& a,
            const torch::Tensor& dt_bias_f32,
            const torch::Tensor& b,
            const torch::Tensor& state_indices_i32,
            torch::Tensor& output,
            int64_t num_k_heads,
            int64_t num_v_heads,
            int64_t head_k_dim,
            int64_t head_v_dim,
            int64_t qk_cols,
            int64_t v_cols,
            int64_t batch_size,
            float scale,
            float softplus_beta,
            float softplus_threshold,
            cudaStream_t stream) {
  const int work_threads =
      static_cast<int>(head_v_dim < head_k_dim ? head_k_dim : head_v_dim);
  const int block_threads = next_power_of_two(work_threads);
  CHECK(block_threads <= kFusedGdnDecodeMaxKV)
      << "fused GDN decode block size " << block_threads
      << " exceeds kFusedGdnDecodeMaxKV " << kFusedGdnDecodeMaxKV;
  const dim3 grid(static_cast<unsigned int>(batch_size),
                  static_cast<unsigned int>(num_v_heads),
                  1);
  fused_gdn_decode_kernel<scalar_t><<<grid, block_threads, 0, stream>>>(
      mixed_qkv.data_ptr<scalar_t>(),
      mixed_qkv.stride(0),
      state.data_ptr<float>(),
      A_log_f32.data_ptr<float>(),
      a.data_ptr<scalar_t>(),
      dt_bias_f32.data_ptr<float>(),
      b.data_ptr<scalar_t>(),
      state_indices_i32.data_ptr<int32_t>(),
      output.data_ptr<scalar_t>(),
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim,
      qk_cols,
      v_cols,
      scale,
      softplus_beta,
      softplus_threshold);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

torch::Tensor fused_gated_delta_rule_decode(
    MateGatedDeltaRuleDecodeParams& params) {
  auto mixed_qkv = params.mixed_qkv.contiguous();
  CHECK(mixed_qkv.dim() == 2)
      << "fused GDN decode expects mixed_qkv [B, D], got " << mixed_qkv.dim()
      << "-D";

  const int64_t batch_size = mixed_qkv.size(0);
  const int64_t num_k_heads = params.num_k_heads;
  const int64_t num_v_heads = params.num_v_heads;
  const int64_t head_k_dim = params.head_k_dim;
  const int64_t head_v_dim = params.head_v_dim;
  const int64_t qk_cols = num_k_heads * head_k_dim;
  const int64_t v_cols = num_v_heads * head_v_dim;

  CHECK(mixed_qkv.size(1) == 2 * qk_cols + v_cols)
      << "fused GDN decode mixed_qkv last dim mismatch: got "
      << mixed_qkv.size(1) << " expected " << (2 * qk_cols + v_cols);
  CHECK(head_k_dim > 0 && head_k_dim <= kFusedGdnDecodeMaxKV)
      << "fused GDN decode head_k_dim " << head_k_dim << " out of range (1, "
      << kFusedGdnDecodeMaxKV << "]";
  CHECK(head_v_dim > 0 && head_v_dim <= kFusedGdnDecodeMaxKV)
      << "fused GDN decode head_v_dim " << head_v_dim << " out of range (1, "
      << kFusedGdnDecodeMaxKV << "]";
  CHECK(num_v_heads % num_k_heads == 0)
      << "fused GDN decode requires num_v_heads divisible by num_k_heads (GQA)";
  CHECK(params.state.dim() == 4)
      << "fused GDN decode expects state [pool, Hv, V, K], got "
      << params.state.dim() << "-D";
  CHECK(params.state.size(1) == num_v_heads &&
        params.state.size(2) == head_v_dim &&
        params.state.size(3) == head_k_dim)
      << "fused GDN decode state shape mismatch with head dims (expected [_, "
      << num_v_heads << ", " << head_v_dim << ", " << head_k_dim << "])";
  CHECK(params.state.is_contiguous())
      << "fused GDN decode requires contiguous state cache";
  CHECK(params.state.scalar_type() == torch::kFloat32)
      << "fused GDN decode requires fp32 state cache (got "
      << params.state.scalar_type() << ")";

  auto a = params.a;
  if (a.dim() == 1) {
    a = a.unsqueeze(0);
  }
  auto b = params.b;
  if (b.dim() == 1) {
    b = b.unsqueeze(0);
  }
  CHECK(a.dim() == 2 && b.dim() == 2)
      << "fused GDN decode expects a/b shaped [B, Hv]";
  a = a.contiguous();
  b = b.contiguous();

  auto A_log_f32 = params.A_log.to(torch::kFloat32).contiguous();
  auto dt_bias_f32 = params.dt_bias.to(torch::kFloat32).contiguous();
  auto state_indices_i32 = params.state_indices.to(torch::kInt32).contiguous();

  torch::Tensor output;
  if (params.decode_output.has_value() &&
      params.decode_output.value().defined()) {
    output = params.decode_output.value();
    CHECK(output.is_contiguous())
        << "fused GDN decode requires contiguous decode_output";
  } else {
    output = torch::empty({batch_size, num_v_heads, head_v_dim},
                          mixed_qkv.options());
  }

  const float scale = static_cast<float>(params.scale);
  const float softplus_beta = 1.0f;
  const float softplus_threshold = 20.0f;

  const at::cuda::OptionalCUDAGuard guard(device_of(mixed_qkv));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (mixed_qkv.scalar_type() == torch::kBFloat16) {
    launch<at::BFloat16>(mixed_qkv,
                         params.state,
                         A_log_f32,
                         a,
                         dt_bias_f32,
                         b,
                         state_indices_i32,
                         output,
                         num_k_heads,
                         num_v_heads,
                         head_k_dim,
                         head_v_dim,
                         qk_cols,
                         v_cols,
                         batch_size,
                         scale,
                         softplus_beta,
                         softplus_threshold,
                         stream);
  } else if (mixed_qkv.scalar_type() == torch::kFloat16) {
    launch<at::Half>(mixed_qkv,
                     params.state,
                     A_log_f32,
                     a,
                     dt_bias_f32,
                     b,
                     state_indices_i32,
                     output,
                     num_k_heads,
                     num_v_heads,
                     head_k_dim,
                     head_v_dim,
                     qk_cols,
                     v_cols,
                     batch_size,
                     scale,
                     softplus_beta,
                     softplus_threshold,
                     stream);
  } else {
    LOG(FATAL) << "fused GDN decode: unsupported dtype "
               << mixed_qkv.scalar_type();
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output;
}

}  // namespace cuda
}  // namespace kernel
}  // namespace xllm

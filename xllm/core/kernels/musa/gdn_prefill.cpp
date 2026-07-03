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

#include <glog/logging.h>

#include <sstream>

#include "core/common/macros.h"
#include "core/kernels/musa/musa_ops_api.h"
#include "core/kernels/param.h"

namespace xllm {
namespace kernel {
namespace cuda {

namespace {

inline torch::Tensor l2norm_last(const torch::Tensor& x, double eps) {
  return x / (x.pow(2).sum(-1, /*keepdim=*/true) + eps).sqrt();
}

}  // namespace

std::pair<torch::Tensor, torch::Tensor> fused_recurrent_gated_delta_rule(
    FusedRecurrentGatedDeltaRuleParams& params) {
  auto query = params.q;
  auto key = params.k;
  auto value = params.v;
  auto g = params.g;
  const auto initial_dtype = query.scalar_type();

  if (params.use_qk_l2norm_in_kernel) {
    query = l2norm_last(query, 1e-6);
    key = l2norm_last(key, 1e-6);
  }

  auto to_f32_bhtd = [](const torch::Tensor& x) {
    return x.transpose(1, 2).contiguous().to(torch::kFloat32);
  };
  query = to_f32_bhtd(query);
  key = to_f32_bhtd(key);
  value = to_f32_bhtd(value);
  g = to_f32_bhtd(g);
  torch::Tensor beta_f32;
  if (params.beta.has_value() && params.beta.value().defined()) {
    beta_f32 = to_f32_bhtd(params.beta.value());
  } else {
    beta_f32 = torch::ones_like(g);
  }

  const int64_t batch_size = query.size(0);
  const int64_t num_heads = query.size(1);
  const int64_t sequence_length = query.size(2);
  const int64_t k_head_dim = key.size(-1);
  const int64_t v_head_dim = value.size(-1);
  const float scale_val =
      params.scale.value_or(1.0f / std::sqrt(static_cast<float>(k_head_dim)));
  query = query * scale_val;

  torch::Tensor last_recurrent_state;
  if (params.initial_state.has_value() &&
      params.initial_state.value().defined()) {
    last_recurrent_state =
        params.initial_state.value().to(torch::kFloat32).transpose(-1, -2);
  } else {
    last_recurrent_state = torch::zeros(
        {batch_size, num_heads, k_head_dim, v_head_dim},
        torch::TensorOptions().dtype(torch::kFloat32).device(value.device()));
  }

  auto core_attn_out = torch::zeros(
      {batch_size, num_heads, sequence_length, v_head_dim},
      torch::TensorOptions().dtype(torch::kFloat32).device(value.device()));

  for (int64_t i = 0; i < sequence_length; ++i) {
    auto q_t = query.select(2, i);
    auto k_t = key.select(2, i);
    auto v_t = value.select(2, i);
    auto g_t = g.select(2, i);
    auto beta_t = beta_f32.select(2, i);
    auto g_exp = g_t.exp().unsqueeze(-1).unsqueeze(-1);
    last_recurrent_state.mul_(g_exp);
    auto kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(-2);
    auto delta = (v_t - kv_mem) * beta_t.unsqueeze(-1);
    last_recurrent_state.add_(k_t.unsqueeze(-1) * delta.unsqueeze(-2));
    core_attn_out.select(2, i) =
        (last_recurrent_state * q_t.unsqueeze(-1)).sum(-2);
  }

  core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype);
  last_recurrent_state = last_recurrent_state.transpose(-1, -2);
  return {core_attn_out, last_recurrent_state};
}

std::pair<torch::Tensor, torch::Tensor> chunk_gated_delta_rule(
    ChunkGatedDeltaRuleParams& params) {
  auto query = params.q;
  auto key = params.k;
  auto value = params.v;
  auto g = params.g;
  auto beta = params.beta;
  const int64_t chunk_size = 64;
  const auto initial_dtype = query.dtype();

  if (params.use_qk_l2norm_in_kernel) {
    query = l2norm_last(query, 1e-6);
    key = l2norm_last(key, 1e-6);
  }

  const int64_t Hqk = query.size(2);
  const int64_t Hv = value.size(2);
  if (Hqk != Hv) {
    CHECK(Hv % Hqk == 0) << "chunk_gated_delta_rule: Hv (" << Hv
                         << ") must be a multiple of Hqk (" << Hqk
                         << ") for GQA expansion";
    const int64_t repeat = Hv / Hqk;
    query = query.repeat_interleave(repeat, /*dim=*/2);
    key = key.repeat_interleave(repeat, /*dim=*/2);
  }

  auto to_f32_thd = [](const torch::Tensor& x) {
    return x.transpose(1, 2).contiguous().to(torch::kFloat32);
  };
  query = to_f32_thd(query);
  key = to_f32_thd(key);
  value = to_f32_thd(value);
  beta = beta.transpose(1, 2).contiguous().to(torch::kFloat32);
  g = g.transpose(1, 2).contiguous().to(torch::kFloat32);

  const int64_t batch_size = query.size(0);
  const int64_t num_heads = query.size(1);
  const int64_t sequence_length = query.size(2);
  const int64_t k_head_dim = key.size(-1);
  const int64_t v_head_dim = value.size(-1);

  const int64_t pad_size =
      (chunk_size - sequence_length % chunk_size) % chunk_size;
  using PadOpts = torch::nn::functional::PadFuncOptions;
  if (pad_size != 0) {
    query = torch::nn::functional::pad(query, PadOpts({0, 0, 0, pad_size}));
    key = torch::nn::functional::pad(key, PadOpts({0, 0, 0, pad_size}));
    value = torch::nn::functional::pad(value, PadOpts({0, 0, 0, pad_size}));
    beta = torch::nn::functional::pad(beta, PadOpts({0, pad_size}));
    g = torch::nn::functional::pad(g, PadOpts({0, pad_size}));
  }
  const int64_t total_sequence_length = sequence_length + pad_size;
  const float scale =
      params.scale.value_or(1.0f / std::sqrt(static_cast<float>(k_head_dim)));
  query = query * scale;
  auto v_beta = value * beta.unsqueeze(-1);
  auto k_beta = key * beta.unsqueeze(-1);

  auto reshape_to_chunks = [chunk_size](const torch::Tensor& x) {
    return x.reshape(
        {x.size(0), x.size(1), x.size(2) / chunk_size, chunk_size, x.size(3)});
  };
  query = reshape_to_chunks(query);
  key = reshape_to_chunks(key);
  value = reshape_to_chunks(value);
  k_beta = reshape_to_chunks(k_beta);
  v_beta = reshape_to_chunks(v_beta);
  g = g.reshape({g.size(0), g.size(1), g.size(2) / chunk_size, chunk_size});

  auto mask = torch::triu(
      torch::ones(
          {chunk_size, chunk_size},
          torch::TensorOptions().dtype(torch::kBool).device(query.device())),
      0);
  g = g.cumsum(-1);
  auto g_diff = g.unsqueeze(-1) - g.unsqueeze(-2);
  auto decay_mask = g_diff.tril().exp().to(torch::kFloat32).tril();
  auto attn = -(torch::matmul(k_beta, key.transpose(-1, -2)) * decay_mask)
                   .masked_fill(mask, 0.0);
  for (int64_t i = 1; i < chunk_size; ++i) {
    if (!attn.is_contiguous()) {
      attn = attn.contiguous();
    }
    auto row = attn.slice(-2, i, i + 1).slice(-1, 0, i).squeeze(-2).clone();
    auto sub = attn.slice(-2, 0, i).slice(-1, 0, i).clone();
    auto row_final = row + (row.unsqueeze(-1) * sub).sum(-2);
    attn.index_put_({torch::indexing::Ellipsis,
                     torch::indexing::Slice(i, i + 1),
                     torch::indexing::Slice(0, i)},
                    row_final.unsqueeze(-2));
  }
  attn = attn +
         torch::eye(
             chunk_size,
             torch::TensorOptions().dtype(attn.dtype()).device(attn.device()));
  value = torch::matmul(attn, v_beta);
  auto k_cumdecay = torch::matmul(attn, k_beta * g.exp().unsqueeze(-1));

  torch::Tensor last_recurrent_state;
  if (params.initial_state.has_value() &&
      params.initial_state.value().defined()) {
    last_recurrent_state = params.initial_state.value().to(value.dtype());
  } else {
    last_recurrent_state = torch::zeros(
        {batch_size, num_heads, k_head_dim, v_head_dim},
        torch::TensorOptions().dtype(value.dtype()).device(value.device()));
  }
  auto core_attn_out = torch::zeros_like(value);
  const int64_t num_chunks = total_sequence_length / chunk_size;

  auto upper_mask = torch::triu(
      torch::ones(
          {chunk_size, chunk_size},
          torch::TensorOptions().dtype(torch::kBool).device(query.device())),
      1);
  for (int64_t i = 0; i < num_chunks; ++i) {
    auto q_i = query.select(2, i);
    auto k_i = key.select(2, i);
    auto v_i = value.select(2, i);
    auto attn_i =
        (torch::matmul(q_i, k_i.transpose(-1, -2)) * decay_mask.select(2, i))
            .masked_fill_(upper_mask, 0.0);
    auto v_prime = torch::matmul(k_cumdecay.select(2, i), last_recurrent_state);
    auto v_new = v_i - v_prime;
    auto attn_inter = torch::matmul(q_i * g.select(2, i).unsqueeze(-1).exp(),
                                    last_recurrent_state);
    core_attn_out.select(2, i) = attn_inter + torch::matmul(attn_i, v_new);
    auto g_i_last = g.select(2, i).select(-1, -1).unsqueeze(-1);
    auto g_exp_term = (g_i_last - g.select(2, i)).exp().unsqueeze(-1);
    auto k_g_exp = (k_i * g_exp_term).transpose(-1, -2).contiguous();
    last_recurrent_state = last_recurrent_state * g_i_last.unsqueeze(-1).exp() +
                           torch::matmul(k_g_exp, v_new);
  }
  const auto s = core_attn_out.sizes();
  core_attn_out = core_attn_out.reshape({s[0], s[1], s[2] * s[3], s[4]});
  core_attn_out = core_attn_out.slice(2, 0, sequence_length);
  core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype);
  return {core_attn_out, last_recurrent_state};
}

namespace {

constexpr int64_t kGdnChunkSize = 64;

int64_t chunk_pad_size(int64_t seq_len, int64_t chunk_size) {
  return (chunk_size - seq_len % chunk_size) % chunk_size;
}

torch::Tensor pad_time_dim_4d(const torch::Tensor& tensor, int64_t pad_size) {
  if (pad_size == 0) {
    return tensor;
  }
  return torch::nn::functional::pad(
      tensor, torch::nn::functional::PadFuncOptions({0, 0, 0, 0, 0, pad_size}));
}

torch::Tensor pad_time_dim_3d(const torch::Tensor& tensor,
                              int64_t pad_size,
                              double pad_value) {
  if (pad_size == 0) {
    return tensor;
  }
  return torch::nn::functional::pad(
      tensor,
      torch::nn::functional::PadFuncOptions({0, 0, 0, pad_size})
          .mode(torch::kConstant)
          .value(pad_value));
}

std::string mate_gdn_dtype_suffix(torch::ScalarType dtype) {
  if (dtype == torch::kBFloat16) {
    return "bf16";
  }
  if (dtype == torch::kFloat16) {
    return "f16";
  }
  LOG(FATAL) << "mate GDN prefill expects bfloat16 or float16 q/k/v";
}

void l2norm_last_dim(torch::Tensor& tensor) {
  const auto orig_dtype = tensor.scalar_type();
  tensor = torch::nn::functional::normalize(
      tensor.to(torch::kFloat32),
      torch::nn::functional::NormalizeFuncOptions().p(2).dim(-1));
  tensor = tensor.to(orig_dtype);
}

torch::Tensor chunk_local_cumsum_log_alpha(const torch::Tensor& g_log,
                                           int64_t chunk_size) {
  auto alpha = g_log.to(torch::kFloat32).exp().contiguous();
  const int64_t batch_size = alpha.size(0);
  const int64_t pad_size = chunk_pad_size(alpha.size(1), chunk_size);
  if (pad_size > 0) {
    alpha = pad_time_dim_3d(alpha, pad_size, 1.0);
  }
  const int64_t padded_len = alpha.size(1);
  auto log_alpha = alpha.clamp_min(1e-20f).log();
  log_alpha =
      log_alpha.reshape({batch_size, padded_len / chunk_size, chunk_size, -1});
  log_alpha = log_alpha.cumsum(/*dim=*/2);
  return log_alpha.reshape({batch_size, padded_len, alpha.size(2)})
      .contiguous();
}

}  // namespace

std::string get_mate_gdn_prefill_uri(int64_t num_q_heads,
                                     int64_t num_v_heads,
                                     torch::ScalarType dtype) {
  std::ostringstream oss;
  oss << "mate_gdn_prefill_hq" << num_q_heads << "_hv" << num_v_heads << "_"
      << mate_gdn_dtype_suffix(dtype);
  return oss.str();
}

std::pair<torch::Tensor, torch::Tensor> mate_gated_delta_rule_prefill(
    MateGatedDeltaRulePrefillParams& params) {
  auto query = params.q.contiguous();
  auto key = params.k.contiguous();
  auto value = params.v.contiguous();
  CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4)
      << "mate GDN prefill expects q/k/v shaped [B, T, H, D]";
  CHECK(query.scalar_type() == key.scalar_type() &&
        query.scalar_type() == value.scalar_type())
      << "mate GDN prefill expects q/k/v to share dtype";

  const int64_t batch_size = query.size(0);
  const int64_t seq_len = query.size(1);
  const int64_t pad_size = chunk_pad_size(seq_len, kGdnChunkSize);
  if (pad_size > 0) {
    query = pad_time_dim_4d(query, pad_size);
    key = pad_time_dim_4d(key, pad_size);
    value = pad_time_dim_4d(value, pad_size);
  }
  const int64_t num_tokens = query.size(1);
  const int64_t num_q_heads = query.size(2);
  const int64_t num_v_heads = value.size(2);
  const int64_t head_k_dim = query.size(3);
  const int64_t head_v_dim = value.size(3);
  CHECK(head_k_dim == head_v_dim)
      << "mate GDN prefill currently requires K == V, got K=" << head_k_dim
      << " V=" << head_v_dim;
  CHECK(num_v_heads % num_q_heads == 0)
      << "mate GDN prefill expects Hv divisible by Hqk";

  if (params.use_qk_l2norm_in_kernel) {
    l2norm_last_dim(query);
    l2norm_last_dim(key);
  }
  query = query.contiguous();
  key = key.contiguous();
  value = value.contiguous();

  auto beta = params.beta.to(torch::kFloat32).contiguous();
  if (pad_size > 0) {
    beta = pad_time_dim_3d(beta, pad_size, 0.0);
  }
  auto g_cumsum = chunk_local_cumsum_log_alpha(
      pad_size > 0
          ? pad_time_dim_3d(
                params.g.to(torch::kFloat32).contiguous(), pad_size, 0.0)
          : params.g.to(torch::kFloat32).contiguous(),
      kGdnChunkSize);

  const std::string uri =
      get_mate_gdn_prefill_uri(num_q_heads, num_v_heads, query.scalar_type());
  bind_tvmffi_stream_to_current_torch_stream(query.device());
  auto run = get_function(uri, "run");

  auto a_dummy = torch::empty(
      {batch_size, num_tokens, num_v_heads, kGdnChunkSize}, query.options());
  auto h0 = torch::zeros(
      {batch_size, num_v_heads, head_v_dim, head_k_dim},
      torch::TensorOptions().dtype(torch::kFloat32).device(query.device()));
  auto output = torch::empty({batch_size, num_tokens, num_v_heads, head_v_dim},
                             value.options());
  auto final_state = torch::empty(
      {batch_size, num_v_heads, head_v_dim, head_k_dim},
      torch::TensorOptions().dtype(torch::kFloat32).device(query.device()));

  run(to_ffi_tensor(query),
      to_ffi_tensor(key),
      to_ffi_tensor(value),
      to_ffi_tensor(a_dummy),
      to_ffi_tensor(g_cumsum),
      to_ffi_tensor(beta),
      to_ffi_tensor(h0),
      to_ffi_tensor(output),
      to_ffi_tensor(final_state));

  if (pad_size > 0) {
    output = output.slice(/*dim=*/1, /*start=*/0, /*end=*/seq_len);
  }
  return {output, final_state};
}

torch::Tensor causal_conv1d(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& conv_state,
    const std::optional<torch::Tensor>& bias_opt,
    const torch::IntArrayRef query_start_loc_opt,
    const torch::IntArrayRef cache_indices_opt,
    const torch::IntArrayRef initial_state_mode_opt,
    const torch::IntArrayRef /*num_accepted_tokens_opt*/,
    int64_t activation_mode,
    int64_t /*pad_slot_id*/,
    int64_t /*run_mode*/) {
  const int64_t dim = x.size(-1);
  const int64_t width = weight.size(0);
  const int64_t state_len = width - 1;
  auto x_f = x.to(torch::kFloat32);
  auto w_f = weight.to(torch::kFloat32);
  torch::Tensor bias_f;
  const bool has_bias = bias_opt.has_value() && bias_opt.value().defined();
  if (has_bias) {
    bias_f = bias_opt.value().to(torch::kFloat32);
  }
  auto output = torch::empty_like(x_f);
  const int64_t num_seq = static_cast<int64_t>(query_start_loc_opt.size()) - 1;
  for (int64_t s = 0; s < num_seq; ++s) {
    const int64_t start = query_start_loc_opt[s];
    const int64_t end = query_start_loc_opt[s + 1];
    const int64_t seq_len = end - start;
    if (seq_len <= 0) {
      continue;
    }
    const int64_t slot = cache_indices_opt.empty() ? s : cache_indices_opt[s];
    const bool has_init =
        !initial_state_mode_opt.empty() && initial_state_mode_opt[s] != 0;
    auto seq = x_f.narrow(0, start, seq_len);
    torch::Tensor prefix;
    if (has_init) {
      prefix = conv_state[slot].to(torch::kFloat32);
    } else {
      prefix = torch::zeros({state_len, dim}, x_f.options());
    }
    auto padded = torch::cat({prefix, seq}, 0);
    auto out_seq = torch::zeros({seq_len, dim}, x_f.options());
    for (int64_t j = 0; j < width; ++j) {
      auto shifted = padded.narrow(0, j, seq_len);
      out_seq += shifted * w_f.select(0, j).unsqueeze(0);
    }
    if (has_bias) {
      out_seq += bias_f.unsqueeze(0);
    }
    if (activation_mode != 0) {
      out_seq = torch::silu(out_seq);
    }
    output.narrow(0, start, seq_len).copy_(out_seq);
    auto tail = padded.narrow(0, padded.size(0) - state_len, state_len);
    conv_state[slot].copy_(tail.to(conv_state.dtype()));
  }
  return output.to(x.dtype());
}

}  // namespace cuda
}  // namespace kernel
}  // namespace xllm

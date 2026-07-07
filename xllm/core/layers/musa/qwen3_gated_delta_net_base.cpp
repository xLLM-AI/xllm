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

#include "layers/musa/qwen3_gated_delta_net_base.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdlib>
#include <tuple>

#include "kernels/ops_api.h"

#if defined(USE_CUDA) || defined(USE_MUSA)
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#endif

namespace xllm {
namespace layer {

namespace {
#if defined(USE_CUDA) || defined(USE_MUSA)
constexpr bool kEnableFusedGdnDecode = true;
constexpr bool kEnableMateGdnDecode = false;
constexpr bool kEnableMateGdnPrefill = false;

bool qwen35_mtp_debug_enabled() {
  static const bool enabled = std::getenv("XLLM_DEBUG_QWEN35_MTP") != nullptr;
  return enabled;
}

void qwen35_mtp_debug_sync(const char* stage) {
  if (!qwen35_mtp_debug_enabled()) {
    return;
  }
  LOG(INFO) << "[Qwen3.5 MTP debug] sync begin: " << stage;
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  LOG(INFO) << "[Qwen3.5 MTP debug] sync end: " << stage;
}
#else
bool qwen35_mtp_debug_enabled() { return false; }

void qwen35_mtp_debug_sync(const char*) {}
#endif

torch::Tensor l2norm(const torch::Tensor& x, int64_t dim, double eps = 1e-6) {
  auto norm =
      torch::sqrt(torch::sum(torch::square(x), dim, /*keepdim=*/true) + eps);
  return x / norm;
}

torch::Tensor repeat_tensor_heads(const torch::Tensor& tensor,
                                  int64_t target_heads,
                                  int64_t head_dim) {
  const int64_t current_heads = tensor.size(head_dim);
  if (current_heads == target_heads) {
    return tensor;
  }
  CHECK_GT(current_heads, 0) << "current heads must be positive";
  CHECK_EQ(target_heads % current_heads, 0)
      << "target heads must be divisible by current heads, target_heads="
      << target_heads << ", current_heads=" << current_heads;

  const int64_t repeats = target_heads / current_heads;
  std::vector<int64_t> view_shape = tensor.sizes().vec();
  view_shape.insert(view_shape.begin() + head_dim + 1, 1);
  std::vector<int64_t> expand_shape = view_shape;
  expand_shape[head_dim + 1] = repeats;
  std::vector<int64_t> output_shape = tensor.sizes().vec();
  output_shape[head_dim] = target_heads;
  return tensor.unsqueeze(head_dim + 1)
      .expand(expand_shape)
      .reshape(output_shape)
      .contiguous();
}

std::tuple<torch::Tensor, torch::Tensor> torch_recurrent_gated_delta_rule(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor g,
    torch::Tensor beta,
    std::optional<torch::Tensor> initial_state,
    bool output_final_state = true,
    bool use_qk_l2norm_in_kernel = true) {
  auto initial_dtype = query.dtype();

  if (use_qk_l2norm_in_kernel) {
    query = l2norm(query, /*dim=*/-1, /*eps=*/1e-6);
    key = l2norm(key, /*dim=*/-1, /*eps=*/1e-6);
  }

  auto to_float32_and_transpose = [](torch::Tensor x) {
    return x.transpose(1, 2).contiguous().to(torch::kFloat32);
  };
  query = to_float32_and_transpose(query);
  key = to_float32_and_transpose(key);
  value = to_float32_and_transpose(value);
  beta = to_float32_and_transpose(beta);
  g = to_float32_and_transpose(g);
  const int64_t value_num_heads = value.size(1);
  query = repeat_tensor_heads(query, value_num_heads, /*head_dim=*/1);
  key = repeat_tensor_heads(key, value_num_heads, /*head_dim=*/1);

  int64_t batch_size = key.size(0);
  int64_t num_heads = key.size(1);
  int64_t sequence_length = key.size(2);
  int64_t k_head_dim = key.size(3);
  int64_t v_head_dim = value.size(3);

  float scale_val = 1.0 / std::sqrt(static_cast<float>(query.size(-1)));
  query = query * scale_val;
  torch::Tensor core_attn_out = torch::zeros(
      {batch_size, num_heads, sequence_length, v_head_dim},
      torch::TensorOptions().dtype(torch::kFloat32).device(value.device()));
  torch::Tensor last_recurrent_state;
  if (!initial_state.has_value()) {
    last_recurrent_state = torch::zeros(
        {batch_size, num_heads, k_head_dim, v_head_dim},
        torch::TensorOptions().dtype(torch::kFloat32).device(value.device()));
  } else {
    last_recurrent_state =
        initial_state.value().to(value.device(), torch::kFloat32);
  }

  for (int64_t i = 0; i < sequence_length; ++i) {
    torch::Tensor q_t = query.select(/*dim=*/2, i);
    torch::Tensor k_t = key.select(/*dim=*/2, i);
    torch::Tensor v_t = value.select(/*dim=*/2, i);
    torch::Tensor g_t = g.select(/*dim=*/2, i)
                            .exp()
                            .unsqueeze(/*dim=*/-1)
                            .unsqueeze(/*dim=*/-1);
    torch::Tensor beta_t = beta.select(/*dim=*/2, i).unsqueeze(/*dim=*/-1);
    last_recurrent_state = last_recurrent_state * g_t;
    torch::Tensor kv_mem =
        torch::sum(last_recurrent_state * k_t.unsqueeze(-1), -2);
    torch::Tensor delta = (v_t - kv_mem) * beta_t;
    last_recurrent_state =
        last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2);
    core_attn_out.select(2, i) =
        torch::sum(last_recurrent_state * q_t.unsqueeze(-1), -2);
  }

  core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype);
  return std::make_tuple(core_attn_out, last_recurrent_state);
}

std::tuple<torch::Tensor, torch::Tensor> torch_chunk_gated_delta_rule(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor g,
    torch::Tensor beta,
    int64_t chunk_size = 64,
    std::optional<torch::Tensor> initial_state = std::nullopt,
    bool output_final_state = true,
    bool use_qk_l2norm_in_kernel = true) {
  auto initial_dtype = query.dtype();
  if (use_qk_l2norm_in_kernel) {
    query = l2norm(query, /*dim=*/-1, /*eps=*/1e-6);
    key = l2norm(key, /*dim=*/-1, /*eps=*/1e-6);
  }
  auto to_float32 = [](torch::Tensor x) {
    return x.transpose(1, 2).contiguous().to(torch::kFloat32);
  };

  query = to_float32(query);
  key = to_float32(key);
  value = to_float32(value);
  beta = to_float32(beta);
  g = to_float32(g);
  const int64_t value_num_heads = value.size(1);
  query = repeat_tensor_heads(query, value_num_heads, /*head_dim=*/1);
  key = repeat_tensor_heads(key, value_num_heads, /*head_dim=*/1);

  int64_t batch_size = query.size(0);
  int64_t num_heads = query.size(1);
  int64_t sequence_length = query.size(2);
  int64_t k_head_dim = key.size(-1);
  int64_t v_head_dim = value.size(-1);

  int64_t pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size;
  query = torch::nn::functional::pad(
      query, torch::nn::functional::PadFuncOptions({0, 0, 0, pad_size}));
  key = torch::nn::functional::pad(
      key, torch::nn::functional::PadFuncOptions({0, 0, 0, pad_size}));
  value = torch::nn::functional::pad(
      value, torch::nn::functional::PadFuncOptions({0, 0, 0, pad_size}));
  beta = torch::nn::functional::pad(
      beta, torch::nn::functional::PadFuncOptions({0, pad_size}));
  g = torch::nn::functional::pad(
      g, torch::nn::functional::PadFuncOptions({0, pad_size}));

  int64_t total_sequence_length = sequence_length + pad_size;
  float scale = 1.0 / std::sqrt(static_cast<float>(query.size(-1)));
  query = query * scale;
  auto v_beta = value * beta.unsqueeze(-1);
  auto k_beta = key * beta.unsqueeze(-1);
  auto reshape_to_chunks = [chunk_size](torch::Tensor x) {
    auto shape = x.sizes();
    std::vector<int64_t> new_shape = {
        shape[0], shape[1], shape[2] / chunk_size, chunk_size, shape[3]};
    return x.reshape(new_shape);
  };

  query = reshape_to_chunks(query);
  key = reshape_to_chunks(key);
  value = reshape_to_chunks(value);
  k_beta = reshape_to_chunks(k_beta);
  v_beta = reshape_to_chunks(v_beta);

  auto g_shape = g.sizes();
  std::vector<int64_t> g_new_shape = {
      g_shape[0], g_shape[1], g_shape[2] / chunk_size, chunk_size};
  g = g.reshape(g_new_shape);
  auto mask = torch::triu(
      torch::ones(
          {chunk_size, chunk_size},
          torch::TensorOptions().dtype(torch::kBool).device(query.device())),
      0);

  g = g.cumsum(-1);
  auto g_diff = g.unsqueeze(-1) - g.unsqueeze(-2);
  auto decay_mask = g_diff.tril().exp().to(torch::kFloat32);
  decay_mask = decay_mask.tril();
  auto attn = -(torch::matmul(k_beta, key.transpose(/*dim0=*/-1, /*dim1=*/-2)) *
                decay_mask)
                   .masked_fill(mask, /*value=*/0.0)
                   .contiguous();
  for (int64_t i = 1; i < chunk_size; ++i) {
    auto row = attn.slice(/*dim=*/-2, /*start=*/i, /*end=*/i + 1)
                   .slice(/*dim=*/-1, /*start=*/0, /*end=*/i)
                   .squeeze(/*dim=*/-2)
                   .clone()
                   .contiguous();
    auto sub = attn.slice(-2, 0, i).slice(-1, 0, i).clone().contiguous();
    auto row_unsq = row.unsqueeze(-1).contiguous();
    auto row_sub_mul = (row_unsq * sub).contiguous();
    auto row_sub_sum = row_sub_mul.sum(-2).contiguous();
    auto row_final = (row + row_sub_sum).contiguous();
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
  auto k_cumdecay = torch::matmul(attn, (k_beta * g.exp().unsqueeze(-1)));
  torch::Tensor last_recurrent_state;
  if (!initial_state.has_value()) {
    last_recurrent_state = torch::zeros(
        {batch_size, num_heads, k_head_dim, v_head_dim},
        torch::TensorOptions().dtype(value.dtype()).device(value.device()));
  } else {
    last_recurrent_state = initial_state.value().to(value);
  }
  auto core_attn_out = torch::zeros_like(value);
  mask = torch::triu(
      torch::ones(
          {chunk_size, chunk_size},
          torch::TensorOptions().dtype(torch::kBool).device(query.device())),
      1);
  int64_t num_chunks = total_sequence_length / chunk_size;
  for (int64_t i = 0; i < num_chunks; ++i) {
    auto q_i = query.select(2, i);
    auto k_i = key.select(2, i);
    auto v_i = value.select(2, i);
    auto attn_i =
        (torch::matmul(q_i, k_i.transpose(-1, -2)) * decay_mask.select(2, i))
            .masked_fill_(mask, 0.0);
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
  auto core_attn_out_shape = core_attn_out.sizes();
  std::vector<int64_t> reshape_shape = {
      core_attn_out_shape[0],
      core_attn_out_shape[1],
      core_attn_out_shape[2] * core_attn_out_shape[3],
      core_attn_out_shape[4]};
  core_attn_out = core_attn_out.reshape(reshape_shape);
  core_attn_out = core_attn_out.slice(2, 0, sequence_length);
  core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype);
  return std::make_tuple(core_attn_out, last_recurrent_state);
}

// Host-side equivalent of expand_sequence_tensor_to_batch for a small int
// vector. Used for num_accepted_tokens in the spec-verify loops so we avoid a
// per-GDN-layer device->host copy + implicit stream sync (the host values are
// already available in ModelInputParams::num_accepted_tokens_host).
std::vector<int64_t> expand_int_vector_to_batch(const std::vector<int64_t>& src,
                                                int64_t target_batch,
                                                const char* name) {
  const int64_t source = static_cast<int64_t>(src.size());
  CHECK_GT(source, 0) << name << " must not be empty";
  CHECK_EQ(target_batch % source, 0) << name << " cannot be expanded from "
                                     << source << " to " << target_batch;
  const int64_t repeat = target_batch / source;
  std::vector<int64_t> out(static_cast<size_t>(target_batch));
  for (int64_t j = 0; j < target_batch; ++j) {
    out[static_cast<size_t>(j)] = src[static_cast<size_t>(j / repeat)];
  }
  return out;
}

int64_t get_checkpoint_stride(const torch::Tensor& conv_cache,
                              const torch::Tensor& ssm_cache) {
  if (!conv_cache.defined() || !ssm_cache.defined() ||
      conv_cache.numel() == 0 || ssm_cache.numel() == 0) {
    return 1;
  }
  CHECK_GT(conv_cache.size(0), 0) << "conv cache must have positive batch dim";
  CHECK_EQ(ssm_cache.size(0) % conv_cache.size(0), 0)
      << "ssm cache checkpoint layout mismatch, ssm_rows=" << ssm_cache.size(0)
      << ", conv_rows=" << conv_cache.size(0);
  return ssm_cache.size(0) / conv_cache.size(0);
}

torch::Tensor build_linear_state_base_indices(
    const torch::Tensor& logical_state_indices,
    int64_t checkpoint_stride) {
  if (checkpoint_stride == 1) {
    return logical_state_indices;
  }
  return logical_state_indices * checkpoint_stride;
}

torch::Tensor expand_sequence_tensor_to_batch(const torch::Tensor& tensor,
                                              int64_t target_batch,
                                              const char* tensor_name) {
  CHECK(tensor.defined()) << tensor_name << " must be defined";
  CHECK_EQ(tensor.dim(), 1) << tensor_name << " must be a 1D tensor.";
  const int64_t source_batch = tensor.size(0);
  if (source_batch == target_batch) {
    return tensor.contiguous();
  }
  CHECK_GT(source_batch, 0) << tensor_name << " must not be empty.";
  CHECK_EQ(target_batch % source_batch, 0)
      << tensor_name << " cannot be expanded from " << source_batch << " to "
      << target_batch;
  const int64_t repeat_count = target_batch / source_batch;
  return tensor.unsqueeze(1)
      .expand({source_batch, repeat_count})
      .reshape({target_batch})
      .contiguous();
}

torch::Tensor run_spec_verify_conv(
    const torch::Tensor& mixed_qkv,
    torch::Tensor& conv_cache,
    const torch::Tensor& logical_state_indices,
    const std::vector<int64_t>& num_accepted_host,
    const torch::Tensor& q_cu_seq_lens,
    const torch::Tensor& conv_weight,
    int32_t conv_kernel_size) {
  const int64_t batch_size = mixed_qkv.size(0);
  const int64_t dim = mixed_qkv.size(1);
  const int64_t seq_len = mixed_qkv.size(2);
  if (qwen35_mtp_debug_enabled()) {
    LOG(INFO) << "[Qwen3.5 MTP debug] spec conv enter: mixed_qkv="
              << mixed_qkv.sizes() << ", conv_cache=" << conv_cache.sizes()
              << ", logical_state_indices=" << logical_state_indices.sizes()
              << ", num_accepted_host.size()=" << num_accepted_host.size()
              << ", q_cu_seq_lens=" << q_cu_seq_lens.sizes()
              << ", conv_weight=" << conv_weight.sizes()
              << ", conv_kernel_size=" << conv_kernel_size;
  }
  qwen35_mtp_debug_sync("spec_conv_enter");
  // conv_cache layout: [num_blocks, dim, state_len] (matches 0526 ref +
  // causal_conv1d_update kernel). state_len lives in dim 2.
  const int64_t expanded_state_len = conv_cache.size(2);
  CHECK_EQ(q_cu_seq_lens.numel(), batch_size + 1)
      << "spec conv q_cu_seq_lens must be cumulative.";
  CHECK_EQ(expanded_state_len, conv_kernel_size - 1 + seq_len - 1)
      << "unexpected speculative conv cache len, expected "
      << (conv_kernel_size - 1 + seq_len - 1) << ", got " << expanded_state_len;
  CHECK_GE(conv_kernel_size, 2)
      << "Qwen3.5 speculative conv expects kernel size >= 2";
  CHECK_EQ(conv_cache.size(1), dim) << "spec conv cache dim mismatch";

  torch::Tensor weight = conv_weight;
  if (weight.dim() == 3) {
    CHECK_EQ(weight.size(1), 1)
        << "spec conv expects weight [dim, 1, width] or [dim, width]";
    weight = weight.squeeze(1);
  }
  CHECK_EQ(weight.dim(), 2)
      << "spec conv expects weight [dim, width], got " << weight.sizes();
  CHECK_EQ(weight.size(0), dim) << "spec conv weight dim mismatch";
  CHECK_EQ(weight.size(1), conv_kernel_size)
      << "spec conv weight width mismatch";

  auto state_indices =
      expand_sequence_tensor_to_batch(
          logical_state_indices, batch_size, "logical_state_indices")
          .to(mixed_qkv.device(), torch::kLong)
          .contiguous();
  const std::vector<int64_t> accepted_host = expand_int_vector_to_batch(
      num_accepted_host, batch_size, "num_accepted_tokens");

  auto x_f32 = mixed_qkv.to(torch::kFloat32);
  auto weight_f32 = weight.to(torch::kFloat32);
  auto output_f32 = torch::empty_like(x_f32);
  auto next_states =
      torch::empty({batch_size, dim, expanded_state_len}, conv_cache.options());
  const int64_t history_len = conv_kernel_size - 1;
  const int64_t old_prefix_len = expanded_state_len - seq_len;

  for (int64_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
    const int64_t accepted_count = accepted_host[static_cast<size_t>(seq_idx)];
    CHECK_GE(accepted_count, 1)
        << "num_accepted_tokens must be >= 1 for spec verify";
    CHECK_LE(accepted_count, seq_len)
        << "num_accepted_tokens must be <= validate token count";
    const int64_t accepted_offset = accepted_count - 1;

    auto cache_idx = state_indices.select(0, seq_idx).reshape({1});
    auto full_state = conv_cache.index_select(/*dim=*/0, cache_idx)
                          .select(/*dim=*/0, 0)
                          .to(torch::kFloat32)
                          .contiguous();
    auto history =
        full_state.narrow(/*dim=*/-1, accepted_offset, history_len).clone();

    for (int64_t token_idx = 0; token_idx < seq_len; ++token_idx) {
      auto x_t = x_f32.select(0, seq_idx).select(1, token_idx);
      auto window = torch::cat({history, x_t.unsqueeze(-1)}, /*dim=*/-1);
      auto token_out = (window * weight_f32).sum(/*dim=*/-1);
      token_out = torch::silu(token_out);
      output_f32.select(0, seq_idx).select(1, token_idx).copy_(token_out);
      history =
          window.narrow(/*dim=*/-1, /*start=*/1, history_len).contiguous();
    }

    auto next_state =
        torch::zeros({dim, expanded_state_len}, conv_cache.options());
    if (old_prefix_len > 0) {
      next_state.narrow(/*dim=*/-1, /*start=*/0, /*length=*/old_prefix_len)
          .copy_(full_state
                     .narrow(/*dim=*/-1,
                             /*start=*/accepted_offset + 1,
                             /*length=*/old_prefix_len)
                     .to(conv_cache.scalar_type()));
    }
    next_state.narrow(/*dim=*/-1, /*start=*/old_prefix_len, /*length=*/seq_len)
        .copy_(mixed_qkv.select(0, seq_idx).to(conv_cache.scalar_type()));
    next_states.select(0, seq_idx).copy_(next_state);
  }

  qwen35_mtp_debug_sync("spec_conv_before_cache_write");
  conv_cache.index_copy_(/*dim=*/0, state_indices, next_states);
  qwen35_mtp_debug_sync("spec_conv_after_cache_write");
  auto output = output_f32.to(mixed_qkv.scalar_type());
  qwen35_mtp_debug_sync("spec_conv_exit");
  return output;
}

torch::Tensor run_spec_verify_gated_delta_rule(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor g,
    torch::Tensor beta,
    torch::Tensor& ssm_cache,
    const torch::Tensor& checkpoint_indices,
    const std::vector<int64_t>& num_accepted_host,
    const torch::Tensor& cu_seq_lens,
    const std::vector<int32_t>& q_seq_lens_vec,
    bool fla_ssm_state_layout,
    double scale) {
  const auto device = value.device();
  const int64_t batch_size = value.size(0);
  const int64_t seq_len = value.size(1);
  if (qwen35_mtp_debug_enabled()) {
    LOG(INFO) << "[Qwen3.5 MTP debug] spec gdn enter: q=" << query.sizes()
              << ", k=" << key.sizes() << ", v=" << value.sizes()
              << ", g=" << g.sizes() << ", beta=" << beta.sizes()
              << ", ssm_cache=" << ssm_cache.sizes()
              << ", checkpoint_indices=" << checkpoint_indices.sizes()
              << ", num_accepted_host.size()=" << num_accepted_host.size()
              << ", cu_seq_lens=" << cu_seq_lens.sizes()
              << ", fla_ssm_state_layout=" << fla_ssm_state_layout;
  }
  qwen35_mtp_debug_sync("spec_gdn_enter");
  CHECK_EQ(cu_seq_lens.numel(), batch_size + 1)
      << "GDN spec verify cu_seq_lens must be cumulative.";
  CHECK_EQ(q_seq_lens_vec.size(), static_cast<size_t>(batch_size))
      << "GDN spec verify q_seq_lens_vec must be per sequence.";
  for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    CHECK_EQ(q_seq_lens_vec[batch_idx], seq_len)
        << "Qwen3.5 spec verify fused recurrent path expects dense "
           "same-length validate tokens.";
  }

  const int64_t value_num_heads = value.size(-2);
  CHECK_EQ(ssm_cache.size(1), value_num_heads)
      << "GDN spec verify ssm cache head mismatch";
  if (fla_ssm_state_layout) {
    CHECK_EQ(ssm_cache.size(2), key.size(-1))
        << "GDN spec verify ssm cache key dim mismatch";
    CHECK_EQ(ssm_cache.size(3), value.size(-1))
        << "GDN spec verify ssm cache value dim mismatch";
  } else {
    CHECK_EQ(ssm_cache.size(2), value.size(-1))
        << "GDN spec verify legacy ssm cache value dim mismatch";
    CHECK_EQ(ssm_cache.size(3), key.size(-1))
        << "GDN spec verify legacy ssm cache key dim mismatch";
  }

  auto checkpoint_indices_long =
      checkpoint_indices.to(device, torch::kLong).contiguous();
  const std::vector<int64_t> accepted_host = expand_int_vector_to_batch(
      num_accepted_host, batch_size, "num_accepted_tokens");

  auto output = torch::empty_like(value);
  const double l2_eps = 1e-6;
  for (int64_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
    const int64_t accepted_count = accepted_host[static_cast<size_t>(seq_idx)];
    CHECK_GE(accepted_count, 1)
        << "num_accepted_tokens must be >= 1 for spec verify";
    CHECK_LE(accepted_count, seq_len)
        << "num_accepted_tokens must be <= validate token count";

    auto init_state_index = checkpoint_indices_long.select(0, seq_idx)
                                .select(0, accepted_count - 1)
                                .reshape({1});
    // The MUSA mate GDN kernels (mate prefill + in-house fused decode) persist
    // the recurrent SSM state transposed relative to the flash-linear-attention
    // [HV, K, V] layout this loop's math expects. Because K == V for Qwen3.5
    // (both 128) the shape check above cannot detect the mismatch, so transpose
    // explicitly when the cache is in fla layout. Without this the spec-verify
    // logits degenerate (repeated newlines -> early EOS) even though the
    // per-token recurrent math is otherwise correct.
    const bool transpose_state = fla_ssm_state_layout;
    auto recurrent_state = ssm_cache.index_select(/*dim=*/0, init_state_index)
                               .to(torch::kFloat32)
                               .contiguous();
    if (transpose_state) {
      recurrent_state = recurrent_state.transpose(-1, -2).contiguous();
    }

    for (int64_t token_idx = 0; token_idx < seq_len; ++token_idx) {
      auto q_t = query.select(0, seq_idx).select(0, token_idx).unsqueeze(0);
      auto k_t = key.select(0, seq_idx).select(0, token_idx).unsqueeze(0);
      auto v_t = value.select(0, seq_idx)
                     .select(0, token_idx)
                     .unsqueeze(0)
                     .to(torch::kFloat32);
      q_t = l2norm(q_t, /*dim=*/-1, /*eps=*/l2_eps).to(torch::kFloat32);
      k_t = l2norm(k_t, /*dim=*/-1, /*eps=*/l2_eps).to(torch::kFloat32);
      q_t = repeat_tensor_heads(q_t, value_num_heads, /*head_dim=*/1) *
            static_cast<float>(scale);
      k_t = repeat_tensor_heads(k_t, value_num_heads, /*head_dim=*/1);

      auto g_t = g.select(0, seq_idx)
                     .select(0, token_idx)
                     .to(torch::kFloat32)
                     .view({1, value_num_heads, 1, 1});
      auto beta_t = beta.select(0, seq_idx)
                        .select(0, token_idx)
                        .to(torch::kFloat32)
                        .view({1, value_num_heads, 1});

      qwen35_mtp_debug_sync("spec_gdn_before_exp");
      recurrent_state = recurrent_state * g_t.exp();
      qwen35_mtp_debug_sync("spec_gdn_after_exp");
      auto kv_mem = (recurrent_state * k_t.unsqueeze(-1)).sum(/*dim=*/-2);
      auto delta = (v_t - kv_mem) * beta_t;
      recurrent_state =
          recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2);
      auto token_out = (recurrent_state * q_t.unsqueeze(-1)).sum(/*dim=*/-2);
      output.select(0, seq_idx)
          .select(0, token_idx)
          .copy_(token_out.select(0, 0).to(output.scalar_type()));

      auto store_index = checkpoint_indices_long.select(0, seq_idx)
                             .select(0, token_idx)
                             .reshape({1});
      torch::Tensor state_to_store =
          transpose_state ? recurrent_state.transpose(-1, -2).contiguous()
                          : recurrent_state;
      ssm_cache.index_copy_(
          /*dim=*/0, store_index, state_to_store.to(ssm_cache.scalar_type()));
      qwen35_mtp_debug_sync("spec_gdn_after_state_write");
    }
  }

  qwen35_mtp_debug_sync("spec_gdn_exit");
  return output;
}

}  // namespace

Qwen3GatedDeltaNetBaseImpl::Qwen3GatedDeltaNetBaseImpl(
    const ModelArgs& args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options) {
  tp_size_ = parallel_args.tp_group_->world_size();
  rank_ = parallel_args.tp_group_->rank();
  num_k_heads_ = args.linear_num_key_heads();
  num_v_heads_ = args.linear_num_value_heads();
  head_k_dim_ = args.linear_key_head_dim();
  head_v_dim_ = args.linear_value_head_dim();
  k_size_ = num_k_heads_ * head_k_dim_;
  v_size_ = num_v_heads_ * head_v_dim_;
  conv_kernel_size_ = args.linear_conv_kernel_dim();

  // Shared causal conv projection over mixed QKV states.
  conv1d_ = register_module("conv1d",
                            ColumnParallelLinear(args.linear_conv_kernel_dim(),
                                                 k_size_ * 2 + v_size_,
                                                 /*bias=*/false,
                                                 /*gather_output=*/false,
                                                 quant_args,
                                                 parallel_args.tp_group_,
                                                 options));

  auto opts = options.dtype(torch::kFloat32);
  dt_bias_ = register_parameter("dt_bias",
                                torch::ones({num_v_heads_ / tp_size_}, opts),
                                /*requires_grad=*/false);

  A_log_ = register_parameter("A_log",
                              torch::empty({num_v_heads_ / tp_size_}, opts),
                              /*requires_grad=*/false);

  // Output projection and gated RMSNorm shared by hybrid variants.
  o_proj_ = register_module("out_proj",
                            RowParallelLinear(v_size_,
                                              args.hidden_size(),
                                              /*bias=*/false,
                                              /*input_is_parallelized=*/true,
                                              /*if_reduce_results=*/true,
                                              quant_args,
                                              parallel_args.tp_group_,
                                              options));

  norm_ = register_module(
      "norm", RmsNormGated(head_v_dim_, args.rms_norm_eps(), options));
}

void Qwen3GatedDeltaNetBaseImpl::load_common_state_dict(
    const StateDict& state_dict) {
  const int64_t rank = rank_;
  const int64_t world_size = tp_size_;
  const int32_t shard_tensor_count = 3;
  const std::vector<int64_t> shard_sizes = {
      k_size_ / tp_size_, k_size_ / tp_size_, v_size_ / tp_size_};

  if (auto w = state_dict.get_tensor("conv1d.weight"); w.defined()) {
    conv1d_->load_state_dict(
        StateDict({{"weight", w.squeeze(1)}}), shard_tensor_count, shard_sizes);
  }
  o_proj_->load_state_dict(state_dict.get_dict_with_prefix("out_proj."));
  if (auto w = state_dict.get_tensor("norm.weight"); w.defined()) {
    norm_->load_state_dict(StateDict({{"weight", w}}));
  }
  LOAD_SHARDED_WEIGHT(dt_bias, 0);
  LOAD_SHARDED_WEIGHT(A_log, 0);
}

void Qwen3GatedDeltaNetBaseImpl::verify_common_loaded_weights(
    const std::string& prefix) const {
  CHECK(dt_bias_is_loaded_)
      << "Missing required weight after all shards loaded: " << prefix
      << "dt_bias";
  CHECK(A_log_is_loaded_) << "Missing required weight after all shards loaded: "
                          << prefix << "A_log";
}

std::pair<torch::Tensor, torch::Tensor>
Qwen3GatedDeltaNetBaseImpl::project_padded_inputs(
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata) {
  if (attn_metadata.is_prefill || attn_metadata.is_chunked_prefill) {
    auto [qkvz_flat, ba_flat] = project_flat_inputs(hidden_states);
    return {reshape_qkvz_with_pad(attn_metadata, qkvz_flat),
            reshape_qkvz_with_pad(attn_metadata, ba_flat)};
  }
  return project_decode_inputs(hidden_states);
}

torch::Tensor Qwen3GatedDeltaNetBaseImpl::forward(
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params) {
  // Save original hidden_states size for potential padding later
  const int64_t original_num_tokens = hidden_states.size(0);
  auto [qkvz_padded, ba_padded] =
      project_padded_inputs(hidden_states, attn_metadata);
  int64_t batch_size = qkvz_padded.size(0);
  int64_t seq_len = qkvz_padded.size(1);

  torch::Tensor qkvz_flat =
      qkvz_padded.view({batch_size * seq_len, qkvz_padded.size(-1)});
  torch::Tensor ba_flat =
      ba_padded.view({batch_size * seq_len, ba_padded.size(-1)});
  xllm::kernel::FusedQkvzbaSplitReshapeParams fused_params;
  fused_params.mixed_qkvz = qkvz_flat;
  fused_params.mixed_ba = ba_flat;
  fused_params.num_heads_qk = static_cast<int32_t>(num_k_heads_ / tp_size_);
  fused_params.num_heads_v = static_cast<int32_t>(num_v_heads_ / tp_size_);
  fused_params.head_qk = static_cast<int32_t>(head_k_dim_);
  fused_params.head_v = static_cast<int32_t>(head_v_dim_);

#if defined(USE_CUDA) || defined(USE_MUSA)
  // Lazily size grow-only persistent output buffers so the kernel can
  // populate them via in-place `copy_` calls instead of allocating new
  // tensors mid-stream-capture. Allocation only happens on the first
  // forward (executed eagerly before MUSA graph capture begins) and
  // whenever a larger bucket is seen.
  const int64_t local_nk = num_k_heads_ / tp_size_;
  const int64_t local_nv = num_v_heads_ / tp_size_;
  const int64_t expected_m = batch_size * seq_len;
  const int64_t expected_qkv_dim =
      2 * local_nk * head_k_dim_ + local_nv * head_v_dim_;
  const int64_t expected_z_dim = local_nv * head_v_dim_;
  const auto opts = qkvz_flat.options();
  const auto opts_ba = ba_flat.options();
  const auto grow_2d = [](torch::Tensor& buf,
                          int64_t m,
                          int64_t d,
                          const torch::TensorOptions& options) {
    const bool needs = !buf.defined() || buf.size(0) < M || buf.size(1) != D ||
                       buf.scalar_type() != options.dtype().toScalarType() ||
                       buf.device() != options.device();
    if (needs) {
      const int64_t target_M = buf.defined() ? std::max(M, buf.size(0)) : M;
      buf = torch::empty({target_M, D}, options);
    }
  };
  grow_2d(mixed_qkv_out_buf_, expected_M, expected_qkv_dim, opts);
  grow_2d(z_out_buf_, expected_M, expected_z_dim, opts);
  grow_2d(b_out_buf_, expected_M, local_nv, opts_ba);
  grow_2d(a_out_buf_, expected_M, local_nv, opts_ba);
  fused_params.mixed_qkv_out_buf = mixed_qkv_out_buf_;
  fused_params.z_out_buf = z_out_buf_;
  fused_params.b_out_buf = b_out_buf_;
  fused_params.a_out_buf = a_out_buf_;
#endif

  torch::Tensor mixed_qkv, z, b, a;
  std::tie(mixed_qkv, z, b, a) =
      xllm::kernel::fused_qkvzba_split_reshape_cat(fused_params);

  mixed_qkv = mixed_qkv.view({batch_size, seq_len, mixed_qkv.size(-1)});
  z = z.view({batch_size, seq_len, num_v_heads_ / tp_size_, head_v_dim_});
  b = b.view({batch_size, seq_len, num_v_heads_ / tp_size_});
  a = a.view({batch_size, seq_len, num_v_heads_ / tp_size_});

  torch::Tensor conv_cache = kv_cache.get_conv_cache();
  torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
  torch::Device device = mixed_qkv.device();
  torch::Tensor conv_weight = conv1d_->weight();
  torch::Tensor logical_state_indices =
      get_linear_state_indices(input_params, device);
  const int64_t checkpoint_stride =
      get_checkpoint_stride(conv_cache, ssm_cache);
  torch::Tensor linear_state_base_indices =
      build_linear_state_base_indices(logical_state_indices, checkpoint_stride);
  const bool use_spec_verify = input_params.is_spec_verify;
  const bool is_any_prefill =
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;
#if defined(USE_CUDA) || defined(USE_MUSA)
  const bool decode_eligible = !attn_metadata.is_prefill && !use_spec_verify &&
                               seq_len == 1 && checkpoint_stride == 1;
  // Production defaults: fused decode on, mate decode/prefill off.
  const bool use_fused_gdn_decode = kEnableFusedGdnDecode && decode_eligible;
  const bool use_mate_gdn_decode =
      kEnableMateGdnDecode && decode_eligible && !use_fused_gdn_decode;
#else
  const bool use_fused_gdn_decode = false;
  const bool use_mate_gdn_decode = false;
#endif
  // Both fused and mate decode paths consume the flat [tokens, dim] mixed_qkv
  // and split q/k/v via strided reads (no contiguous() copies).
  const bool use_flat_mixed_qkv_decode =
      use_fused_gdn_decode || use_mate_gdn_decode;

  if (!use_spec_verify && attn_metadata.is_prefill &&
      !attn_metadata.is_chunked_prefill) {
    // xllm_0526 (xllm_0623_build container) prefill fast path: depthwise causal
    // conv via torch::conv1d for the output, and index_put_ on conv_cache to
    // store the post-prefill conv state. Does NOT call causal_conv1d_update on
    // prefill -- the kernel's slow-path .item() host syncs fail on MUSA
    // ("operation not permitted" / "illegal memory access").
    mixed_qkv = mixed_qkv.transpose(1, 2);
    const torch::Tensor conv_in = mixed_qkv;
    const int64_t history_len = conv_kernel_size_ - 1;
    const int64_t cache_state_len = conv_cache.size(2);
    CHECK_GE(cache_state_len, history_len)
        << "conv cache state len must be at least kernel_size - 1";
    // history_state is the last kernel_size-1 real prompt tokens. Under MTP the
    // conv cache is wider; the extra speculative slots stay zero on the right.
    torch::Tensor history_state =
        (seq_len < history_len)
            ? torch::nn::functional::pad(conv_in,
                                         torch::nn::functional::PadFuncOptions(
                                             {0, history_len - seq_len}))
        : (seq_len > history_len)
            ? conv_in.narrow(-1, seq_len - history_len, history_len)
            : conv_in;
    torch::Tensor conv_state;
    if (cache_state_len == history_len) {
      conv_state = history_state;
    } else {
      conv_state = torch::zeros({batch_size, conv_in.size(1), cache_state_len},
                                conv_in.options());
      conv_state.narrow(/*dim=*/-1, /*start=*/0, /*length=*/history_len)
          .copy_(history_state);
    }
    conv_cache.index_put_({logical_state_indices},
                          conv_state.contiguous().to(conv_cache.dtype()));
    torch::Tensor bias;
    auto conv_output =
        torch::conv1d(conv_in,
                      conv_weight.unsqueeze(1).to(device),
                      bias,
                      /*stride=*/std::vector<int64_t>{1},
                      /*padding=*/std::vector<int64_t>{conv_kernel_size_ - 1},
                      /*dilation=*/std::vector<int64_t>{1},
                      /*groups=*/static_cast<int64_t>(conv_in.size(1)));
    mixed_qkv = torch::silu(conv_output.slice(2, 0, seq_len));
  } else if (!use_spec_verify && is_any_prefill) {
    torch::IntArrayRef num_accepted_tokens_opt;
    std::vector<int64_t> linear_state_indices_vec(
        input_params.embedding.linear_state_ids.begin(),
        input_params.embedding.linear_state_ids.end());
    torch::Tensor conv_input = reshape_qkvz_unpad(attn_metadata, mixed_qkv);
    mixed_qkv = xllm::kernel::causal_conv1d(
        conv_input,
        conv_weight,
        conv_cache,
        std::optional<torch::Tensor>(),  // bias (no bias for qwen3)
        torch::IntArrayRef(input_params.parallel.query_start_loc),
        torch::IntArrayRef(linear_state_indices_vec),
        torch::IntArrayRef(input_params.parallel.has_initial_state),
        num_accepted_tokens_opt,
        1,   // activation_mode (silu)
        -1,  // pad_slot_id
        0);  // run_mode forward

    mixed_qkv = reshape_qkvz_with_pad(attn_metadata, mixed_qkv);
    mixed_qkv = mixed_qkv.transpose(1, 2);
  } else if (use_spec_verify) {
    CHECK(input_params.num_accepted_tokens.defined())
        << "num_accepted_tokens must be populated for Qwen3.5 spec verify";
    if (qwen35_mtp_debug_enabled()) {
      LOG(INFO) << "[Qwen3.5 MTP debug] forward spec conv: mixed_qkv="
                << mixed_qkv.sizes() << ", conv_cache=" << conv_cache.sizes()
                << ", logical_state_indices=" << logical_state_indices.sizes()
                << ", num_accepted_tokens="
                << input_params.num_accepted_tokens.sizes()
                << ", q_cu_seq_lens=" << attn_metadata.q_cu_seq_lens.sizes()
                << ", max_query_len=" << attn_metadata.max_query_len;
    }
    qwen35_mtp_debug_sync("forward_before_spec_conv");
    torch::Tensor pre_conv_mixed_qkv = mixed_qkv.transpose(1, 2);
    mixed_qkv = run_spec_verify_conv(pre_conv_mixed_qkv,
                                     conv_cache,
                                     logical_state_indices,
                                     input_params.num_accepted_tokens_host,
                                     attn_metadata.q_cu_seq_lens,
                                     conv_weight,
                                     conv_kernel_size_);
    qwen35_mtp_debug_sync("forward_after_spec_conv");
  } else {
    // xllm_0526 (xllm_0623_build container) decode fast path: standard
    // single-token causal_conv1d_update with conv_state = conv_cache directly.
    // conv_cache layout is [num_blocks, dim, state_len] (kernel expectation),
    // so no transpose is needed.
    xllm::kernel::CausalConv1dUpdateParams conv1d_params;
    conv1d_params.x = mixed_qkv.reshape({-1, mixed_qkv.size(-1)});
    conv1d_params.conv_state = conv_cache;
    conv1d_params.weight = conv_weight;
    conv1d_params.conv_state_indices = logical_state_indices;
    conv1d_params.block_idx_last_scheduled_token =
        std::optional<torch::Tensor>();
    conv1d_params.initial_state_idx = std::optional<torch::Tensor>();
    conv1d_params.query_start_loc = attn_metadata.q_cu_seq_lens;
    conv1d_params.max_query_len = attn_metadata.max_query_len;
#if defined(USE_CUDA) || defined(USE_MUSA)
    // Provide a persistent, grow-only output buffer so the kernel takes its
    // graph-safe fused fast path (causal_conv1d_decode_fused) instead of the
    // libtorch `.to(fp32) ... torch::empty_like` slow path that aborts
    // MUSA graph capture with "operation not permitted when stream is
    // capturing".
    {
      const auto& x_in = conv1d_params.x;
      const int64_t m = x_in.size(0);
      const int64_t d = x_in.size(1);
      const bool needs =
          !conv1d_decode_out_buf_.defined() ||
          conv1d_decode_out_buf_.size(0) < M ||
          conv1d_decode_out_buf_.size(1) != D ||
          conv1d_decode_out_buf_.scalar_type() != x_in.scalar_type() ||
          conv1d_decode_out_buf_.device() != x_in.device();
      if (needs) {
        const int64_t target_M =
            conv1d_decode_out_buf_.defined()
                ? std::max(M, conv1d_decode_out_buf_.size(0))
                : M;
        conv1d_decode_out_buf_ = torch::empty({target_M, D}, x_in.options());
      }
      conv1d_params.output_buf =
          conv1d_decode_out_buf_.narrow(/*dim=*/0, /*start=*/0, /*length=*/M);
    }
#endif
    mixed_qkv = xllm::kernel::causal_conv1d_update(conv1d_params);
    if (use_flat_mixed_qkv_decode) {
      // Keep flat [tokens, dim] for the packed mate / in-house fused decode
      // kernels; both consume the flat mixed_qkv directly via strided reads.
      CHECK(mixed_qkv.stride(-1) == 1)
          << "GDN decode mixed_qkv last dim must be contiguous";
    } else {
      // Reshape back to 3D [batch_size, seq_len, dim], then transpose for the
      // process_mixed_qkv path which expects [batch_size, dim, seq_len].
      mixed_qkv =
          mixed_qkv.view({batch_size, -1, mixed_qkv.size(-1)}).contiguous();
      mixed_qkv = mixed_qkv.transpose(1, 2);
    }
  }
  const bool fla_ssm_state_layout = use_fla_ssm_state_layout();
  const bool use_fused_sigmoid_gdn_decode =
      fla_ssm_state_layout && !use_spec_verify && !is_any_prefill &&
      checkpoint_stride == 1;
  torch::Tensor g;
  torch::Tensor beta;
  // Compute gated delta net decay and beta terms.
  if (use_spec_verify || attn_metadata.is_chunked_prefill ||
      checkpoint_stride > 1) {
    qwen35_mtp_debug_sync("forward_before_gating");
    beta = torch::sigmoid(b);
    torch::Tensor A_log_exp = A_log_.exp();
    torch::Tensor a_float = a.to(torch::kFloat32);
    torch::Tensor a_plus_dt = a_float + dt_bias_;
    torch::Tensor softplus_out = torch::nn::functional::softplus(
        a_plus_dt,
        torch::nn::functional::SoftplusFuncOptions().beta(1.0).threshold(20.0));
    g = -A_log_exp * softplus_out;
    g = g.to(a.dtype()).contiguous();
    if (qwen35_mtp_debug_enabled() && use_spec_verify) {
      LOG(INFO) << "[Qwen3.5 MTP debug] forward after gating: a=" << a.sizes()
                << ", b=" << b.sizes() << ", g=" << g.sizes()
                << ", beta=" << beta.sizes();
    }
    qwen35_mtp_debug_sync("forward_after_gating");
  } else if (attn_metadata.is_prefill) {
    xllm::kernel::FusedGdnGatingParams gdn_params;
    gdn_params.A_log = A_log_;
    gdn_params.a = a.contiguous().view({-1, a.size(-1)});
    gdn_params.b = b.contiguous().view({-1, b.size(-1)});
    gdn_params.dt_bias = dt_bias_;
    gdn_params.beta = 1.0f;
    gdn_params.threshold = 20.0f;
    std::tie(g, beta) = xllm::kernel::fused_gdn_gating(gdn_params);
    g = g.squeeze(0).contiguous().view({batch_size, seq_len, a.size(-1)});
    beta = beta.squeeze(0).contiguous().view({batch_size, seq_len, b.size(-1)});
  } else if (!use_flat_mixed_qkv_decode) {
    xllm::kernel::FusedGdnGatingParams gdn_params;
    gdn_params.A_log = A_log_;
    gdn_params.a = a.view({-1, a.size(-1)});
    gdn_params.b = b.view({-1, b.size(-1)});
    gdn_params.dt_bias = dt_bias_;
    gdn_params.beta = 1.0f;
    gdn_params.threshold = 20.0f;
    std::tie(g, beta) = xllm::kernel::fused_gdn_gating(gdn_params);
  }
  torch::Tensor processed_q;
  torch::Tensor processed_k;
  torch::Tensor processed_v;
  if (!use_flat_mixed_qkv_decode) {
    std::tie(processed_q, processed_k, processed_v) =
        process_mixed_qkv(mixed_qkv);
  }
  torch::Tensor core_attn_out;
  torch::Tensor last_recurrent_state;
#if defined(USE_CUDA) || defined(USE_MUSA)
  const bool use_mate_gdn_prefill =
      kEnableMateGdnPrefill && attn_metadata.is_prefill && !use_spec_verify;
#else
  const bool use_mate_gdn_prefill = false;
#endif
  // Apply chunked or recurrent gated-delta attention and update caches.
  if (use_mate_gdn_prefill) {
#if defined(USE_CUDA) || defined(USE_MUSA)
    // xllm_0526: pass full [B, T, H, D] tensors; mate returns VK state layout.
    xllm::kernel::MateGatedDeltaRulePrefillParams mate_params;
    mate_params.q = processed_q;
    mate_params.k = processed_k;
    mate_params.v = processed_v;
    mate_params.g = g;
    mate_params.beta = beta;
    mate_params.scale =
        1.0 / std::sqrt(static_cast<double>(processed_q.size(-1)));
    torch::Tensor mate_final_state;
    std::tie(core_attn_out, mate_final_state) =
        xllm::kernel::mate_gated_delta_rule_prefill(mate_params);
    ssm_cache.index_put_({linear_state_base_indices},
                         mate_final_state.to(ssm_cache.dtype()));
#endif
  } else if (!use_spec_verify && attn_metadata.is_prefill &&
             !attn_metadata.is_chunked_prefill) {
    // xllm_0526 prefill path: chunk_gated_delta_rule (MATE-backed on MUSA).
    xllm::kernel::ChunkGatedDeltaRuleParams chunk_gated_delta_params;
    chunk_gated_delta_params.q = processed_q;
    chunk_gated_delta_params.k = processed_k;
    chunk_gated_delta_params.v = processed_v;
    chunk_gated_delta_params.g = g;
    chunk_gated_delta_params.beta = beta;
    torch::Tensor initial_state_tensor =
        torch::index_select(ssm_cache, 0, linear_state_base_indices);
    initial_state_tensor.fill_(0.0);
    chunk_gated_delta_params.initial_state = initial_state_tensor;
    chunk_gated_delta_params.output_final_state = true;
    chunk_gated_delta_params.cu_seqlens = attn_metadata.q_cu_seq_lens;
    chunk_gated_delta_params.head_first = false;
    chunk_gated_delta_params.use_qk_l2norm_in_kernel = true;
    std::tie(core_attn_out, last_recurrent_state) =
        xllm::kernel::chunk_gated_delta_rule(chunk_gated_delta_params);
    ssm_cache.index_put_(
        {linear_state_base_indices},
        last_recurrent_state.transpose(-1, -2).to(ssm_cache.dtype()));
  } else if (use_spec_verify) {
    torch::Tensor spec_linear_state_base_indices =
        expand_sequence_tensor_to_batch(
            linear_state_base_indices, batch_size, "linear_state_base_indices");
    torch::Tensor step_offsets =
        torch::arange(seq_len,
                      torch::TensorOptions()
                          .dtype(spec_linear_state_base_indices.dtype())
                          .device(device));
    torch::Tensor checkpoint_indices =
        spec_linear_state_base_indices.unsqueeze(1) + step_offsets;
    double scale = 1.0 / std::sqrt(static_cast<float>(processed_q.size(-1)));
    if (qwen35_mtp_debug_enabled()) {
      LOG(INFO) << "[Qwen3.5 MTP debug] forward spec gdn: processed_q="
                << processed_q.sizes()
                << ", processed_k=" << processed_k.sizes()
                << ", processed_v=" << processed_v.sizes()
                << ", ssm_cache=" << ssm_cache.sizes()
                << ", checkpoint_indices=" << checkpoint_indices.sizes()
                << ", num_accepted_host.size()="
                << input_params.num_accepted_tokens_host.size()
                << ", q_cu_seq_lens=" << attn_metadata.q_cu_seq_lens.sizes();
    }
    qwen35_mtp_debug_sync("forward_before_spec_gdn");
    core_attn_out =
        run_spec_verify_gated_delta_rule(processed_q,
                                         processed_k,
                                         processed_v,
                                         g,
                                         beta,
                                         ssm_cache,
                                         checkpoint_indices,
                                         input_params.num_accepted_tokens_host,
                                         attn_metadata.q_cu_seq_lens,
                                         attn_metadata.q_seq_lens_vec,
                                         fla_ssm_state_layout,
                                         scale);
    qwen35_mtp_debug_sync("forward_after_spec_gdn");
  } else if (is_any_prefill) {
    CHECK_GE(attn_metadata.q_seq_lens_vec.size(),
             static_cast<size_t>(batch_size))
        << "q_seq_lens_vec must be populated for Qwen3.5 prefill.";
    std::vector<torch::Tensor> packed_q;
    std::vector<torch::Tensor> packed_k;
    std::vector<torch::Tensor> packed_v;
    std::vector<torch::Tensor> packed_g;
    std::vector<torch::Tensor> packed_beta;
    packed_q.reserve(batch_size);
    packed_k.reserve(batch_size);
    packed_v.reserve(batch_size);
    packed_g.reserve(batch_size);
    packed_beta.reserve(batch_size);
    for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      const int64_t valid_len = attn_metadata.q_seq_lens_vec[batch_idx];
      packed_q.emplace_back(
          processed_q[batch_idx].narrow(/*dim=*/0, /*start=*/0, valid_len));
      packed_k.emplace_back(
          processed_k[batch_idx].narrow(/*dim=*/0, /*start=*/0, valid_len));
      packed_v.emplace_back(
          processed_v[batch_idx].narrow(/*dim=*/0, /*start=*/0, valid_len));
      packed_g.emplace_back(
          g[batch_idx].narrow(/*dim=*/0, /*start=*/0, valid_len));
      packed_beta.emplace_back(
          beta[batch_idx].narrow(/*dim=*/0, /*start=*/0, valid_len));
    }
    torch::Tensor packed_processed_q = torch::cat(packed_q, 0).unsqueeze(0);
    torch::Tensor packed_processed_k = torch::cat(packed_k, 0).unsqueeze(0);
    torch::Tensor packed_processed_v = torch::cat(packed_v, 0).unsqueeze(0);
    torch::Tensor packed_g_tensor = torch::cat(packed_g, 0).unsqueeze(0);
    torch::Tensor packed_beta_tensor = torch::cat(packed_beta, 0).unsqueeze(0);

    xllm::kernel::MegaChunkGdnParams mega_chunk_gdn_params;
    mega_chunk_gdn_params.q = packed_processed_q;
    mega_chunk_gdn_params.k = packed_processed_k;
    mega_chunk_gdn_params.v = packed_processed_v;
    mega_chunk_gdn_params.g = packed_g_tensor;
    mega_chunk_gdn_params.beta = packed_beta_tensor;
    // Get initial state from ssm_cache for sequences with previous state
    // Shape: [batch_size, num_heads, head_k_dim, head_v_dim]
    torch::Tensor initial_state_tensor =
        torch::index_select(ssm_cache, 0, linear_state_base_indices);
    CHECK_EQ(input_params.parallel.has_initial_state.size(),
             input_params.embedding.linear_state_ids.size())
        << "has_initial_state must be sequence-scoped.";
    for (size_t i = 0; i < input_params.parallel.has_initial_state.size();
         ++i) {
      if (input_params.parallel.has_initial_state[i] == 0) {
        initial_state_tensor.select(0, static_cast<int64_t>(i)).fill_(0.0);
      }
    }
    if (!fla_ssm_state_layout && attn_metadata.is_chunked_prefill) {
      initial_state_tensor =
          initial_state_tensor.transpose(-1, -2).contiguous();
    }
    mega_chunk_gdn_params.initial_state = initial_state_tensor;
    mega_chunk_gdn_params.output_final_state = true;
    mega_chunk_gdn_params.cu_seqlens = attn_metadata.q_cu_seq_lens;
    mega_chunk_gdn_params.use_qk_l2norm_in_kernel = true;
    torch::Tensor packed_core_attn_out;
    std::tie(packed_core_attn_out, last_recurrent_state) =
        xllm::kernel::mega_chunk_gdn(mega_chunk_gdn_params);
    core_attn_out = torch::zeros_like(processed_v);
    int64_t packed_offset = 0;
    for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      const int64_t valid_len = attn_metadata.q_seq_lens_vec[batch_idx];
      core_attn_out[batch_idx]
          .narrow(/*dim=*/0, /*start=*/0, valid_len)
          .copy_(packed_core_attn_out[0].narrow(
              /*dim=*/0, packed_offset, valid_len));
      packed_offset += valid_len;
    }
    torch::Tensor state_to_store = fla_ssm_state_layout
                                       ? last_recurrent_state
                                       : last_recurrent_state.transpose(-1, -2);
    ssm_cache.index_put_({linear_state_base_indices},
                         state_to_store.to(ssm_cache.dtype()));
  } else if (checkpoint_stride > 1) {
    auto ssm_state =
        torch::index_select(ssm_cache, 0, linear_state_base_indices);
    if (!fla_ssm_state_layout) {
      ssm_state = ssm_state.transpose(-1, -2);
    }
    ssm_state = ssm_state.contiguous();
    std::tie(core_attn_out, last_recurrent_state) =
        torch_recurrent_gated_delta_rule(
            processed_q, processed_k, processed_v, g, beta, ssm_state);
    torch::Tensor state_to_store = fla_ssm_state_layout
                                       ? last_recurrent_state
                                       : last_recurrent_state.transpose(-1, -2);
    ssm_cache.index_put_({linear_state_base_indices},
                         state_to_store.to(ssm_cache.dtype()));
  } else if (use_fused_gdn_decode) {
#if defined(USE_CUDA) || defined(USE_MUSA)
    // In-house single-launch fused GDN decode kernel: fuses QKV split from
    // mixed_qkv, gating, L2-norm, scale, recurrent step, and in-place fp32
    // state I/O (Qwen3.5 mamba_ssm_dtype). Reuses
    // MateGatedDeltaRuleDecodeParams.
    xllm::kernel::MateGatedDeltaRuleDecodeParams fused_params;
    fused_params.mixed_qkv = mixed_qkv;
    fused_params.state = ssm_cache;
    fused_params.A_log = A_log_;
    fused_params.a = a.dim() == 3 ? a.select(1, 0) : a.squeeze(1);
    fused_params.dt_bias = dt_bias_;
    fused_params.b = b.dim() == 3 ? b.select(1, 0) : b.squeeze(1);
    fused_params.state_indices = logical_state_indices;
    fused_params.num_k_heads = num_k_heads_ / tp_size_;
    fused_params.num_v_heads = num_v_heads_ / tp_size_;
    fused_params.head_k_dim = head_k_dim_;
    fused_params.head_v_dim = head_v_dim_;
    fused_params.scale = 1.0 / std::sqrt(static_cast<double>(head_k_dim_));
    fused_params.use_qk_l2norm = true;
    // Provide a persistent, grow-only output buffer so the kernel skips its
    // `torch::empty({B, Hv, V}, ...)` allocation (which aborts MUSA graph
    // capture with "operation not permitted when stream is capturing") and
    // writes directly into pre-allocated storage.
    {
      const int64_t b = fused_params.mixed_qkv.size(0);
      const int64_t hv = num_v_heads_ / tp_size_;
      const int64_t v = head_v_dim_;
      const auto opts = mixed_qkv.options();
      const bool needs = !fused_gdn_decode_out_buf_.defined() ||
                         fused_gdn_decode_out_buf_.size(0) < B ||
                         fused_gdn_decode_out_buf_.size(1) != Hv ||
                         fused_gdn_decode_out_buf_.size(2) != V ||
                         fused_gdn_decode_out_buf_.scalar_type() !=
                             opts.dtype().toScalarType() ||
                         fused_gdn_decode_out_buf_.device() != opts.device();
      if (needs) {
        const int64_t target_B =
            fused_gdn_decode_out_buf_.defined()
                ? std::max(B, fused_gdn_decode_out_buf_.size(0))
                : B;
        fused_gdn_decode_out_buf_ = torch::empty({target_B, Hv, V}, opts);
      }
      fused_params.decode_output = fused_gdn_decode_out_buf_.narrow(
          /*dim=*/0, /*start=*/0, /*length=*/B);
    }
    core_attn_out =
        xllm::kernel::fused_gated_delta_rule_decode(fused_params).unsqueeze(0);
#endif
  } else if (use_mate_gdn_decode) {
#if defined(USE_CUDA) || defined(USE_MUSA)
    xllm::kernel::MateGatedDeltaRuleDecodeParams mate_params;
    mate_params.mixed_qkv = mixed_qkv;
    mate_params.state = ssm_cache;
    mate_params.A_log = A_log_;
    mate_params.a = a.dim() == 3 ? a.select(1, 0) : a.squeeze(1);
    mate_params.dt_bias = dt_bias_;
    mate_params.b = b.dim() == 3 ? b.select(1, 0) : b.squeeze(1);
    mate_params.state_indices = logical_state_indices;
    mate_params.num_k_heads = num_k_heads_ / tp_size_;
    mate_params.num_v_heads = num_v_heads_ / tp_size_;
    mate_params.head_k_dim = head_k_dim_;
    mate_params.head_v_dim = head_v_dim_;
    mate_params.scale = 1.0 / std::sqrt(static_cast<double>(head_k_dim_));
    mate_params.use_qk_l2norm = true;
    core_attn_out =
        xllm::kernel::mate_gated_delta_rule_decode(mate_params).unsqueeze(0);
#endif
  } else {
    double scale = 1.0 / std::sqrt(static_cast<float>(processed_q.size(-1)));
    if (fla_ssm_state_layout) {
      xllm::kernel::FusedSigmoidGatingDeltaRuleUpdateParams params;
      params.A_log = A_log_.contiguous();
      params.a = a.contiguous();
      params.dt_bias = dt_bias_.contiguous();
      params.q = processed_q.contiguous();
      params.k = processed_k.contiguous();
      params.v = processed_v.contiguous();
      params.b = b.contiguous();
      params.initial_state_source = ssm_cache;
      params.initial_state_indices = linear_state_base_indices.contiguous();
      params.cu_seqlens = attn_metadata.q_cu_seq_lens.contiguous();
      params.scale = static_cast<float>(scale);
      params.use_qk_l2norm_in_kernel = true;
      params.softplus_beta = 1.0f;
      params.softplus_threshold = 20.0f;
      core_attn_out =
          xllm::kernel::fused_sigmoid_gating_delta_rule_update(params);
    } else {
      // Align with xllm_0526 reference (xllm_0623_build container): on
      // CUDA/MUSA the qk L2-norm is fused into recurrent_gated_delta_rule
      // (the hand-written decode kernel). Other backends still normalize here.
      // The kernel ignores actual_seq_lengths for the standard 1-token decode
      // (uses slot index b..b+1); pass c10::nullopt to avoid extra .item()
      // syncs that fail on MUSA ("operation not permitted").
#if !defined(USE_CUDA) && !defined(USE_MUSA)
      processed_q = xllm::kernel::l2_norm(processed_q, /*eps=*/1e-6);
      processed_k = xllm::kernel::l2_norm(processed_k, /*eps=*/1e-6);
      auto zero = torch::zeros({1}, attn_metadata.q_seq_lens.options());
      torch::Tensor actual_seq_lengths =
          torch::cat({zero, attn_metadata.q_seq_lens}, 0);
#endif
      core_attn_out = xllm::kernel::recurrent_gated_delta_rule(
                          processed_q.reshape(
                              {-1, processed_q.size(-2), processed_q.size(-1)}),
                          processed_k.reshape(
                              {-1, processed_k.size(-2), processed_k.size(-1)}),
                          processed_v.reshape(
                              {-1, processed_v.size(-2), processed_v.size(-1)}),
                          ssm_cache,
                          beta.squeeze(0).contiguous(),
                          scale,
#if defined(USE_CUDA) || defined(USE_MUSA)
                          std::nullopt,
#else
                          actual_seq_lengths,
#endif
                          logical_state_indices,
                          std::nullopt,
                          g.squeeze(0).contiguous(),
                          std::nullopt)
                          .unsqueeze(0)
                          .contiguous();
    }
  }
  auto z_reshaped = z.view({-1, z.size(-1)});
  auto core_attn_out_reshaped =
      core_attn_out.view({-1, core_attn_out.size(-1)});
  auto norm_out = norm_->forward(core_attn_out_reshaped, z_reshaped);
  auto z_shape_og = z.sizes().vec();
  norm_out = norm_out.view(z_shape_og);
  norm_out = norm_out.view({-1, norm_out.size(2), norm_out.size(3)});

  // Project the normalized attention output back to hidden size.
  auto rearranged_norm =
      norm_out.reshape({norm_out.size(0), norm_out.size(1) * norm_out.size(2)});
  rearranged_norm = reshape_qkvz_unpad(attn_metadata, rearranged_norm);
  // For chunked prefill or spec verify, reshape_qkvz_with_pad may pad each
  // batch to max_len, causing output tokens > original_num_tokens. We need to
  // slice back to original_num_tokens to match residual shape for add_rms_norm.
  if (rearranged_norm.size(0) > original_num_tokens) {
    // Slice excess padding tokens
    rearranged_norm =
        rearranged_norm.slice(0, 0, original_num_tokens).contiguous();
  }
  return o_proj_->forward(rearranged_norm);
}

torch::Tensor Qwen3GatedDeltaNetBaseImpl::reshape_qkvz_unpad(
    const AttentionMetadata& attn_metadata,
    const torch::Tensor& padded_qkvz) const {
  const bool has_padded_queries =
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;
  if (!has_padded_queries) {
    return padded_qkvz;
  }
  std::vector<torch::Tensor> valid_batches;
  const bool has_host_lens = !attn_metadata.q_seq_lens_vec.empty();
  int64_t bs = has_host_lens
                   ? static_cast<int64_t>(attn_metadata.q_seq_lens_vec.size())
                   : attn_metadata.q_seq_lens.size(0);
  valid_batches.reserve(bs);
  int64_t max_len = attn_metadata.max_query_len;
  const auto& ori_seq_lens = attn_metadata.q_seq_lens;
  auto reshaped_qkvz = padded_qkvz.view({bs, max_len, -1});
  // Fallback when host lengths are absent: stage the device q_seq_lens to the
  // host once. Calling .item() per batch on a device tensor issues a separate
  // synchronous device->host copy each iteration, stalling schedule/execute
  // overlap; a single bulk copy avoids the per-iteration syncs.
  const torch::Tensor host_seq_lens =
      has_host_lens ? torch::Tensor() : ori_seq_lens.to(torch::kCPU);
  for (int64_t b = 0; b < bs; ++b) {
    int64_t ori_len = has_host_lens ? attn_metadata.q_seq_lens_vec[b]
                                    : host_seq_lens[b].item<int64_t>();
    torch::Tensor valid_batch =
        reshaped_qkvz[b].slice(/*dim=*/0, /*start=*/0, ori_len);
    valid_batches.emplace_back(valid_batch);
  }
  if (valid_batches.size() == 1) {
    return valid_batches[0].contiguous();
  }
  return torch::cat(valid_batches, 0).contiguous();
}

torch::Tensor Qwen3GatedDeltaNetBaseImpl::get_linear_state_indices(
    const ModelInputParams& input_params,
    const torch::Device& device) const {
  CHECK(!input_params.embedding.linear_state_ids.empty())
      << "linear_state_ids must be populated for gated delta net";
  if (input_params.embedding.linear_state_indices.defined()) {
    auto indices = input_params.embedding.linear_state_indices;
    if (indices.device() != device || indices.scalar_type() != torch::kInt) {
      indices =
          indices.to(torch::TensorOptions().dtype(torch::kInt).device(device),
                     /*non_blocking=*/true,
                     /*copy=*/true);
    }
    return indices.contiguous();
  }
  return torch::tensor(
      input_params.embedding.linear_state_ids,
      torch::TensorOptions().dtype(torch::kInt).device(device));
}

torch::Tensor Qwen3GatedDeltaNetBaseImpl::reshape_qkvz_with_pad(
    const AttentionMetadata& attn_metadata,
    const torch::Tensor& qkvz) const {
  const bool has_host_lens = !attn_metadata.q_seq_lens_vec.empty();
  int64_t bs = has_host_lens
                   ? static_cast<int64_t>(attn_metadata.q_seq_lens_vec.size())
                   : attn_metadata.q_seq_lens.size(0);
  int64_t max_len = attn_metadata.max_query_len;
  const auto& start_loc = attn_metadata.q_seq_lens;
  const bool need_padding =
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;
  if (!need_padding) {
    return qkvz.view({bs, -1, qkvz.size(-1)});
  }
  std::vector<torch::Tensor> batches;
  batches.reserve(bs);
  int64_t idx = 0;
  // See reshape_qkvz_unpad: stage device lengths to host once when host
  // lengths are absent to avoid a per-batch .item() device->host sync.
  const torch::Tensor host_seq_lens =
      has_host_lens ? torch::Tensor() : start_loc.to(torch::kCPU);
  for (int64_t b = 0; b < bs; ++b) {
    int64_t cur_len = has_host_lens ? attn_metadata.q_seq_lens_vec[b]
                                    : host_seq_lens[b].item<int64_t>();
    torch::Tensor batch =
        qkvz.slice(/*dim=*/0, idx, idx + cur_len).contiguous();
    idx = idx + cur_len;
    if (batch.size(0) != max_len) {
      batch = batch.size(0) > max_len
                  ? batch.slice(/*dim=*/0, /*start=*/0, max_len).contiguous()
                  : torch::nn::functional::pad(
                        batch,
                        torch::nn::functional::PadFuncOptions(
                            {0, 0, 0, max_len - batch.size(0)}))
                        .contiguous();
    }
    batches.emplace_back(batch);
  }
  auto ret = torch::stack(batches, 0).contiguous();
  return ret;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
Qwen3GatedDeltaNetBaseImpl::process_mixed_qkv(torch::Tensor& mixed_qkv) const {
  mixed_qkv = mixed_qkv.transpose(1, 2);
  int64_t batch_size = mixed_qkv.size(0);
  int64_t seq_len = mixed_qkv.size(1);
  std::vector<int64_t> split_sizes = {
      k_size_ / tp_size_, k_size_ / tp_size_, v_size_ / tp_size_};
  auto processed_qkv = torch::split(mixed_qkv, split_sizes, 2);
  auto processed_q = processed_qkv[0];
  auto processed_k = processed_qkv[1];
  auto processed_v = processed_qkv[2];
  processed_q = processed_q.view(
      {batch_size, seq_len, num_k_heads_ / tp_size_, head_k_dim_});
  processed_k = processed_k.view(
      {batch_size, seq_len, num_k_heads_ / tp_size_, head_k_dim_});
  processed_v = processed_v.view(
      {batch_size, seq_len, num_v_heads_ / tp_size_, head_v_dim_});
  return std::make_tuple(processed_q, processed_k, processed_v);
}

}  // namespace layer
}  // namespace xllm

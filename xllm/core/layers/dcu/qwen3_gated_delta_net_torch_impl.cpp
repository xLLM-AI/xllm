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

// Pure-torch fallback for Qwen3 gated-delta-net kernels. Used when the
// optimized HIP kernels are unavailable, or via env-flag A/B testing.

#include <glog/logging.h>

#include <cmath>
#include <tuple>

#include "qwen3_gated_delta_net_helpers.h"
#include "qwen3_gated_delta_net_impls.h"

namespace xllm {
namespace layer {
namespace qwen3_gdn {
namespace torch_impl {

torch::Tensor causal_conv1d_prefill(const torch::Tensor& flat_input,
                                    const torch::Tensor& conv_weight_2d,
                                    torch::Tensor& conv_cache,
                                    const std::vector<int64_t>& cu_seqlens,
                                    const std::vector<int64_t>& state_indices,
                                    int32_t kernel_size,
                                    bool activation) {
  // flat_input: [total_tokens, channels]
  // conv_weight_2d: [kernel_size, channels]
  const int64_t channels = flat_input.size(1);
  const int64_t batch_size = cu_seqlens.size() - 1;
  auto options = flat_input.options();
  // Transpose to [channels, kernel_size] for per-channel dot product.
  auto weight = conv_weight_2d.transpose(0, 1).contiguous();

  std::vector<torch::Tensor> outputs;
  outputs.reserve(batch_size);

  for (int64_t b = 0; b < batch_size; ++b) {
    const int64_t start = cu_seqlens[b];
    const int64_t end = cu_seqlens[b + 1];
    const int64_t seq_len = end - start;
    if (seq_len == 0) continue;

    auto seq_input = flat_input.slice(0, start, end);

    const int64_t state_idx = state_indices[b];
    torch::Tensor state;
    if (conv_cache.defined() && conv_cache.numel() > 0) {
      state = conv_cache[state_idx].clone();
    } else {
      state = torch::zeros({kernel_size - 1, channels}, options);
    }

    // Pad: [state(k-1,c); input(seq,c)] -> [seq + k - 1, channels]
    auto padded = torch::cat({state, seq_input}, 0);

    auto out = torch::zeros({seq_len, channels}, options);
    for (int64_t t = 0; t < seq_len; ++t) {
      auto window = padded.slice(0, t, t + kernel_size).to(torch::kFloat32);
      // out_t[c] = sum_i weight[c, i] * window[i, c]
      out[t] = torch::sum(weight * window.transpose(0, 1), {1});
    }

    if (conv_cache.defined() && conv_cache.numel() > 0) {
      conv_cache[state_idx] =
          padded.slice(0, seq_len, seq_len + kernel_size - 1).clone();
    }

    if (activation) {
      out = torch::silu(out);
    }
    outputs.push_back(out);
  }
  return torch::cat(outputs, 0).contiguous();
}

torch::Tensor causal_conv1d_decode(const torch::Tensor& flat_input,
                                   const torch::Tensor& conv_weight_2d,
                                   torch::Tensor& conv_cache,
                                   const torch::Tensor& state_indices,
                                   int32_t kernel_size,
                                   bool activation) {
  const int64_t batch = flat_input.size(0);
  const int64_t channels = flat_input.size(1);
  auto options = flat_input.options();
  auto weight = conv_weight_2d.transpose(0, 1).contiguous();
  auto weight_f32 = weight.to(torch::kFloat32);
  auto outputs = torch::empty({batch, channels}, options);

  for (int64_t b = 0; b < batch; ++b) {
    const int64_t idx = state_indices[b].item<int64_t>();
    auto state_t = conv_cache[idx].to(torch::kFloat32);
    auto x_t = flat_input[b].to(torch::kFloat32);
    auto frame = torch::cat({state_t, x_t.unsqueeze(0)}, 0);
    auto out_t = torch::sum(weight_f32 * frame.transpose(0, 1), {1});
    outputs[b] = out_t.to(options.dtype());
    conv_cache[idx] = frame.slice(0, 1, kernel_size).to(options.dtype());
  }

  if (activation) {
    outputs = torch::silu(outputs);
  }
  return outputs;
}

std::tuple<torch::Tensor, torch::Tensor> gated_delta_rule(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor g,
    torch::Tensor beta,
    std::optional<torch::Tensor> initial_state,
    bool output_final_state,
    bool use_qk_l2norm_in_kernel) {
  (void)output_final_state;
  const auto initial_dtype = query.dtype();

  if (use_qk_l2norm_in_kernel) {
    query = l2norm(query, -1, 1e-6);
    key = l2norm(key, -1, 1e-6);
  }

  auto to_f32_trans = [](torch::Tensor x) {
    return x.transpose(1, 2).contiguous().to(torch::kFloat32);
  };
  query = to_f32_trans(query);
  key = to_f32_trans(key);
  value = to_f32_trans(value);
  beta = to_f32_trans(beta);
  g = to_f32_trans(g);
  const int64_t value_num_heads = value.size(1);
  query = repeat_tensor_heads(query, value_num_heads, 1);
  key = repeat_tensor_heads(key, value_num_heads, 1);

  const int64_t batch_size = key.size(0);
  const int64_t num_heads = key.size(1);
  const int64_t sequence_length = key.size(2);
  const int64_t k_head_dim = key.size(3);
  const int64_t v_head_dim = value.size(3);

  const float scale_val = 1.0f / std::sqrt(static_cast<float>(query.size(-1)));
  query = query * scale_val;
  torch::Tensor core_attn_out = torch::zeros(
      {batch_size, num_heads, sequence_length, v_head_dim},
      torch::TensorOptions().dtype(torch::kFloat32).device(value.device()));
  torch::Tensor last_state;
  if (!initial_state.has_value()) {
    last_state = torch::zeros(
        {batch_size, num_heads, k_head_dim, v_head_dim},
        torch::TensorOptions().dtype(torch::kFloat32).device(value.device()));
  } else {
    last_state = initial_state.value().to(value.device(), torch::kFloat32);
  }

  for (int64_t i = 0; i < sequence_length; ++i) {
    auto q_t = query.select(2, i);
    auto k_t = key.select(2, i);
    auto v_t = value.select(2, i);
    auto g_t = g.select(2, i).exp().unsqueeze(-1).unsqueeze(-1);
    auto beta_t = beta.select(2, i).unsqueeze(-1);
    last_state = last_state * g_t;
    auto kv_mem = torch::sum(last_state * k_t.unsqueeze(-1), -2);
    auto delta = (v_t - kv_mem) * beta_t;
    last_state = last_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2);
    core_attn_out.select(2, i) = torch::sum(last_state * q_t.unsqueeze(-1), -2);
  }

  core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype);
  return std::make_tuple(core_attn_out, last_state);
}

}  // namespace torch_impl
}  // namespace qwen3_gdn
}  // namespace layer
}  // namespace xllm

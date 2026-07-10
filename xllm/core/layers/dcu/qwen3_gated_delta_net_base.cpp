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

#include "qwen3_gated_delta_net_base.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <tuple>

#include "qwen3_5_gated_delta_net.h"

namespace xllm {
namespace layer {

namespace {

// =========================================================================
// Pure-torch fallback implementations (no NPU/DCU kernels required)
// =========================================================================

torch::Tensor l2norm(const torch::Tensor& x, int64_t dim, double eps = 1e-6) {
  auto norm = torch::sqrt(torch::sum(torch::square(x), dim, true) + eps);
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
      << "target heads must be divisible by current heads";

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

// -----------------------------------------------------------------------
// Torch causal conv1d (prefill)
// conv_weight_2d: [out_channels, kernel_size] from ColumnParallelLinear
// conv_cache: [num_sequences, kernel_size - 1, channels]
// -----------------------------------------------------------------------
torch::Tensor torch_causal_conv1d(const torch::Tensor& flat_input,
                                  const torch::Tensor& conv_weight_2d,
                                  torch::Tensor& conv_cache,
                                  const std::vector<int64_t>& cu_seqlens,
                                  const std::vector<int64_t>& state_indices,
                                  int32_t kernel_size,
                                  bool activation) {
  // flat_input: [total_tokens, channels]
  // conv_weight_2d: [kernel_size, channels] (from conv1d_->weight())
  int64_t channels = flat_input.size(1);
  int64_t batch_size = cu_seqlens.size() - 1;
  auto options = flat_input.options();
  // Transpose to [channels, kernel_size] for per-channel dot product
  auto weight = conv_weight_2d.transpose(0, 1).contiguous();
  // weight: [channels, kernel_size]

  std::vector<torch::Tensor> outputs;
  outputs.reserve(batch_size);

  for (int64_t b = 0; b < batch_size; ++b) {
    int64_t start = cu_seqlens[b];
    int64_t end = cu_seqlens[b + 1];
    int64_t seq_len = end - start;
    if (seq_len == 0) continue;

    // seq_input: [seq_len, channels]
    auto seq_input = flat_input.slice(0, start, end);

    // Load / init conv state: [kernel_size - 1, channels]
    int64_t state_idx = state_indices[b];
    torch::Tensor state;
    if (conv_cache.defined() && conv_cache.numel() > 0) {
      state = conv_cache[state_idx].clone();
    } else {
      state = torch::zeros({kernel_size - 1, channels}, options);
    }

    // Pad: [state(k-1,c); input(seq,c)] -> [seq + k - 1, channels]
    auto padded = torch::cat({state, seq_input}, 0);

    // For each time step, depthwise conv: weight[c, ks] dot window[ks, c]
    auto out = torch::zeros({seq_len, channels}, options);
    for (int64_t t = 0; t < seq_len; ++t) {
      auto window = padded.slice(0, t, t + kernel_size).to(torch::kFloat32);
      // window: [kernel_size, channels]
      // weight: [channels, kernel_size]
      // out_t[c] = sum_i weight[c, i] * window[i, c]
      auto out_t = torch::sum(weight * window.transpose(0, 1), {1});
      // out_t: [channels]
      out[t] = out_t;
    }

    // Store last state: last (kernel_size-1) tokens
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

// -----------------------------------------------------------------------
// Torch causal conv1d update (decode): single-step
// conv_weight_2d: [out_channels, kernel_size]
// conv_cache: [num_sequences, kernel_size - 1, channels]
// -----------------------------------------------------------------------
torch::Tensor torch_causal_conv1d_update(const torch::Tensor& flat_input,
                                         const torch::Tensor& conv_weight_2d,
                                         torch::Tensor& conv_cache,
                                         const torch::Tensor& state_indices,
                                         int32_t kernel_size,
                                         bool activation) {
  // flat_input: [batch, channels]
  // conv_weight_2d: [kernel_size, channels] (from conv1d_->weight())
  int64_t batch = flat_input.size(0);
  int64_t channels = flat_input.size(1);
  auto options = flat_input.options();
  auto weight = conv_weight_2d.transpose(0, 1).contiguous();
  // weight: [channels, kernel_size]
  auto weight_f32 = weight.to(torch::kFloat32);
  auto outputs = torch::empty({batch, channels}, options);

  for (int64_t b = 0; b < batch; ++b) {
    int64_t idx = state_indices[b].item<int64_t>();
    // cache format: [kernel_size - 1, channels]
    auto state_t = conv_cache[idx].to(torch::kFloat32);
    auto x_t = flat_input[b].to(torch::kFloat32);
    // frame: [kernel_size, channels]
    auto frame = torch::cat({state_t, x_t.unsqueeze(0)}, 0);
    // depthwise conv: weight[c, ks] dot frame[ks, c] -> [c]
    auto out_t = torch::sum(weight_f32 * frame.transpose(0, 1), {1});
    outputs[b] = out_t.to(options.dtype());
    // Update state: shift, keep last (kernel_size-1) tokens
    conv_cache[idx] = frame.slice(0, 1, kernel_size).to(options.dtype());
  }

  if (activation) {
    outputs = torch::silu(outputs);
  }
  return outputs;
}

// -----------------------------------------------------------------------
// Torch fused_qkvzba_split_reshape_cat replacement
// -----------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
torch_fused_qkvzba_split(const torch::Tensor& qkvz_flat,
                         const torch::Tensor& ba_flat,
                         int64_t num_k_heads_local,
                         int64_t num_v_heads_local,
                         int64_t head_k_dim,
                         int64_t head_v_dim) {
  int64_t total_tokens = qkvz_flat.size(0);
  int64_t k_size = num_k_heads_local * head_k_dim;
  int64_t v_size = num_v_heads_local * head_v_dim;

  // qkvz_flat: [total_tokens, 2*k_size + 2*v_size]
  // = [total_tokens, q(k_size) + k(k_size) + v(v_size) + z(v_size)]
  auto qkvz_split =
      torch::split(qkvz_flat, {k_size, k_size, v_size, v_size}, 1);
  auto q = qkvz_split[0];
  auto k = qkvz_split[1];
  auto v = qkvz_split[2];
  auto z = qkvz_split[3];

  // ba_flat: [total_tokens, 2 * num_v_heads_local]
  // = [total_tokens, b(num_v_heads) + a(num_v_heads)]
  int64_t num_v = num_v_heads_local;
  auto ba_split = torch::split(ba_flat, {num_v, num_v}, 1);
  auto b = ba_split[0];
  auto a = ba_split[1];

  // mixed_qkv = cat([q, k, v], -1) → [total_tokens, 2*k_size + v_size]
  auto mixed_qkv = torch::cat({q, k, v}, 1).contiguous();

  return std::make_tuple(mixed_qkv, z, b, a);
}

// =========================================================================
// Gated Delta Rule torch implementations (ported from NPU)
// =========================================================================

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
    query = l2norm(query, -1, 1e-6);
    key = l2norm(key, -1, 1e-6);
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
  query = repeat_tensor_heads(query, value_num_heads, 1);
  key = repeat_tensor_heads(key, value_num_heads, 1);

  int64_t batch_size = key.size(0);
  int64_t num_heads = key.size(1);
  int64_t sequence_length = key.size(2);
  int64_t k_head_dim = key.size(3);
  int64_t v_head_dim = value.size(3);

  float scale_val = 1.0 / std::sqrt(static_cast<float>(query.size(-1)));
  torch::Tensor scale = torch::tensor(scale_val, query.options());
  query = query * scale;
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
    torch::Tensor q_t = query.select(2, i);
    torch::Tensor k_t = key.select(2, i);
    torch::Tensor v_t = value.select(2, i);
    torch::Tensor g_t = g.select(2, i).exp().unsqueeze(-1).unsqueeze(-1);
    torch::Tensor beta_t = beta.select(2, i).unsqueeze(-1);
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

int64_t get_checkpoint_stride(const torch::Tensor& conv_cache,
                              const torch::Tensor& ssm_cache) {
  if (!conv_cache.defined() || !ssm_cache.defined() ||
      conv_cache.numel() == 0 || ssm_cache.numel() == 0) {
    return 1;
  }
  CHECK_GT(conv_cache.size(0), 0) << "conv cache must have positive batch dim";
  CHECK_EQ(ssm_cache.size(0) % conv_cache.size(0), 0)
      << "ssm cache checkpoint layout mismatch";
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

}  // namespace

// =========================================================================
// Qwen3GatedDeltaNetBaseImpl
// =========================================================================

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
    conv1d_->weight().set_(conv1d_->weight().transpose(0, 1).contiguous());
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
      << "Missing required weight: " << prefix << "dt_bias";
  CHECK(A_log_is_loaded_) << "Missing required weight: " << prefix << "A_log";
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
  const int64_t original_num_tokens = hidden_states.size(0);
  auto [qkvz_padded, ba_padded] =
      project_padded_inputs(hidden_states, attn_metadata);
  int64_t batch_size = qkvz_padded.size(0);
  int64_t seq_len = qkvz_padded.size(1);

  // Flatten and split QKVZ/BA via torch fallback
  auto qkvz_flat =
      qkvz_padded.reshape({batch_size * seq_len, qkvz_padded.size(-1)});
  auto ba_flat = ba_padded.reshape({batch_size * seq_len, ba_padded.size(-1)});

  torch::Tensor mixed_qkv, z, b, a;
  std::tie(mixed_qkv, z, b, a) =
      torch_fused_qkvzba_split(qkvz_flat,
                               ba_flat,
                               num_k_heads_ / tp_size_,
                               num_v_heads_ / tp_size_,
                               head_k_dim_,
                               head_v_dim_);

  mixed_qkv = mixed_qkv.reshape({batch_size, seq_len, mixed_qkv.size(-1)});
  z = z.reshape({batch_size, seq_len, num_v_heads_ / tp_size_, head_v_dim_});
  b = b.reshape({batch_size, seq_len, num_v_heads_ / tp_size_});
  a = a.reshape({batch_size, seq_len, num_v_heads_ / tp_size_});

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
  const bool is_any_prefill =
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;

  // ---- Causal Conv1d ----
  if (is_any_prefill) {
    std::vector<int64_t> cu_seqlens_vec;
    if (!attn_metadata.q_seq_lens_vec.empty()) {
      cu_seqlens_vec.push_back(0);
      for (size_t i = 0; i < attn_metadata.q_seq_lens_vec.size(); ++i) {
        cu_seqlens_vec.push_back(cu_seqlens_vec.back() +
                                 attn_metadata.q_seq_lens_vec[i]);
      }
    } else {
      auto cpu_lens = attn_metadata.q_seq_lens.cpu();
      auto* ptr = cpu_lens.data_ptr<int64_t>();
      cu_seqlens_vec.assign(ptr, ptr + cpu_lens.numel());
    }

    auto conv_input = reshape_qkvz_unpad(attn_metadata, mixed_qkv);

    std::vector<int64_t> state_indices_vec(
        input_params.embedding.linear_state_ids.begin(),
        input_params.embedding.linear_state_ids.end());

    mixed_qkv = torch_causal_conv1d(conv_input.contiguous(),
                                    conv_weight,
                                    conv_cache,
                                    cu_seqlens_vec,
                                    state_indices_vec,
                                    conv_kernel_size_,
                                    /*activation=*/true);

    mixed_qkv = reshape_qkvz_with_pad(attn_metadata, mixed_qkv);
    mixed_qkv = mixed_qkv.transpose(1, 2);
  } else {
    // Decode: single-step conv update
    auto flat_input =
        mixed_qkv.reshape({batch_size * seq_len, mixed_qkv.size(-1)});
    auto updated = torch_causal_conv1d_update(flat_input,
                                              conv_weight,
                                              conv_cache,
                                              logical_state_indices,
                                              conv_kernel_size_,
                                              /*activation=*/true);
    mixed_qkv = updated.reshape({batch_size, seq_len, -1}).transpose(1, 2);
  }

  // ---- Gating (torch path: no kernel required) ----
  torch::Tensor g;
  torch::Tensor beta;
  {
    beta = torch::sigmoid(b);
    auto A_log_exp = A_log_.exp();
    auto a_float = a.to(torch::kFloat32);
    auto a_plus_dt = a_float + dt_bias_;
    auto softplus_out = torch::nn::functional::softplus(
        a_plus_dt,
        torch::nn::functional::SoftplusFuncOptions().beta(1.0).threshold(20.0));
    g = -A_log_exp * softplus_out;
    g = g.to(a.dtype()).contiguous();
  }

  auto [processed_q, processed_k, processed_v] = process_mixed_qkv(mixed_qkv);

  // ---- Gated Delta Rule ----
  torch::Tensor core_attn_out;
  if (is_any_prefill) {
    CHECK_GE(attn_metadata.q_seq_lens_vec.size(),
             static_cast<size_t>(batch_size))
        << "q_seq_lens_vec must be populated for Qwen3.5 prefill.";

    // Pack valid tokens per batch, run chunked gated delta rule
    std::vector<torch::Tensor> packed_q, packed_k, packed_v, packed_g,
        packed_beta;
    packed_q.reserve(batch_size);
    packed_k.reserve(batch_size);
    packed_v.reserve(batch_size);
    packed_g.reserve(batch_size);
    packed_beta.reserve(batch_size);

    for (int64_t bidx = 0; bidx < batch_size; ++bidx) {
      int64_t vlen = attn_metadata.q_seq_lens_vec[bidx];
      packed_q.push_back(processed_q[bidx].narrow(0, 0, vlen));
      packed_k.push_back(processed_k[bidx].narrow(0, 0, vlen));
      packed_v.push_back(processed_v[bidx].narrow(0, 0, vlen));
      packed_g.push_back(g[bidx].narrow(0, 0, vlen));
      packed_beta.push_back(beta[bidx].narrow(0, 0, vlen));
    }

    auto cat_q = torch::cat(packed_q, 0).unsqueeze(0);
    auto cat_k = torch::cat(packed_k, 0).unsqueeze(0);
    auto cat_v = torch::cat(packed_v, 0).unsqueeze(0);
    auto cat_g = torch::cat(packed_g, 0).unsqueeze(0);
    auto cat_beta = torch::cat(packed_beta, 0).unsqueeze(0);

    // Get initial state from ssm_cache
    auto initial_state =
        torch::index_select(ssm_cache, 0, linear_state_base_indices);
    if (!use_fla_ssm_state_layout()) {
      initial_state = initial_state.transpose(-1, -2).contiguous();
    }

    torch::Tensor last_state;
    std::tie(core_attn_out, last_state) = torch_recurrent_gated_delta_rule(
        cat_q, cat_k, cat_v, cat_g, cat_beta, initial_state);

    // Scatter back to per-batch output
    core_attn_out = core_attn_out.squeeze(0);
    auto final_out = torch::zeros_like(processed_v);
    int64_t offset = 0;
    for (int64_t bidx = 0; bidx < batch_size; ++bidx) {
      int64_t vlen = attn_metadata.q_seq_lens_vec[bidx];
      final_out[bidx]
          .narrow(0, 0, vlen)
          .copy_(core_attn_out.narrow(0, offset, vlen));
      offset += vlen;
    }
    core_attn_out = final_out;

    // Store state
    auto state_to_store =
        use_fla_ssm_state_layout() ? last_state : last_state.transpose(-1, -2);
    ssm_cache.index_put_({linear_state_base_indices},
                         state_to_store.to(ssm_cache.dtype()));
  } else {
    // Decode: recurrent step
    torch::Tensor init_state;
    if (ssm_cache.defined() && ssm_cache.numel() > 0) {
      init_state = torch::index_select(ssm_cache, 0, linear_state_base_indices);
      if (!use_fla_ssm_state_layout()) {
        init_state = init_state.transpose(-1, -2);
      }
    }
    torch::Tensor last_state;
    std::tie(core_attn_out, last_state) = torch_recurrent_gated_delta_rule(
        processed_q, processed_k, processed_v, g, beta, init_state);

    auto state_to_store =
        use_fla_ssm_state_layout() ? last_state : last_state.transpose(-1, -2);
    ssm_cache.index_put_({linear_state_base_indices},
                         state_to_store.to(ssm_cache.dtype()));
  }

  // ---- Z-gate + Output Projection ----
  auto z_reshaped = z.reshape({-1, z.size(-1)});
  auto core_attn_out_reshaped =
      core_attn_out.reshape({-1, core_attn_out.size(-1)});
  auto norm_out = norm_->forward(core_attn_out_reshaped, z_reshaped);
  auto z_shape_og = z.sizes().vec();
  norm_out = norm_out.view(z_shape_og);
  norm_out = norm_out.reshape({-1, norm_out.size(2), norm_out.size(3)});

  auto rearranged_norm =
      norm_out.reshape({norm_out.size(0), norm_out.size(1) * norm_out.size(2)});
  rearranged_norm = reshape_qkvz_unpad(attn_metadata, rearranged_norm);

  if (rearranged_norm.size(0) > original_num_tokens) {
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
  auto reshaped_qkvz = padded_qkvz.reshape({bs, max_len, -1});
  for (int64_t b = 0; b < bs; ++b) {
    int64_t ori_len = has_host_lens ? attn_metadata.q_seq_lens_vec[b]
                                    : ori_seq_lens[b].template item<int64_t>();
    torch::Tensor valid_batch = reshaped_qkvz[b].slice(0, 0, ori_len);
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
  const bool need_padding =
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;
  if (!need_padding) {
    return qkvz.reshape({bs, -1, qkvz.size(-1)});
  }
  std::vector<torch::Tensor> batches;
  batches.reserve(bs);
  int64_t idx = 0;
  for (int64_t b = 0; b < bs; ++b) {
    int64_t cur_len = has_host_lens ? attn_metadata.q_seq_lens_vec[b] : 0;
    torch::Tensor batch = qkvz.slice(0, idx, idx + cur_len).contiguous();
    idx = idx + cur_len;
    if (batch.size(0) != max_len) {
      batch = batch.size(0) > max_len
                  ? batch.slice(0, 0, max_len).contiguous()
                  : torch::nn::functional::pad(
                        batch,
                        torch::nn::functional::PadFuncOptions(
                            {0, 0, 0, max_len - batch.size(0)}))
                        .contiguous();
    }
    batches.emplace_back(batch);
  }
  return torch::stack(batches, 0).contiguous();
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
  processed_q = processed_q.reshape(
      {batch_size, seq_len, num_k_heads_ / tp_size_, head_k_dim_});
  processed_k = processed_k.reshape(
      {batch_size, seq_len, num_k_heads_ / tp_size_, head_k_dim_});
  processed_v = processed_v.reshape(
      {batch_size, seq_len, num_v_heads_ / tp_size_, head_v_dim_});
  return std::make_tuple(processed_q, processed_k, processed_v);
}

// =========================================================================
// Qwen3_5GatedDeltaNet DCU forward (direct projections, no merge/split)
// =========================================================================

torch::Tensor Qwen3_5GatedDeltaNetImpl::forward(
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params) {
  // Direct projection (no merge/split), matching vLLM reference.
  auto mixed_qkv_flat = in_proj_qkv_->forward(hidden_states);
  auto z_flat = in_proj_z_->forward(hidden_states);
  auto b_flat = in_proj_b_->forward(hidden_states);
  auto a_flat = in_proj_a_->forward(hidden_states);

  const int64_t num_tokens = hidden_states.size(0);
  const bool is_prefill =
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;

  int64_t batch_size, seq_len;
  torch::Tensor mixed_qkv, z, b, a;

  if (is_prefill) {
    const bool has_host = !attn_metadata.q_seq_lens_vec.empty();
    batch_size = has_host
                     ? static_cast<int64_t>(attn_metadata.q_seq_lens_vec.size())
                     : attn_metadata.q_seq_lens.size(0);
    seq_len = attn_metadata.max_query_len;
    mixed_qkv = reshape_qkvz_with_pad(attn_metadata, mixed_qkv_flat);
    z = reshape_qkvz_with_pad(attn_metadata, z_flat);
    b = reshape_qkvz_with_pad(attn_metadata, b_flat);
    a = reshape_qkvz_with_pad(attn_metadata, a_flat);
  } else {
    batch_size = mixed_qkv_flat.size(0);
    seq_len = 1;
    mixed_qkv = mixed_qkv_flat.reshape({batch_size, seq_len, -1});
    z = z_flat.reshape({batch_size, seq_len, -1});
    b = b_flat.reshape({batch_size, seq_len, -1});
    a = a_flat.reshape({batch_size, seq_len, -1});
  }

  // z: reshape to [batch, seq, num_v_heads/tp, head_v_dim]
  z = z.reshape({batch_size, seq_len, num_v_heads_ / tp_size_, head_v_dim_});

  // ---- Causal Conv1d ----
  torch::Tensor conv_cache = kv_cache.get_conv_cache();
  torch::Tensor conv_weight = conv1d_->weight();

  if (is_prefill) {
    std::vector<int64_t> cu_seqlens_vec;
    if (!attn_metadata.q_seq_lens_vec.empty()) {
      cu_seqlens_vec.push_back(0);
      for (size_t i = 0; i < attn_metadata.q_seq_lens_vec.size(); ++i)
        cu_seqlens_vec.push_back(cu_seqlens_vec.back() +
                                 attn_metadata.q_seq_lens_vec[i]);
    } else {
      auto cpu_lens = attn_metadata.q_seq_lens.cpu();
      auto* ptr = cpu_lens.data_ptr<int64_t>();
      cu_seqlens_vec.assign(ptr, ptr + cpu_lens.numel());
    }
    auto conv_input = reshape_qkvz_unpad(attn_metadata, mixed_qkv);
    std::vector<int64_t> state_indices(
        input_params.embedding.linear_state_ids.begin(),
        input_params.embedding.linear_state_ids.end());
    mixed_qkv = torch_causal_conv1d(conv_input.contiguous(),
                                    conv_weight,
                                    conv_cache,
                                    cu_seqlens_vec,
                                    state_indices,
                                    conv_kernel_size_,
                                    /*activation=*/true);
    mixed_qkv = reshape_qkvz_with_pad(attn_metadata, mixed_qkv);
  } else {
    auto flat_input = mixed_qkv.reshape({batch_size * seq_len, -1});
    auto state_indices =
        get_linear_state_indices(input_params, mixed_qkv.device());
    auto updated = torch_causal_conv1d_update(flat_input,
                                              conv_weight,
                                              conv_cache,
                                              state_indices,
                                              conv_kernel_size_,
                                              /*activation=*/true);
    mixed_qkv = updated.reshape({batch_size, seq_len, -1});
  }

  mixed_qkv = mixed_qkv.transpose(1, 2);
  auto [q, k, v] = process_mixed_qkv(mixed_qkv);

  // ---- Gating ----
  auto beta = torch::sigmoid(b);
  auto a_f32 = a.to(torch::kFloat32);
  auto softplus_out = torch::nn::functional::softplus(
      a_f32 + dt_bias_,
      torch::nn::functional::SoftplusFuncOptions().beta(1.0).threshold(20.0));
  auto g = (-A_log_.exp() * softplus_out).to(b.dtype()).contiguous();

  // ---- Gated Delta Rule ----
  torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
  auto state_indices = get_linear_state_indices(input_params, q.device());
  auto linear_state_base = state_indices;

  torch::Tensor core_attn_out;
  if (is_prefill) {
    std::vector<torch::Tensor> pq, pk, pv, pg, pb;
    for (int64_t bi = 0; bi < batch_size; ++bi) {
      int64_t vlen = attn_metadata.q_seq_lens_vec[bi];
      pq.push_back(q[bi].narrow(0, 0, vlen));
      pk.push_back(k[bi].narrow(0, 0, vlen));
      pv.push_back(v[bi].narrow(0, 0, vlen));
      pg.push_back(g[bi].narrow(0, 0, vlen));
      pb.push_back(beta[bi].narrow(0, 0, vlen));
    }
    auto cat_q = torch::cat(pq, 0).unsqueeze(0);
    auto cat_k = torch::cat(pk, 0).unsqueeze(0);
    auto cat_v = torch::cat(pv, 0).unsqueeze(0);
    auto cat_g = torch::cat(pg, 0).unsqueeze(0);
    auto cat_beta = torch::cat(pb, 0).unsqueeze(0);

    auto init_state = torch::index_select(ssm_cache, 0, linear_state_base);

    torch::Tensor last_state;
    std::tie(core_attn_out, last_state) = torch_recurrent_gated_delta_rule(
        cat_q, cat_k, cat_v, cat_g, cat_beta, init_state);

    core_attn_out = core_attn_out.squeeze(0);
    auto out3d = torch::zeros_like(v);
    int64_t off = 0;
    for (int64_t bi = 0; bi < batch_size; ++bi) {
      int64_t vlen = attn_metadata.q_seq_lens_vec[bi];
      out3d[bi].narrow(0, 0, vlen).copy_(core_attn_out.narrow(0, off, vlen));
      off += vlen;
    }
    core_attn_out = out3d;
    ssm_cache.index_put_({linear_state_base}, last_state.to(ssm_cache.dtype()));
  } else {
    auto init_state = torch::index_select(ssm_cache, 0, linear_state_base);
    torch::Tensor last_state;
    std::tie(core_attn_out, last_state) =
        torch_recurrent_gated_delta_rule(q, k, v, g, beta, init_state);
    ssm_cache.index_put_({linear_state_base}, last_state.to(ssm_cache.dtype()));
  }

  // ---- Z-gate + Output Projection ----
  auto z_2d = z.reshape({-1, head_v_dim_});
  auto co_2d = core_attn_out.reshape({-1, head_v_dim_});
  auto norm_out = norm_->forward(co_2d, z_2d);
  norm_out = norm_out.reshape(
      {batch_size, seq_len, num_v_heads_ / tp_size_, head_v_dim_});
  norm_out = norm_out.reshape({norm_out.size(0),
                               norm_out.size(1),
                               norm_out.size(2) * norm_out.size(3)});
  auto unpad = reshape_qkvz_unpad(attn_metadata, norm_out);
  if (unpad.size(0) > num_tokens) {
    unpad = unpad.slice(0, 0, num_tokens).contiguous();
  }
  return o_proj_->forward(unpad);
}

}  // namespace layer
}  // namespace xllm

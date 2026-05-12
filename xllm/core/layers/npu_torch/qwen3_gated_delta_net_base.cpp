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

#include "common/flash_comm1_context.h"
#include "xllm/core/kernels/ops_api.h"

namespace xllm {
namespace layer {

namespace {
torch::Tensor l2norm(const torch::Tensor& x, int64_t dim, double eps = 1e-6) {
  auto norm = torch::sqrt(torch::sum(torch::square(x), dim, true) + eps);
  return x / norm;
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

std::tuple<torch::Tensor, torch::Tensor> torch_chunk_gated_delta_rule(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor g,
    torch::Tensor beta,
    int64_t chunk_size = 64,
    c10::optional<torch::Tensor> initial_state = c10::nullopt,
    bool output_final_state = true,
    bool use_qk_l2norm_in_kernel = true) {
  auto initial_dtype = query.dtype();
  if (use_qk_l2norm_in_kernel) {
    query = l2norm(query, -1, 1e-6);
    key = l2norm(key, -1, 1e-6);
  }
  auto to_float32 = [](torch::Tensor x) {
    return x.transpose(1, 2).contiguous().to(torch::kFloat32);
  };

  query = to_float32(query);
  key = to_float32(key);
  value = to_float32(value);
  beta = to_float32(beta);
  g = to_float32(g);

  auto batch_size = query.size(0);
  auto num_heads = query.size(1);
  auto sequence_length = query.size(2);
  auto k_head_dim = key.size(-1);
  auto v_head_dim = value.size(-1);

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
  auto attn = -(torch::matmul(k_beta, key.transpose(-1, -2)) * decay_mask)
                   .masked_fill(mask, 0.0);
  for (int64_t i = 1; i < chunk_size; ++i) {
    if (!attn.is_contiguous()) {
      attn = attn.contiguous();
    }
    auto row = attn.slice(-2, i, i + 1)
                   .slice(-1, 0, i)
                   .squeeze(-2)
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

torch::Tensor Qwen3GatedDeltaNetBaseImpl::forward(
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params) {
  return forward(hidden_states, attn_metadata, kv_cache, input_params, nullptr);
}

torch::Tensor Qwen3GatedDeltaNetBaseImpl::forward(
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
    const FlashComm1Context* fc1_ctx) {
  torch::Tensor h = hidden_states;

  if (fc1_ctx && fc1_ctx->is_sequence_sharded()) {
    h = gather_sequence(hidden_states, *fc1_ctx);
  }

  auto [qkvz_padded, ba_padded] =
      project_padded_inputs(h, attn_metadata);
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

  torch::Tensor mixed_qkv, z, b, a;
  std::tie(mixed_qkv, z, b, a) =
      xllm::kernel::fused_qkvzba_split_reshape_cat(fused_params);

  mixed_qkv = mixed_qkv.view({batch_size, seq_len, mixed_qkv.size(-1)});
  z = z.view({batch_size, seq_len, num_v_heads_ / tp_size_, head_v_dim_});
  b = b.view({batch_size, seq_len, num_v_heads_ / tp_size_});
  a = a.view({batch_size, seq_len, num_v_heads_ / tp_size_});

  torch::Tensor conv_cache = kv_cache.get_conv_cache();
  torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
  torch::Tensor g, beta, core_attn_out, last_recurrent_state;
  auto device = mixed_qkv.device();
  auto conv_weight = conv1d_->weight();
  auto linear_state_indices = get_linear_state_indices(input_params, device);

  if (attn_metadata.is_prefill) {
    mixed_qkv = mixed_qkv.transpose(1, 2);
    torch::Tensor conv_state =
        (seq_len < conv_kernel_size_ - 1)
            ? torch::pad(mixed_qkv, {0, conv_kernel_size_ - 1 - seq_len})
        : (seq_len > conv_kernel_size_ - 1)
            ? mixed_qkv.narrow(
                  -1, seq_len - conv_kernel_size_ + 1, conv_kernel_size_ - 1)
            : mixed_qkv;
    conv_state = conv_state.transpose(1, 2).contiguous();
    conv_cache.index_put_({linear_state_indices},
                          conv_state.to(conv_cache.dtype()));
    torch::Tensor bias;
    auto conv_output =
        torch::conv1d(mixed_qkv,
                      conv_weight.unsqueeze(1).to(device),
                      bias,
                      /*stride=*/std::vector<int64_t>{1},
                      /*padding=*/std::vector<int64_t>{3},
                      /*dilation=*/std::vector<int64_t>{1},
                      /*groups=*/static_cast<int64_t>(mixed_qkv.size(1)));
    mixed_qkv = torch::silu(conv_output.slice(2, 0, seq_len));

  } else {
    xllm::kernel::CausalConv1dUpdateParams conv1d_params;
    conv1d_params.x = mixed_qkv.reshape({-1, mixed_qkv.size(-1)});
    conv1d_params.conv_state = conv_cache;
    conv1d_params.weight = conv_weight;
    conv1d_params.conv_state_indices = linear_state_indices;
    conv1d_params.block_idx_last_scheduled_token =
        std::optional<torch::Tensor>();
    conv1d_params.initial_state_idx = std::optional<torch::Tensor>();
    conv1d_params.query_start_loc = attn_metadata.q_cu_seq_lens;
    conv1d_params.max_query_len = attn_metadata.max_query_len;
    mixed_qkv = xllm::kernel::causal_conv1d_update(conv1d_params);
    mixed_qkv =
        mixed_qkv.view({batch_size, -1, mixed_qkv.size(-1)}).contiguous();
    mixed_qkv = mixed_qkv.transpose(1, 2);
  }

  if (attn_metadata.is_prefill) {
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
  } else {
    xllm::kernel::FusedGdnGatingParams gdn_params;
    gdn_params.A_log = A_log_;
    gdn_params.a = a.view({-1, a.size(-1)});
    gdn_params.b = b.view({-1, b.size(-1)});
    gdn_params.dt_bias = dt_bias_;
    gdn_params.beta = 1.0f;
    gdn_params.threshold = 20.0f;
    std::tie(g, beta) = xllm::kernel::fused_gdn_gating(gdn_params);
  }
  auto [processed_q, processed_k, processed_v] = process_mixed_qkv(mixed_qkv);
  if (attn_metadata.is_prefill) {
    xllm::kernel::ChunkGatedDeltaRuleParams chunk_gated_delta_params;
    chunk_gated_delta_params.q = processed_q;
    chunk_gated_delta_params.k = processed_k;
    chunk_gated_delta_params.v = processed_v;
    chunk_gated_delta_params.g = g;
    chunk_gated_delta_params.beta = beta;
    torch::Tensor initial_state_tensor =
        torch::index_select(ssm_cache, 0, linear_state_indices);
    initial_state_tensor.fill_(0.0);
    chunk_gated_delta_params.initial_state = initial_state_tensor;
    chunk_gated_delta_params.output_final_state = true;
    chunk_gated_delta_params.cu_seqlens = attn_metadata.q_cu_seq_lens;
    chunk_gated_delta_params.head_first = false;
    chunk_gated_delta_params.use_qk_l2norm_in_kernel = true;
    std::tie(core_attn_out, last_recurrent_state) =
        xllm::kernel::chunk_gated_delta_rule(chunk_gated_delta_params);
    ssm_cache.index_put_(
        {linear_state_indices},
        last_recurrent_state.transpose(-1, -2).to(ssm_cache.dtype()));
  } else {
    processed_q = xllm::kernel::l2_norm(processed_q, 1e-6);
    processed_k = xllm::kernel::l2_norm(processed_k, 1e-6);
    auto zero = torch::zeros({1}, attn_metadata.q_seq_lens.options());
    torch::Tensor actual_seq_lengths =
        torch::cat({zero, attn_metadata.q_seq_lens}, 0);
    double scale = 1.0 / std::sqrt(static_cast<float>(processed_q.size(-1)));
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
                        actual_seq_lengths,
                        linear_state_indices,
                        c10::nullopt,
                        g.squeeze(0).contiguous(),
                        c10::nullopt)
                        .unsqueeze(0)
                        .contiguous();
  }

  auto z_reshaped = z.view({-1, z.size(-1)});
  auto core_attn_out_reshaped =
      core_attn_out.view({-1, core_attn_out.size(-1)});
  auto norm_out = norm_->forward(core_attn_out_reshaped, z_reshaped);
  auto z_shape_og = z.sizes().vec();
  norm_out = norm_out.view(z_shape_og);
  norm_out = norm_out.view({-1, norm_out.size(2), norm_out.size(3)});

  auto rearranged_norm =
      norm_out.reshape({norm_out.size(0), norm_out.size(1) * norm_out.size(2)});
  rearranged_norm = reshape_qkvz_unpad(attn_metadata, rearranged_norm);

  if (fc1_ctx && fc1_ctx->is_sequence_sharded()) {
    return o_proj_->forward(rearranged_norm, RowParallelReduceMode::REDUCE_SCATTER, fc1_ctx);
  }
  return o_proj_->forward(rearranged_norm);
}

torch::Tensor Qwen3GatedDeltaNetBaseImpl::reshape_qkvz_unpad(
    const AttentionMetadata& attn_metadata,
    const torch::Tensor& padded_qkvz) const {
  if (!attn_metadata.is_prefill) {
    return padded_qkvz;
  }
  std::vector<torch::Tensor> valid_batches;
  int64_t bs = attn_metadata.q_seq_lens.size(0);
  int64_t max_len = attn_metadata.max_query_len;
  const auto& ori_seq_lens = attn_metadata.q_seq_lens;
  auto reshaped_qkvz = padded_qkvz.view({bs, max_len, -1});
  for (int64_t b = 0; b < bs; ++b) {
    int64_t ori_len = ori_seq_lens[b].template item<int64_t>();
    torch::Tensor valid_batch = reshaped_qkvz[b].slice(0, 0, ori_len);
    valid_batches.push_back(valid_batch);
  }
  return torch::cat(valid_batches, 0).contiguous();
}

torch::Tensor Qwen3GatedDeltaNetBaseImpl::get_linear_state_indices(
    const ModelInputParams& input_params,
    const torch::Device& device) const {
  CHECK(!input_params.linear_state_ids.empty())
      << "linear_state_ids must be populated for gated delta net";
  if (input_params.linear_state_indices.defined()) {
    return input_params.linear_state_indices;
  }
  return torch::tensor(
      input_params.linear_state_ids,
      torch::TensorOptions().dtype(torch::kInt).device(device));
}

torch::Tensor Qwen3GatedDeltaNetBaseImpl::reshape_qkvz_with_pad(
    const AttentionMetadata& attn_metadata,
    const torch::Tensor& qkvz) const {
  int64_t bs = attn_metadata.q_seq_lens.size(0);
  int64_t max_len = attn_metadata.max_query_len;
  const auto& start_loc = attn_metadata.q_seq_lens;
  if (!attn_metadata.is_prefill) {
    return qkvz.view({qkvz.size(0), -1, qkvz.size(-1)});
  }
  std::vector<torch::Tensor> batches;
  int64_t idx = 0;
  for (int64_t b = 0; b < bs; ++b) {
    int64_t cur_len = start_loc[b].template item<int64_t>();
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
    batches.push_back(batch);
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

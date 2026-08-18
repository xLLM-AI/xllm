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
#include <cmath>
#include <optional>
#include <tuple>
#include <vector>

#include "kernels/musa/gdn_ops.h"
#include "kernels/ops_api.h"

namespace xllm {
namespace layer {

namespace {
constexpr bool kEnableFusedGdnDecode = true;
constexpr bool kEnableMateGdnDecode = false;
constexpr bool kEnableMateGdnPrefill = false;
constexpr int64_t kDefaultMaxGraphBatchSize = 256;

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
  // conv_cache layout: [num_blocks, dim, state_len].
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

  conv_cache.index_copy_(/*dim=*/0, state_indices, next_states);
  auto output = output_f32.to(mixed_qkv.scalar_type());
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
    // The fused GDN kernels persist the recurrent state transposed relative to
    // the [HV, K, V] layout used by the reference recurrence.
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

      recurrent_state = recurrent_state * g_t.exp();
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
    }
  }

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
      "norm",
      musa::RmsNormGated(head_v_dim_,
                         kDefaultMaxGraphBatchSize * (num_v_heads_ / tp_size_),
                         args.rms_norm_eps(),
                         options));
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
  // Early-return on dummy shards. Under dp>1, an empty shard is padded with a
  // fake token by worker_impl but its GDN state tensors (linear_state_ids,
  // has_initial_states etc.) may be empty/undefined. Entering
  // get_linear_state_indices() would CHECK-fail or write fake index 0 into a
  // real cache slot.
  if (attn_metadata.is_dummy) {
    return torch::zeros_like(hidden_states);
  }
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

  xllm::kernel::musa::FusedQkvzbaSplitReshapeExtras fused_extras;
  const int64_t local_nk = num_k_heads_ / tp_size_;
  const int64_t local_nv = num_v_heads_ / tp_size_;
  const int64_t expected_m = batch_size * seq_len;
  constexpr int64_t kPersistentMaxRows = 256;
  if (expected_m <= kPersistentMaxRows) {
    const int64_t expected_qkv_dim =
        2 * local_nk * head_k_dim_ + local_nv * head_v_dim_;
    const int64_t expected_z_dim = local_nv * head_v_dim_;
    const auto grow_2d = [kPersistentMaxRows](
                             torch::Tensor& buf,
                             int64_t d,
                             const torch::TensorOptions& options) {
      const bool needs = !buf.defined() || buf.size(0) != kPersistentMaxRows ||
                         buf.size(1) != d ||
                         buf.scalar_type() != options.dtype().toScalarType() ||
                         buf.device() != options.device();
      if (needs) {
        buf = torch::empty({kPersistentMaxRows, d}, options);
      }
    };
    grow_2d(mixed_qkv_out_buf_, expected_qkv_dim, qkvz_flat.options());
    grow_2d(z_out_buf_, expected_z_dim, qkvz_flat.options());
    grow_2d(b_out_buf_, local_nv, ba_flat.options());
    grow_2d(a_out_buf_, local_nv, ba_flat.options());
    fused_extras.mixed_qkv_out_buf = mixed_qkv_out_buf_.narrow(
        /*dim=*/0, /*start=*/0, /*length=*/expected_m);
    fused_extras.z_out_buf = z_out_buf_.narrow(
        /*dim=*/0, /*start=*/0, /*length=*/expected_m);
    fused_extras.b_out_buf = b_out_buf_.narrow(
        /*dim=*/0, /*start=*/0, /*length=*/expected_m);
    fused_extras.a_out_buf = a_out_buf_.narrow(
        /*dim=*/0, /*start=*/0, /*length=*/expected_m);
  }

  torch::Tensor mixed_qkv, z, b, a;
  std::tie(mixed_qkv, z, b, a) =
      xllm::kernel::musa::fused_qkvzba_split_reshape_cat(fused_params,
                                                         fused_extras);

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
  // Exclude chunked-prefill: seq_len==1 chunked batches must not take the fused
  // decode path (that skips process_mixed_qkv while still hitting prefill
  // attn).
  const bool decode_eligible = !is_any_prefill && !use_spec_verify &&
                               seq_len == 1 && checkpoint_stride == 1;
  // Production defaults: fused decode on, mate decode/prefill off.
  const bool use_fused_gdn_decode = kEnableFusedGdnDecode && decode_eligible;
  const bool use_mate_gdn_decode =
      kEnableMateGdnDecode && decode_eligible && !use_fused_gdn_decode;
  // Both fused and mate decode paths consume the flat [tokens, dim] mixed_qkv
  // and split q/k/v via strided reads (no contiguous() copies).
  const bool use_flat_mixed_qkv_decode =
      use_fused_gdn_decode || use_mate_gdn_decode;

  if (!use_spec_verify && is_any_prefill) {
    torch::Tensor conv_input = reshape_qkvz_unpad(attn_metadata, mixed_qkv);
    // Canonical recurrent-state validity from linear_state_validity_mask.
    // Do not derive from kv_cache_tokens_nums: prefix-cached tokens do not
    // imply a valid GDN recurrent state (see AttentionMetadataBuilder tests).
    const torch::Tensor& has_initial_state = attn_metadata.has_initial_states;
    CHECK(has_initial_state.defined())
        << "has_initial_states must be populated for Qwen3.5 prefill";
    CHECK_EQ(has_initial_state.dim(), 1);
    CHECK_EQ(has_initial_state.numel(), batch_size);
    CHECK_EQ(has_initial_state.scalar_type(), torch::kBool);
    CHECK(has_initial_state.device() == mixed_qkv.device())
        << "has_initial_states must be on the same device as mixed_qkv";
    mixed_qkv =
        xllm::kernel::musa::causal_conv1d_prefill(conv_input,
                                                  conv_weight,
                                                  conv_cache,
                                                  std::nullopt,
                                                  attn_metadata.q_cu_seq_lens,
                                                  logical_state_indices,
                                                  has_initial_state,
                                                  /*silu_activation=*/true);

    mixed_qkv = reshape_qkvz_with_pad(attn_metadata, mixed_qkv);
    mixed_qkv = mixed_qkv.transpose(1, 2);
  } else if (use_spec_verify) {
    CHECK(input_params.num_accepted_tokens.defined())
        << "num_accepted_tokens must be populated for Qwen3.5 spec verify";
    torch::Tensor pre_conv_mixed_qkv = mixed_qkv.transpose(1, 2);
    mixed_qkv = run_spec_verify_conv(pre_conv_mixed_qkv,
                                     conv_cache,
                                     logical_state_indices,
                                     input_params.num_accepted_tokens_host,
                                     attn_metadata.q_cu_seq_lens,
                                     conv_weight,
                                     conv_kernel_size_);
  } else {
    // Decode uses the in-place causal convolution update on the cache layout.
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
    std::optional<torch::Tensor> conv1d_output_buf = std::nullopt;
    {
      const auto& x_in = conv1d_params.x;
      const int64_t m = x_in.size(0);
      const int64_t d = x_in.size(1);
      const int64_t capacity = std::max(m, kDefaultMaxGraphBatchSize);
      const bool needs =
          !conv1d_decode_out_buf_.defined() ||
          conv1d_decode_out_buf_.size(0) < m ||
          conv1d_decode_out_buf_.size(1) != d ||
          conv1d_decode_out_buf_.scalar_type() != x_in.scalar_type() ||
          conv1d_decode_out_buf_.device() != x_in.device();
      if (needs) {
        conv1d_decode_out_buf_ = torch::empty({capacity, d}, x_in.options());
      }
      conv1d_output_buf =
          conv1d_decode_out_buf_.narrow(/*dim=*/0, /*start=*/0, /*length=*/m);
    }
    mixed_qkv = xllm::kernel::musa::causal_conv1d_update(conv1d_params,
                                                         conv1d_output_buf);
    if (use_flat_mixed_qkv_decode) {
      // Fused decode kernels consume flat [tokens, dim] inputs.
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
  torch::Tensor g;
  torch::Tensor beta;
  // Compute gated delta net decay and beta terms.
  if (use_spec_verify || checkpoint_stride > 1) {
    beta = torch::sigmoid(b);
    torch::Tensor A_log_exp = A_log_.exp();
    torch::Tensor a_float = a.to(torch::kFloat32);
    torch::Tensor a_plus_dt = a_float + dt_bias_;
    torch::Tensor softplus_out = torch::nn::functional::softplus(
        a_plus_dt,
        torch::nn::functional::SoftplusFuncOptions().beta(1.0).threshold(20.0));
    g = -A_log_exp * softplus_out;
    g = g.to(a.dtype()).contiguous();
  } else if (is_any_prefill) {
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
  const bool use_mate_gdn_prefill =
      kEnableMateGdnPrefill && attn_metadata.is_prefill && !use_spec_verify;
  // Apply chunked or recurrent gated-delta attention and update caches.
  if (use_mate_gdn_prefill) {
    xllm::kernel::musa::MateGatedDeltaRulePrefillParams mate_params;
    mate_params.q = processed_q;
    mate_params.k = processed_k;
    mate_params.v = processed_v;
    mate_params.g = g;
    mate_params.beta = beta;
    mate_params.scale =
        1.0 / std::sqrt(static_cast<double>(processed_q.size(-1)));
    torch::Tensor mate_final_state;
    std::tie(core_attn_out, mate_final_state) =
        xllm::kernel::musa::mate_gated_delta_rule_prefill(mate_params);
    ssm_cache.index_put_({linear_state_base_indices},
                         mate_final_state.to(ssm_cache.dtype()));
  } else if (!use_spec_verify && attn_metadata.is_prefill &&
             !attn_metadata.is_chunked_prefill) {
    xllm::kernel::ChunkGatedDeltaRuleParams chunk_gated_delta_params;
    chunk_gated_delta_params.q = processed_q;
    chunk_gated_delta_params.k = processed_k;
    chunk_gated_delta_params.v = processed_v;
    CHECK_GE(attn_metadata.q_seq_lens_vec.size(),
             static_cast<size_t>(processed_q.size(0)))
        << "q_seq_lens_vec must be populated for GDN prefill.";
    for (int64_t batch_idx = 0; batch_idx < processed_q.size(0); ++batch_idx) {
      const int64_t valid_len = attn_metadata.q_seq_lens_vec[batch_idx];
      CHECK_GE(valid_len, 0);
      CHECK_LE(valid_len, processed_q.size(1));
      if (valid_len < processed_q.size(1)) {
        processed_q[batch_idx]
            .narrow(/*dim=*/0,
                    /*start=*/valid_len,
                    /*length=*/processed_q.size(1) - valid_len)
            .zero_();
        processed_k[batch_idx]
            .narrow(/*dim=*/0,
                    /*start=*/valid_len,
                    /*length=*/processed_k.size(1) - valid_len)
            .zero_();
        processed_v[batch_idx]
            .narrow(/*dim=*/0,
                    /*start=*/valid_len,
                    /*length=*/processed_v.size(1) - valid_len)
            .zero_();
        g[batch_idx]
            .narrow(/*dim=*/0,
                    /*start=*/valid_len,
                    /*length=*/processed_q.size(1) - valid_len)
            .zero_();
        beta[batch_idx]
            .narrow(/*dim=*/0,
                    /*start=*/valid_len,
                    /*length=*/processed_q.size(1) - valid_len)
            .zero_();
      }
    }
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
  } else if (is_any_prefill) {
    CHECK_GE(attn_metadata.q_seq_lens_vec.size(),
             static_cast<size_t>(batch_size))
        << "q_seq_lens_vec must be populated for Qwen3.5 prefill.";
    torch::Tensor initial_state_tensor =
        torch::index_select(ssm_cache, 0, linear_state_base_indices);
    CHECK_EQ(input_params.linear_state_validity_mask.size(),
             input_params.embedding.linear_state_ids.size())
        << "linear state validity mask must be sequence-scoped.";
    for (size_t i = 0; i < input_params.linear_state_validity_mask.size();
         ++i) {
      if (input_params.linear_state_validity_mask[i] == 0) {
        initial_state_tensor.select(0, static_cast<int64_t>(i)).fill_(0.0);
      }
    }
    // The MUSA decode cache is stored as [V, K]; the chunk kernel consumes
    // and returns [K, V] states.
    initial_state_tensor = initial_state_tensor.transpose(-1, -2).contiguous();

    core_attn_out = torch::zeros_like(processed_v);
    std::vector<torch::Tensor> final_states;
    final_states.reserve(static_cast<size_t>(batch_size));
    for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      const int64_t valid_len = attn_metadata.q_seq_lens_vec[batch_idx];
      CHECK_GE(valid_len, 0);
      CHECK_LE(valid_len, processed_q.size(1));
      if (valid_len == 0) {
        final_states.emplace_back(initial_state_tensor[batch_idx]);
        continue;
      }

      xllm::kernel::ChunkGatedDeltaRuleParams chunk_gated_delta_params;
      chunk_gated_delta_params.q =
          processed_q[batch_idx]
              .narrow(/*dim=*/0, /*start=*/0, valid_len)
              .unsqueeze(0);
      chunk_gated_delta_params.k =
          processed_k[batch_idx]
              .narrow(/*dim=*/0, /*start=*/0, valid_len)
              .unsqueeze(0);
      chunk_gated_delta_params.v =
          processed_v[batch_idx]
              .narrow(/*dim=*/0, /*start=*/0, valid_len)
              .unsqueeze(0);
      chunk_gated_delta_params.g =
          g[batch_idx].narrow(/*dim=*/0, /*start=*/0, valid_len).unsqueeze(0);
      chunk_gated_delta_params.beta =
          beta[batch_idx]
              .narrow(/*dim=*/0, /*start=*/0, valid_len)
              .unsqueeze(0);
      chunk_gated_delta_params.initial_state =
          initial_state_tensor[batch_idx].unsqueeze(0);
      chunk_gated_delta_params.output_final_state = true;
      chunk_gated_delta_params.head_first = false;
      chunk_gated_delta_params.use_qk_l2norm_in_kernel = true;

      torch::Tensor sequence_output;
      torch::Tensor final_state;
      std::tie(sequence_output, final_state) =
          xllm::kernel::chunk_gated_delta_rule(chunk_gated_delta_params);
      core_attn_out[batch_idx]
          .narrow(/*dim=*/0, /*start=*/0, valid_len)
          .copy_(sequence_output[0]);
      final_states.emplace_back(final_state[0]);
    }

    last_recurrent_state = torch::stack(final_states, /*dim=*/0);
    torch::Tensor state_to_store =
        last_recurrent_state.transpose(-1, -2).contiguous();
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
    // Fuse QKV split, gating, normalization, and the recurrent update.
    xllm::kernel::musa::MateGatedDeltaRuleDecodeParams fused_params;
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
    {
      const int64_t b = fused_params.mixed_qkv.size(0);
      const int64_t hv = num_v_heads_ / tp_size_;
      const int64_t v = head_v_dim_;
      const int64_t capacity = std::max(b, kDefaultMaxGraphBatchSize);
      const auto opts = mixed_qkv.options();
      const bool needs = !fused_gdn_decode_out_buf_.defined() ||
                         fused_gdn_decode_out_buf_.size(0) < b ||
                         fused_gdn_decode_out_buf_.size(1) != hv ||
                         fused_gdn_decode_out_buf_.size(2) != v ||
                         fused_gdn_decode_out_buf_.scalar_type() !=
                             opts.dtype().toScalarType() ||
                         fused_gdn_decode_out_buf_.device() != opts.device();
      if (needs) {
        fused_gdn_decode_out_buf_ = torch::empty({capacity, hv, v}, opts);
      }
      fused_params.decode_output = fused_gdn_decode_out_buf_.narrow(
          /*dim=*/0, /*start=*/0, /*length=*/b);
    }
    core_attn_out =
        xllm::kernel::musa::fused_gated_delta_rule_decode(fused_params)
            .unsqueeze(0);
  } else if (use_mate_gdn_decode) {
    xllm::kernel::musa::MateGatedDeltaRuleDecodeParams mate_params;
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
        xllm::kernel::musa::mate_gated_delta_rule_decode(mate_params)
            .unsqueeze(0);
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
                          std::nullopt,
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

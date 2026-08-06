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

#include "layers/mlu/qwen3_5/qwen3_5_gated_delta_net.h"

#include <glog/logging.h>

#include <cmath>

#include "framework/state_dict/utils.h"
#include "kernels/mlu/mlu_ops_api.h"
#include "kernels/ops_api.h"

namespace xllm {
namespace layer {

torch::Tensor build_linear_state_base_indices(
    const torch::Tensor& logical_state_indices,
    int64_t checkpoint_stride) {
  torch::Tensor state_indices =
      logical_state_indices.contiguous().to(torch::kInt32);
  if (checkpoint_stride == 1) {
    return state_indices;
  }
  return (state_indices * checkpoint_stride).contiguous();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> split_mixed_qkv(
    const torch::Tensor& mixed_qkv,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
  const int64_t num_tokens = mixed_qkv.size(0);
  const int64_t q_size = num_k_heads * head_k_dim;
  const int64_t k_size = num_k_heads * head_k_dim;
  const int64_t v_size = num_v_heads * head_v_dim;
  CHECK_EQ(mixed_qkv.size(1), q_size + k_size + v_size)
      << "Qwen3.5 MLU GDN mixed qkv channel mismatch";

  torch::Tensor q = mixed_qkv.slice(/*dim=*/1, /*start=*/0, /*end=*/q_size)
                        .contiguous()
                        .view({1, num_tokens, num_k_heads, head_k_dim});
  torch::Tensor k =
      mixed_qkv.slice(/*dim=*/1, /*start=*/q_size, /*end=*/q_size + k_size)
          .contiguous()
          .view({1, num_tokens, num_k_heads, head_k_dim});
  torch::Tensor v = mixed_qkv
                        .slice(/*dim=*/1,
                               /*start=*/q_size + k_size,
                               /*end=*/q_size + k_size + v_size)
                        .contiguous()
                        .view({1, num_tokens, num_v_heads, head_v_dim});
  return {q, k, v};
}

torch::Tensor build_rebased_ssm_state_indices(
    const torch::Tensor& logical_state_indices,
    int64_t checkpoint_stride,
    int64_t q_max_seq_len) {
  torch::Tensor base_indices =
      build_linear_state_base_indices(logical_state_indices, checkpoint_stride);
  torch::Tensor offsets = torch::arange(q_max_seq_len, base_indices.options());
  return (base_indices.unsqueeze(/*dim=*/1) + offsets).contiguous();
}

Qwen3_5GatedDeltaNetImpl::Qwen3_5GatedDeltaNetImpl(
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

  // The gated delta net (linear_attn) projections are kept in high precision
  // and are NOT quantized in W8A8/SmoothQuant checkpoints for now.
  const QuantArgs no_quant_args{};

  conv1d_ = register_module("conv1d",
                            ColumnParallelLinear(args.linear_conv_kernel_dim(),
                                                 k_size_ * 2 + v_size_,
                                                 /*bias=*/false,
                                                 /*gather_output=*/false,
                                                 no_quant_args,
                                                 parallel_args.tp_group_,
                                                 options));

  in_proj_qkv_ = register_module("in_proj_qkv",
                                 ColumnParallelLinear(args.hidden_size(),
                                                      k_size_ * 2 + v_size_,
                                                      /*bias=*/false,
                                                      /*gather_output=*/false,
                                                      no_quant_args,
                                                      parallel_args.tp_group_,
                                                      options));

  in_proj_z_ = register_module("in_proj_z",
                               ColumnParallelLinear(args.hidden_size(),
                                                    v_size_,
                                                    /*bias=*/false,
                                                    /*gather_output=*/false,
                                                    no_quant_args,
                                                    parallel_args.tp_group_,
                                                    options));

  in_proj_b_ = register_module("in_proj_b",
                               ColumnParallelLinear(args.hidden_size(),
                                                    num_v_heads_,
                                                    /*bias=*/false,
                                                    /*gather_output=*/false,
                                                    no_quant_args,
                                                    parallel_args.tp_group_,
                                                    options));

  in_proj_a_ = register_module("in_proj_a",
                               ColumnParallelLinear(args.hidden_size(),
                                                    num_v_heads_,
                                                    /*bias=*/false,
                                                    /*gather_output=*/false,
                                                    no_quant_args,
                                                    parallel_args.tp_group_,
                                                    options));

  auto opts = options.dtype(torch::kBFloat16);
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
                                              /*enable_result_reduction=*/true,
                                              no_quant_args,
                                              parallel_args.tp_group_,
                                              options));

  norm_ = register_module(
      "norm", RmsNormGated(head_v_dim_, args.rms_norm_eps(), options));
  int64_t num_k_heads_per_shard = num_k_heads_ / tp_size_;
  int64_t num_v_heads_per_shard = num_v_heads_ / tp_size_;
  chunk_gated_delta_rule_ =
      register_module("chunk_gated_delta_rule",
                      xllm::kernel::mlu::ChunkGatedDeltaRule(
                          num_k_heads_per_shard, num_v_heads_per_shard));
}

void Qwen3_5GatedDeltaNetImpl::load_state_dict(const StateDict& state_dict) {
  const int32_t shard_tensor_count = 3;
  const std::vector<int64_t> shard_sizes = {
      k_size_ / tp_size_, k_size_ / tp_size_, v_size_ / tp_size_};

  if (auto w = state_dict.get_tensor("conv1d.weight"); w.defined()) {
    conv1d_->load_state_dict(
        StateDict({{"weight", w.squeeze(1)}}), shard_tensor_count, shard_sizes);
  }

  auto qkv_state_dict = state_dict.get_dict_with_prefix("in_proj_qkv.");
  if (qkv_state_dict.size() > 0 && !in_proj_qkv_->is_weight_loaded()) {
    in_proj_qkv_->load_state_dict(
        qkv_state_dict, shard_tensor_count, shard_sizes);
  }

  auto z_state_dict = state_dict.get_dict_with_prefix("in_proj_z.");
  if (z_state_dict.size() > 0 && !in_proj_z_->is_weight_loaded()) {
    in_proj_z_->load_state_dict(z_state_dict);
  }

  auto b_state_dict = state_dict.get_dict_with_prefix("in_proj_b.");
  if (b_state_dict.size() > 0 && !in_proj_b_->is_weight_loaded()) {
    in_proj_b_->load_state_dict(b_state_dict);
  }

  auto a_state_dict = state_dict.get_dict_with_prefix("in_proj_a.");
  if (a_state_dict.size() > 0 && !in_proj_a_->is_weight_loaded()) {
    in_proj_a_->load_state_dict(a_state_dict);
  }

  o_proj_->load_state_dict(state_dict.get_dict_with_prefix("out_proj."));
  if (auto w = state_dict.get_tensor("norm.weight"); w.defined()) {
    norm_->load_state_dict(StateDict({{"weight", w}}));
  }
  weight::load_sharded_weight(state_dict,
                              "dt_bias",
                              /*dim=*/0,
                              static_cast<int32_t>(rank_),
                              static_cast<int32_t>(tp_size_),
                              dt_bias_,
                              dt_bias_is_loaded_);
  weight::load_sharded_weight(state_dict,
                              "A_log",
                              /*dim=*/0,
                              static_cast<int32_t>(rank_),
                              static_cast<int32_t>(tp_size_),
                              A_log_,
                              A_log_is_loaded_);
}

void Qwen3_5GatedDeltaNetImpl::verify_loaded_weights(
    const std::string& prefix) const {
  CHECK(conv1d_ && conv1d_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "conv1d.weight";
  CHECK(in_proj_qkv_ && in_proj_qkv_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "in_proj_qkv.weight";
  CHECK(in_proj_z_ && in_proj_z_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "in_proj_z.weight";
  CHECK(in_proj_b_ && in_proj_b_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "in_proj_b.weight";
  CHECK(in_proj_a_ && in_proj_a_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "in_proj_a.weight";
  CHECK(dt_bias_is_loaded_)
      << "Missing required weight after all shards loaded: " << prefix
      << "dt_bias";
  CHECK(A_log_is_loaded_) << "Missing required weight after all shards loaded: "
                          << prefix << "A_log";
}

torch::Tensor Qwen3_5GatedDeltaNetImpl::get_linear_state_indices(
    const ModelInputParams& input_params,
    const torch::Device& device) const {
  CHECK(!input_params.embedding.linear_state_ids.empty())
      << "linear_state_ids must be populated for gated delta net";
  if (input_params.embedding.linear_state_indices.defined()) {
    return input_params.embedding.linear_state_indices;
  }
  return torch::tensor(
      input_params.embedding.linear_state_ids,
      torch::TensorOptions().dtype(torch::kInt).device(device));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
Qwen3_5GatedDeltaNetImpl::split_mixed_qkv(
    const torch::Tensor& mixed_qkv) const {
  return layer::split_mixed_qkv(mixed_qkv,
                                num_k_heads_ / tp_size_,
                                num_v_heads_ / tp_size_,
                                head_k_dim_,
                                head_v_dim_);
}

int64_t Qwen3_5GatedDeltaNetImpl::get_checkpoint_stride(
    const KVCache& kv_cache) const {
  torch::Tensor conv_cache = kv_cache.get_conv_cache();
  torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
  CHECK_GT(conv_cache.size(0), 0)
      << "Qwen3.5 MLU GDN conv cache must have rows";
  CHECK_EQ(ssm_cache.size(0) % conv_cache.size(0), 0)
      << "Qwen3.5 MLU GDN SSM checkpoint layout mismatch";
  const int64_t checkpoint_stride = ssm_cache.size(0) / conv_cache.size(0);
  return checkpoint_stride;
}

torch::Tensor Qwen3_5GatedDeltaNetImpl::forward(
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params) {
  int64_t num_tokens = hidden_states.size(0);
  if (input_params.is_spec_verify) {
    CHECK(attn_metadata.is_chunked_prefill)
        << "Qwen3.5 MLU GDN Spec Verify requires chunked-prefill Dense "
           "Validate Span";
  }

  // ============================================================
  // Part 1: Input Projection
  // ============================================================
  auto mixed_qkv = in_proj_qkv_->forward(hidden_states);
  auto z = in_proj_z_->forward(hidden_states);
  z = z.view({z.size(0), -1, head_v_dim_});

  auto b = in_proj_b_->forward(hidden_states).contiguous();
  auto a = in_proj_a_->forward(hidden_states).contiguous();

  // ============================================================
  // Part 2: Core Attention
  // ============================================================
  torch::Tensor core_attn_out =
      torch::zeros({num_tokens, num_v_heads_ / tp_size_, head_v_dim_},
                   hidden_states.options());

  torch::Tensor conv_cache = kv_cache.get_conv_cache().transpose(-1, -2);
  torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
  const int64_t checkpoint_stride = get_checkpoint_stride(kv_cache);
  if (input_params.is_spec_verify) {
    CHECK_EQ(checkpoint_stride, attn_metadata.max_query_len)
        << "Qwen3.5 MLU GDN Spec Verify checkpoint_stride must equal "
           "q_max_seq_len";
    const int64_t expected_conv_state_len =
        (attn_metadata.max_query_len - 1) + (conv_kernel_size_ - 1);
    CHECK_EQ(conv_cache.size(2), expected_conv_state_len)
        << "Qwen3.5 MLU GDN Spec Verify conv_state_len mismatch";
  }
  torch::Tensor last_recurrent_state;
  auto conv_weight = conv1d_->weight();
  auto device = mixed_qkv.device();
  torch::Tensor logical_state_indices =
      get_linear_state_indices(input_params, device);
  torch::Tensor linear_state_base_indices =
      build_linear_state_base_indices(logical_state_indices, checkpoint_stride);

  if (input_params.is_spec_verify) {
    const int64_t q_max_seq_len = attn_metadata.max_query_len;
    mixed_qkv = xllm::kernel::mlu::causal_conv1d_update_decode(
        mixed_qkv,
        conv_cache,
        conv_weight,
        /*bias_opt=*/std::nullopt,
        logical_state_indices,
        /*activation=*/true,
        /*pad_slot_id=*/-1,
        attn_metadata.q_cu_seq_lens,
        static_cast<int32_t>(q_max_seq_len),
        input_params.num_accepted_tokens);

    auto [q, k, v] = split_mixed_qkv(mixed_qkv);
    torch::Tensor ssm_state_indices = build_rebased_ssm_state_indices(
        logical_state_indices, checkpoint_stride, q_max_seq_len);

    xllm::kernel::FusedSigmoidGatingDeltaRuleUpdateParams params;
    params.A_log = A_log_;
    params.a = a;
    params.dt_bias = dt_bias_;
    params.q = q;
    params.k = k;
    params.v = v;
    params.b = b;
    params.initial_state_source = ssm_cache;
    params.initial_state_indices = ssm_state_indices;
    params.cu_seqlens = attn_metadata.q_cu_seq_lens;
    params.scale =
        static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_k_dim_)));
    params.num_accepted_tokens = input_params.num_accepted_tokens;
    params.use_qk_l2norm_in_kernel = true;

    core_attn_out = xllm::kernel::fused_sigmoid_gating_delta_rule_update(params)
                        .squeeze(/*dim=*/0)
                        .contiguous();
  } else if (attn_metadata.is_prefill || attn_metadata.is_chunked_prefill) {
    // [num_tokens, channels] -> [channels, num_tokens]
    mixed_qkv = mixed_qkv.transpose(0, 1);
    int64_t seq_len = mixed_qkv.size(-1);
    std::optional<torch::Tensor> bias = std::nullopt;
    std::optional<torch::Tensor> initial_state_idx = std::nullopt;
    std::optional<torch::Tensor> num_accepted_tokens = std::nullopt;
    mixed_qkv =
        xllm::kernel::mlu::causal_conv1d_fn(mixed_qkv,
                                            conv_weight,
                                            conv_cache,
                                            attn_metadata.q_cu_seq_lens,
                                            attn_metadata.batch,
                                            attn_metadata.token_block_offset,
                                            attn_metadata.tot,
                                            bias,
                                            logical_state_indices,
                                            attn_metadata.has_initial_states,
                                            initial_state_idx,
                                            num_accepted_tokens,
                                            /*inplace_final_state=*/true);
    mixed_qkv = mixed_qkv.transpose(0, 1);
    auto [q_conv, k_conv, v_conv, g, beta] =
        xllm::kernel::mlu::fused_post_conv_prep(mixed_qkv,
                                                a,
                                                b,
                                                A_log_,
                                                dt_bias_,
                                                num_k_heads_ / tp_size_,
                                                head_k_dim_,
                                                head_v_dim_,
                                                /*apply_l2norm=*/true,
                                                /*output_g_exp=*/false);
    q_conv = q_conv.unsqueeze(0);
    k_conv = k_conv.unsqueeze(0);
    v_conv = v_conv.unsqueeze(0);
    g = g.unsqueeze(0);
    beta = beta.unsqueeze(0);

    auto cu_seqlens = attn_metadata.q_cu_seq_lens.contiguous();
    auto chunk_indices = attn_metadata.chunk_indices.contiguous();
    auto initial_state = ssm_cache.index({linear_state_base_indices});
    initial_state.index_put_(
        {~attn_metadata.has_initial_states, torch::indexing::Ellipsis}, 0.0f);
    std::tie(core_attn_out, last_recurrent_state) =
        chunk_gated_delta_rule_->forward(q_conv,
                                         k_conv,
                                         v_conv,
                                         g,
                                         beta,
                                         initial_state,
                                         cu_seqlens,
                                         chunk_indices,
                                         /*output_final_state=*/true,
                                         /*use_qk_l2norm_in_kernel=*/false);
    ssm_cache.index_put_({linear_state_base_indices},
                         last_recurrent_state.to(ssm_cache.dtype()));
  } else {
    mixed_qkv =
        xllm::kernel::mlu::causal_conv1d_update_decode(mixed_qkv,
                                                       conv_cache,
                                                       conv_weight,
                                                       std::nullopt,
                                                       logical_state_indices,
                                                       /*activation=*/true,
                                                       /*pad_slot_id=*/-1);

    double scale = 1.0 / std::sqrt(static_cast<double>(head_k_dim_));
    std::tie(core_attn_out, last_recurrent_state) =
        xllm::kernel::mlu::fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv,
            a,
            b,
            A_log_,
            dt_bias_,
            scale,
            ssm_cache,
            linear_state_base_indices,
            /*use_qk_l2norm_in_kernel=*/true);
  }

  // ============================================================
  // Part 3: Output Projection
  // ============================================================
  auto z_shape_og = z.sizes().vec();
  core_attn_out = core_attn_out.view({-1, core_attn_out.size(-1)});
  z = z.view({-1, z.size(-1)});
  auto norm_out = norm_->forward(core_attn_out, z);
  norm_out = norm_out.view(z_shape_og);
  norm_out = norm_out.view({-1, norm_out.size(-1) * norm_out.size(-2)});

  auto output = o_proj_->forward(norm_out);
  return output;
}

}  // namespace layer
}  // namespace xllm

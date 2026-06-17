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

#include "minimax_m2_attention.h"

#include "core/layers/common/rotary_embedding_util.h"

namespace xllm {
namespace layer {

MiniMaxM2AttentionImpl::MiniMaxM2AttentionImpl(const ModelContext& context) {
  const auto& args = context.get_model_args();
  auto quant_args = context.get_quant_args();
  if (quant_args.quant_method() == "fp8") {
    quant_args.quant_method("");
  }
  const auto& parallel_args = context.get_parallel_args();
  const auto& options = context.get_tensor_options();
  const int64_t tp_size = parallel_args.tp_group_->world_size();
  const int64_t total_num_heads = args.n_heads();
  const int64_t total_num_kv_heads = args.n_kv_heads().value_or(args.n_heads());

  CHECK(total_num_heads % tp_size == 0);
  num_heads_ = total_num_heads / tp_size;
  if (total_num_kv_heads >= tp_size) {
    CHECK(total_num_kv_heads % tp_size == 0);
    num_kv_heads_ = total_num_kv_heads / tp_size;
    num_kv_head_replicas_ = 1;
  } else {
    CHECK(tp_size % total_num_kv_heads == 0);
    num_kv_heads_ = 1;
    num_kv_head_replicas_ = tp_size / total_num_kv_heads;
  }

  head_dim_ = args.head_dim();
  q_size_ = num_heads_ * head_dim_;
  kv_size_ = num_kv_heads_ * head_dim_;
  CHECK_EQ(num_heads_ % num_kv_heads_, 0)
      << "MiniMax local attention heads must be divisible by local kv heads";
  scaling_ = std::sqrt(1.0f / head_dim_);
  sliding_window_ = args.sliding_window();

  qkv_proj_ = register_module("qkv_proj",
                              layer::QKVParallelLinear(args.hidden_size(),
                                                       num_heads_,
                                                       num_kv_heads_,
                                                       head_dim_,
                                                       num_kv_head_replicas_,
                                                       /*bias=*/false,
                                                       /*gather_output=*/false,
                                                       parallel_args,
                                                       options));

  o_proj_ =
      register_module("o_proj",
                      layer::RowParallelLinear(total_num_heads * head_dim_,
                                               args.hidden_size(),
                                               /*bias=*/false,
                                               /*input_is_parallelized=*/true,
                                               /*enable_result_reduction=*/true,
                                               quant_args,
                                               parallel_args.tp_group_,
                                               options));

  q_norm_tp_ =
      register_module("q_norm",
                      MiniMaxTensorParallelRMSNorm(q_size_,
                                                   total_num_heads * head_dim_,
                                                   /*replica_factor=*/1,
                                                   args.rms_norm_eps(),
                                                   parallel_args.tp_group_,
                                                   options));
  k_norm_tp_ = register_module(
      "k_norm",
      MiniMaxTensorParallelRMSNorm(kv_size_,
                                   total_num_kv_heads * head_dim_,
                                   num_kv_head_replicas_,
                                   args.rms_norm_eps(),
                                   parallel_args.tp_group_,
                                   options));

  const int64_t rotary_dim =
      args.rotary_dim() > 0 ? args.rotary_dim() : args.head_dim();
  const auto inv_freq =
      layer::rotary::compute_inv_freq(rotary_dim, args.rope_theta(), options);
  rotary_emb_ = register_module(
      "rope",
      std::make_shared<RotaryEmbeddingGeneric>(rotary_dim,
                                               args.max_position_embeddings(),
                                               inv_freq,
                                               /*interleaved=*/false,
                                               options));
  attn_ = register_module("attn",
                          layer::Attention(num_heads_,
                                           head_dim_,
                                           scaling_,
                                           num_kv_heads_,
                                           args.sliding_window()));
}

torch::Tensor MiniMaxM2AttentionImpl::forward(
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    const layer::AttentionMetadata& attn_metadata,
    KVCache& kv_cache) {
  if (attn_metadata.is_dummy) {
    return torch::zeros_like(hidden_states);
  }

  auto qkv = qkv_proj_->forward(hidden_states);
  auto q = qkv.slice(/*dim=*/-1, 0, q_size_);
  auto k = qkv.slice(/*dim=*/-1, q_size_, q_size_ + kv_size_);
  auto v = qkv.slice(/*dim=*/-1, q_size_ + kv_size_, q_size_ + 2 * kv_size_);

  const int64_t num_tokens = q.size(0);
  std::tie(q, k) = forward_qk_rms_norm(q_norm_tp_, k_norm_tp_, q, k);

  auto q_heads = q.view({num_tokens, num_heads_, head_dim_});
  auto k_heads = k.view({num_tokens, num_kv_heads_, head_dim_});
  auto v_heads = v.view({num_tokens, num_kv_heads_, head_dim_});

  std::tie(q_heads, k_heads) =
      rotary_emb_->forward(q_heads, k_heads, positions);

  auto q_flat = q_heads.reshape({num_tokens, q_size_}).contiguous();
  auto k_flat = k_heads.reshape({num_tokens, kv_size_}).contiguous();
  auto v_flat = v_heads.reshape({num_tokens, kv_size_}).contiguous();
  auto out = std::get<0>(
      attn_->forward(attn_metadata, q_flat, k_flat, v_flat, kv_cache));
  return o_proj_->forward(out);
}

void MiniMaxM2AttentionImpl::load_state_dict(const StateDict& state_dict) {
  qkv_proj_->load_state_dict(state_dict, {"q_proj.", "k_proj.", "v_proj."});
  o_proj_->load_state_dict(state_dict.get_dict_with_prefix("o_proj."));
  q_norm_tp_->load_state_dict(state_dict.get_dict_with_prefix("q_norm."));
  k_norm_tp_->load_state_dict(state_dict.get_dict_with_prefix("k_norm."));
}

}  // namespace layer
}  // namespace xllm

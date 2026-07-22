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

#include "minimax_m2_attention.h"

#include <glog/logging.h>

#include <cmath>
#include <tuple>

namespace xllm {
namespace layer {

MiniMaxM2AttentionImpl::MiniMaxM2AttentionImpl(const ModelContext& context) {
  const auto& args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& parallel_args = context.get_parallel_args();
  const auto& options = context.get_tensor_options();
  const int64_t tp_size = parallel_args.tp_group_->world_size();
  const int64_t total_num_heads = args.n_heads();
  const int64_t total_num_kv_heads = args.n_kv_heads().value_or(args.n_heads());

  CHECK_EQ(total_num_heads % tp_size, 0);
  num_heads_ = total_num_heads / tp_size;
  if (total_num_kv_heads >= tp_size) {
    CHECK_EQ(total_num_kv_heads % tp_size, 0);
    num_kv_heads_ = total_num_kv_heads / tp_size;
    num_kv_head_replicas_ = 1;
  } else {
    CHECK_EQ(tp_size % total_num_kv_heads, 0);
    num_kv_heads_ = 1;
    num_kv_head_replicas_ = tp_size / total_num_kv_heads;
  }

  head_dim_ = args.head_dim();
  q_size_ = num_heads_ * head_dim_;
  kv_size_ = num_kv_heads_ * head_dim_;
  CHECK_EQ(num_heads_ % num_kv_heads_, 0)
      << "MiniMax-M2 local attention heads must be divisible by local KV "
         "heads.";
  scaling_ = std::sqrt(1.0f / static_cast<float>(head_dim_));

  qkv_proj_ = register_module("qkv_proj",
                              QKVParallelLinear(args.hidden_size(),
                                                num_heads_,
                                                num_kv_heads_,
                                                head_dim_,
                                                num_kv_head_replicas_,
                                                /*bias=*/false,
                                                /*gather_output=*/false,
                                                parallel_args,
                                                options,
                                                quant_args));

  o_proj_ = register_module("o_proj",
                            RowParallelLinear(total_num_heads * head_dim_,
                                              args.hidden_size(),
                                              /*bias=*/false,
                                              /*input_is_parallelized=*/true,
                                              /*enable_result_reduction=*/true,
                                              quant_args,
                                              parallel_args.tp_group_,
                                              options));

  q_norm_tp_ = register_module(
      "q_norm",
      MiniMaxM2TensorParallelRMSNorm(q_size_,
                                     total_num_heads * head_dim_,
                                     /*replica_factor=*/1,
                                     args.rms_norm_eps(),
                                     parallel_args.tp_group_,
                                     options));
  k_norm_tp_ = register_module(
      "k_norm",
      MiniMaxM2TensorParallelRMSNorm(kv_size_,
                                     total_num_kv_heads * head_dim_,
                                     num_kv_head_replicas_,
                                     args.rms_norm_eps(),
                                     parallel_args.tp_group_,
                                     options));

  const int64_t rotary_dim =
      args.rotary_dim() > 0 ? args.rotary_dim() : args.head_dim();
  rotary_emb_ = register_module("rope",
                                RotaryEmbedding(rotary_dim,
                                                args.max_position_embeddings(),
                                                args.rope_theta(),
                                                /*interleaved=*/false,
                                                options));

  attn_ = register_module("attn",
                          Attention(num_heads_,
                                    head_dim_,
                                    scaling_,
                                    num_kv_heads_,
                                    args.sliding_window()));
}

torch::Tensor MiniMaxM2AttentionImpl::forward(
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache) {
  if (attn_metadata.is_dummy) {
    return torch::zeros_like(hidden_states);
  }

  torch::Tensor qkv = qkv_proj_->forward(hidden_states);
  torch::Tensor q = qkv.slice(/*dim=*/-1, /*start=*/0, /*end=*/q_size_);
  torch::Tensor k = qkv.slice(/*dim=*/-1,
                              /*start=*/q_size_,
                              /*end=*/q_size_ + kv_size_);
  torch::Tensor v = qkv.slice(/*dim=*/-1, /*start=*/q_size_ + kv_size_);

  const int64_t num_tokens = q.size(0);
  std::tie(q, k) = forward_minimax_m2_qk_rms_norm(q_norm_tp_, k_norm_tp_, q, k);

  torch::Tensor q_heads =
      q.view({num_tokens, num_heads_, head_dim_}).contiguous();
  torch::Tensor k_heads =
      k.view({num_tokens, num_kv_heads_, head_dim_}).contiguous();
  torch::Tensor position_ids = positions;
  if (position_ids.dim() == 2) {
    position_ids = position_ids[0];
  }
  rotary_emb_->forward(
      q_heads,
      k_heads,
      position_ids,
      attn_metadata.q_cu_seq_lens,
      attn_metadata.max_query_len,
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill);

  torch::Tensor q_flat = q_heads.reshape({num_tokens, q_size_}).contiguous();
  torch::Tensor k_flat = k_heads.reshape({num_tokens, kv_size_}).contiguous();
  torch::Tensor v_flat = v.reshape({num_tokens, kv_size_}).contiguous();
  torch::Tensor out = std::get<0>(
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

/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/layers/common/qwen2_attention.h"

#include <glog/logging.h>

#include <tuple>

namespace xllm {
namespace layer {

Qwen2AttentionImpl::Qwen2AttentionImpl(const ModelContext& context) {
  const auto& args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& parallel_args = context.get_parallel_args();
  const auto& options = context.get_tensor_options();
  const int64_t tp_size = parallel_args.tp_group_->world_size();
  const int64_t total_num_heads = args.n_heads();
  const int64_t total_num_kv_heads = args.n_kv_heads().value_or(args.n_heads());

  CHECK(total_num_heads % tp_size == 0);
  num_heads_ = total_num_heads / tp_size;

  int64_t num_kv_head_replicas = 1;
  if (total_num_kv_heads >= tp_size) {
    CHECK(total_num_kv_heads % tp_size == 0);
    num_kv_heads_ = total_num_kv_heads / tp_size;
  } else {
    CHECK(tp_size % total_num_kv_heads == 0);
    num_kv_heads_ = 1;
    num_kv_head_replicas = tp_size / total_num_kv_heads;
  }

  head_dim_ = args.head_dim();
  q_size_ = num_heads_ * head_dim_;
  kv_size_ = num_kv_heads_ * head_dim_;

  qkv_proj_ = register_module("qkv_proj",
                              QKVParallelLinear(args.hidden_size(),
                                                num_heads_,
                                                num_kv_heads_,
                                                args.head_dim(),
                                                num_kv_head_replicas,
                                                args.attention_bias(),
                                                /*gather_output=*/false,
                                                parallel_args,
                                                options,
                                                quant_args));

  o_proj_ = register_module("o_proj",
                            RowParallelLinear(total_num_heads * args.head_dim(),
                                              args.hidden_size(),
                                              /*bias=*/false,
                                              /*input_is_parallelized=*/true,
                                              /*enable_result_reduction=*/true,
                                              quant_args,
                                              parallel_args.tp_group_,
                                              options));

  rotary_emb_ =
      register_module("rope",
                      MRotaryEmbedding(/*rotary_dim=*/head_dim_,
                                       args.max_position_embeddings(),
                                       args.rope_theta(),
                                       /*interleaved=*/false,
                                       args.rope_scaling_mrope_section(),
                                       options));

  const float scaling = std::sqrt(1.0f / head_dim_);
  attn_ = register_module("attn",
                          Attention(num_heads_,
                                    head_dim_,
                                    scaling,
                                    num_kv_heads_,
                                    args.sliding_window()));
}

torch::Tensor Qwen2AttentionImpl::forward(
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache) {
  auto qkv = qkv_proj_->forward(hidden_states);

  auto q = qkv.slice(/*dim=*/-1, 0, q_size_);
  auto k = qkv.slice(/*dim=*/-1, q_size_, q_size_ + kv_size_);
  auto v = qkv.slice(/*dim=*/-1, q_size_ + kv_size_, q_size_ + 2 * kv_size_);

  const int64_t num_tokens = qkv.size(/*dim=*/0);
  apply_qk_transform(qkv, q, k, positions, attn_metadata);
  q = q.view({num_tokens, q_size_});
  k = k.view({num_tokens, kv_size_});
  auto out = std::get<0>(attn_->forward(attn_metadata, q, k, v, kv_cache));
  return o_proj_->forward(out);
}

void Qwen2AttentionImpl::apply_qk_transform(
    torch::Tensor&,
    torch::Tensor& q,
    torch::Tensor& k,
    const torch::Tensor& positions,
    const AttentionMetadata& attn_metadata) {
  rotary_emb_->forward(q, k, positions, attn_metadata);
}

void Qwen2AttentionImpl::load_state_dict(const StateDict& state_dict) {
  q_proj_weight_seen_ = q_proj_weight_seen_ || state_dict.has("q_proj.weight");
  k_proj_weight_seen_ = k_proj_weight_seen_ || state_dict.has("k_proj.weight");
  v_proj_weight_seen_ = v_proj_weight_seen_ || state_dict.has("v_proj.weight");
  qkv_proj_->load_state_dict(state_dict, {"q_proj.", "k_proj.", "v_proj."});
  o_proj_->load_state_dict(state_dict.get_dict_with_prefix("o_proj."));
}

void Qwen2AttentionImpl::verify_loaded_weights(
    const std::string& prefix) const {
  if (!qkv_proj_->is_weight_loaded()) {
    CHECK(q_proj_weight_seen_)
        << "weight is not loaded for " << prefix + "q_proj.weight";
    CHECK(k_proj_weight_seen_)
        << "weight is not loaded for " << prefix + "k_proj.weight";
    CHECK(v_proj_weight_seen_)
        << "weight is not loaded for " << prefix + "v_proj.weight";
  }
  CHECK(qkv_proj_->is_weight_loaded()) << "weight is not loaded for " << prefix
                                       << "{q_proj,k_proj,v_proj}.weight";
  CHECK(o_proj_->is_weight_loaded())
      << "weight is not loaded for " << prefix + "o_proj.weight";
}

std::optional<torch::Tensor> Qwen2AttentionImpl::get_fp8_input_scale() const {
  return qkv_proj_->get_input_scale();
}

}  // namespace layer
}  // namespace xllm

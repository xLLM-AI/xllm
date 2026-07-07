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

#include "qwen3_5_attention.h"

#include <glog/logging.h>

#include <tuple>

#include "kernels/ops_api.h"
namespace xllm {
namespace layer {

Qwen3_5AttentionImpl::Qwen3_5AttentionImpl(const ModelArgs& args,
                                           const QuantArgs& quant_args,
                                           const ParallelArgs& parallel_args,
                                           const torch::TensorOptions& options,
                                           int32_t layer_id) {
  const int64_t tp_size = parallel_args.tp_group_->world_size();
  const int64_t total_num_heads = args.n_heads();
  const int64_t total_num_kv_heads = args.n_kv_heads().value_or(args.n_heads());
  layer_id_ = layer_id;
  rank_ = parallel_args.tp_group_->rank();
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
  scaling_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));
  attn_output_gate_ = args.attn_output_gate();
  mrope_cu_seq_lens_ = torch::zeros(2, torch::kInt32).to(options.device());
  use_split_qkv_ = quant_args.quant_method() == kQuantMethodSmoothquant;
  // 1. QKV linear
  if (use_split_qkv_) {
    q_proj_ = register_module(
        "q_proj",
        ColumnParallelLinear(
            args.hidden_size(),
            (attn_output_gate_ ? total_num_heads * 2 : total_num_heads) *
                args.head_dim(),
            /*bias=*/args.attention_bias(),
            /*gather_output=*/false,
            quant_args,
            parallel_args.tp_group_,
            options));
    k_proj_ = register_module(
        "k_proj",
        ColumnParallelLinear(args.hidden_size(),
                             total_num_kv_heads * args.head_dim(),
                             /*bias=*/args.attention_bias(),
                             /*gather_output=*/false,
                             quant_args,
                             parallel_args.tp_group_,
                             options,
                             LinearExtraArgs(),
                             num_kv_head_replicas_));
    v_proj_ = register_module(
        "v_proj",
        ColumnParallelLinear(args.hidden_size(),
                             total_num_kv_heads * args.head_dim(),
                             /*bias=*/args.attention_bias(),
                             /*gather_output=*/false,
                             quant_args,
                             parallel_args.tp_group_,
                             options,
                             LinearExtraArgs(),
                             num_kv_head_replicas_));
  } else {
    qkv_proj_ = register_module(
        "qkv_proj",
        QKVParallelLinear(args.hidden_size(),
                          attn_output_gate_ ? num_heads_ * 2 : num_heads_,
                          num_kv_heads_,
                          args.head_dim(),
                          num_kv_head_replicas_,
                          /*bias=*/args.attention_bias(),
                          /*gather_output=*/false,
                          parallel_args,
                          options,
                          quant_args));
  }

  // 2. O proj
  o_proj_ = register_module("o_proj",
                            RowParallelLinear(total_num_heads * head_dim_,
                                              args.hidden_size(),
                                              /*bias=*/false,
                                              /*input_is_parallelized=*/true,
                                              /*if_reduce_results=*/true,
                                              quant_args,
                                              parallel_args.tp_group_,
                                              options));

  // 3. Q norm
  q_norm_ = register_module(
      "q_norm", Qwen3NextRMSNorm(head_dim_, args.rms_norm_eps(), options));

  // 4. K norm
  k_norm_ = register_module(
      "k_norm", Qwen3NextRMSNorm(head_dim_, args.rms_norm_eps(), options));

  // 5. Attention
  attn_ = register_module("attn",
                          Attention(num_heads_,
                                    head_dim_,
                                    scaling_,
                                    num_kv_heads_,
                                    args.sliding_window()));

  // 6. Rotary embedding
  const int32_t rotary_dim =
      static_cast<int32_t>(head_dim_ * args.partial_rotary_factor());
  rotary_emb_ =
      register_module("rope",
                      MRotaryEmbedding(rotary_dim,
                                       args.max_position_embeddings(),
                                       args.rope_theta(),
                                       /*interleaved=*/false,
                                       args.rope_scaling_mrope_section(),
                                       options));
}

void Qwen3_5AttentionImpl::rotary_emb_forward(
    torch::Tensor& q,
    torch::Tensor& k,
    const torch::Tensor& positions,
    const AttentionMetadata& attn_metadata) {
  auto q_shape = q.sizes();
  auto k_shape = k.sizes();
  auto num_tokens = positions.size(-1);
  mrope_cu_seq_lens_[1] = num_tokens;

  xllm::kernel::RotaryParams rotary_params;
  rotary_params.sin = attn_metadata.mrope_sin;
  rotary_params.cos = attn_metadata.mrope_cos;
  rotary_params.position_ids = std::nullopt;
  rotary_params.cu_query_lens = mrope_cu_seq_lens_;
  rotary_params.interleaved = false;
  rotary_params.discrete = false;
  rotary_params.max_query_len = num_tokens;

  rotary_params.q = q.view({num_tokens, -1, head_dim_});
  xllm::kernel::apply_rotary(rotary_params);
  q = rotary_params.q.reshape(q_shape);

  rotary_params.q = k.view({num_tokens, -1, head_dim_});
  xllm::kernel::apply_rotary(rotary_params);
  k = rotary_params.q.reshape(k_shape);
}

torch::Tensor Qwen3_5AttentionImpl::forward(
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache) {
  torch::Tensor q, k, v;
  torch::Tensor gate;

  if (use_split_qkv_) {
    q = q_proj_->forward(hidden_states);
    k = k_proj_->forward(hidden_states);
    v = v_proj_->forward(hidden_states).contiguous();
  } else {
    // 1. qkv projection
    auto qkv = qkv_proj_->forward(hidden_states);
    if (attn_output_gate_) {
      // Split qkv for attn_output_gate case: [q_size*2, kv_size, kv_size]
      q = qkv.slice(/*dim=*/-1, 0, q_size_ * 2);
      k = qkv.slice(/*dim=*/-1, q_size_ * 2, q_size_ * 2 + kv_size_);
      v = qkv.slice(
          /*dim=*/-1, q_size_ * 2 + kv_size_, q_size_ * 2 + kv_size_ * 2);
      v = v.contiguous();
    } else {
      // Normal case: [q_size, kv_size, kv_size]
      q = qkv.slice(/*dim=*/-1, 0, q_size_);
      k = qkv.slice(/*dim=*/-1, q_size_, q_size_ + kv_size_);
      v = qkv.slice(/*dim=*/-1, q_size_ + kv_size_, q_size_ + 2 * kv_size_);
    }
  }

  if (attn_output_gate_) {
    // Split qkv for attn_output_gate case: [q_size*2, kv_size, kv_size]
    torch::Tensor q_gate = q;
    std::vector<int64_t> orig_shape;
    for (int64_t i = 0; i < q_gate.dim() - 1; i++) {
      orig_shape.push_back(q_gate.size(i));
    }
    std::vector<int64_t> new_shape = orig_shape;
    new_shape.push_back(num_heads_);
    new_shape.push_back(-1);
    torch::Tensor q_gate_reshaped = q_gate.reshape(new_shape);
    auto chunks = torch::chunk(q_gate_reshaped, 2, /*dim=*/-1);
    q = chunks[0];
    gate = chunks[1];

    std::vector<int64_t> q_new_shape = orig_shape;
    q_new_shape.push_back(-1);
    q = q.reshape(q_new_shape);

    std::vector<int64_t> gate_new_shape = orig_shape;
    gate_new_shape.push_back(-1);
    gate = gate.reshape(gate_new_shape);
  }

  const int64_t T = q.size(0);

  auto q_reshaped = q.reshape({T, num_heads_, head_dim_});
  auto q_normed = std::get<0>(q_norm_->forward(q_reshaped));
  auto k_reshaped = k.reshape({T, num_kv_heads_, head_dim_});
  auto k_normed = std::get<0>(k_norm_->forward(k_reshaped));

  q = q_normed.view({T, q_size_});
  k = k_normed.view({T, kv_size_});
  rotary_emb_forward(q, k, positions, attn_metadata);
  auto out = std::get<0>(attn_->forward(attn_metadata, q, k, v, kv_cache));

  if (attn_output_gate_) {
    gate = torch::sigmoid(gate);
    out = out * gate;
  }

  out = o_proj_->forward(out);
  return out;
}

void Qwen3_5AttentionImpl::load_state_dict(const StateDict& state_dict) {
  if (use_split_qkv_) {
    q_proj_->load_state_dict(state_dict.get_dict_with_prefix("q_proj."));
    k_proj_->load_state_dict(state_dict.get_dict_with_prefix("k_proj."));
    v_proj_->load_state_dict(state_dict.get_dict_with_prefix("v_proj."));
  } else {
    qkv_proj_->load_state_dict(state_dict, {"q_proj.", "k_proj.", "v_proj."});
  }
  o_proj_->load_state_dict(state_dict.get_dict_with_prefix("o_proj."));
  if (auto w = state_dict.get_tensor("q_norm.weight"); w.defined()) {
    q_norm_->load_state_dict(StateDict({{"weight", w}}));
  }
  if (auto w = state_dict.get_tensor("k_norm.weight"); w.defined()) {
    k_norm_->load_state_dict(StateDict({{"weight", w}}));
  }
}

}  // namespace layer
}  // namespace xllm

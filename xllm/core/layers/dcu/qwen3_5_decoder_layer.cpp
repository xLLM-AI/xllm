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

#include "qwen3_5_decoder_layer.h"

#include <glog/logging.h>

namespace xllm {
namespace layer {
namespace {

bool is_full_attention_layer(const ModelArgs& model_args, int32_t layer_id) {
  auto layer_types = model_args.layer_types();
  if (layer_types.empty()) {
    int32_t interval = model_args.full_attention_interval();
    return (layer_id + 1) % interval == 0;
  }
  if (layer_id >= 0 && layer_id < static_cast<int32_t>(layer_types.size())) {
    return layer_types[layer_id] == "full_attention";
  }
  return true;
}

bool is_moe_layer(const ModelArgs& model_args, int32_t layer_id) {
  const auto& mlp_only_layers = model_args.mlp_only_layers();
  return std::count(mlp_only_layers.begin(), mlp_only_layers.end(), layer_id) ==
             0 &&
         model_args.n_routed_experts() > 0 &&
         (layer_id + 1) % model_args.decoder_sparse_step() == 0;
}

}  // namespace

Qwen3_5DecoderLayerImpl::Qwen3_5DecoderLayerImpl(const ModelContext& context,
                                                 int32_t layer_id)
    : parallel_args_(context.get_parallel_args()) {
  const auto& model_args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& options = context.get_tensor_options();

  is_full_attention_ = is_full_attention_layer(model_args, layer_id);
  const bool use_moe = is_moe_layer(model_args, layer_id);

  // ---- Attention ----
  if (is_full_attention_) {
    const int64_t tp_size = parallel_args_.tp_group_->world_size();
    const int64_t total_num_heads = model_args.n_heads();
    const int64_t total_num_kv_heads =
        model_args.n_kv_heads().value_or(model_args.n_heads());
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

    head_dim_ = model_args.head_dim();
    q_size_ = num_heads_ * head_dim_;
    kv_size_ = num_kv_heads_ * head_dim_;
    scaling_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    attn_output_gate_ = model_args.attn_output_gate();

    qkv_proj_ = register_module(
        "qkv_proj",
        QKVParallelLinear(model_args.hidden_size(),
                          attn_output_gate_ ? num_heads_ * 2 : num_heads_,
                          num_kv_heads_,
                          model_args.head_dim(),
                          num_kv_head_replicas_,
                          /*bias=*/model_args.attention_bias(),
                          /*gather_output=*/false,
                          parallel_args_,
                          options));

    o_proj_ = register_module("o_proj",
                              RowParallelLinear(total_num_heads * head_dim_,
                                                model_args.hidden_size(),
                                                /*bias=*/false,
                                                /*input_is_parallelized=*/true,
                                                /*if_reduce_results=*/true,
                                                quant_args,
                                                parallel_args_.tp_group_,
                                                options));

    q_norm_ = register_module(
        "q_norm",
        Qwen3NextRMSNorm(head_dim_, model_args.rms_norm_eps(), options));
    k_norm_ = register_module(
        "k_norm",
        Qwen3NextRMSNorm(head_dim_, model_args.rms_norm_eps(), options));

    attn_ = register_module("attn",
                            Attention(num_heads_,
                                      head_dim_,
                                      scaling_,
                                      num_kv_heads_,
                                      model_args.sliding_window()));

    const int32_t rotary_dim =
        static_cast<int32_t>(head_dim_ * model_args.partial_rotary_factor());
    rotary_emb_ = register_module(
        "rope",
        MRotaryEmbedding(rotary_dim,
                         model_args.max_position_embeddings(),
                         model_args.rope_theta(),
                         /*interleaved=*/false,
                         model_args.rope_scaling_mrope_section(),
                         options));
  } else {
    linear_attn_ = register_module(
        "linear_attn",
        Qwen3_5GatedDeltaNet(model_args, quant_args, parallel_args_, options));
  }

  // ---- Layer Norms ----
  input_norm_ = register_module(
      "input_layernorm",
      Qwen3NextRMSNorm(
          model_args.hidden_size(), model_args.rms_norm_eps(), options));
  post_norm_ = register_module(
      "post_attention_layernorm",
      Qwen3NextRMSNorm(
          model_args.hidden_size(), model_args.rms_norm_eps(), options));

  // ---- MLP / MoE ----
  if (use_moe) {
    moe_mlp_ = register_module("mlp",
                               FusedMoE(model_args,
                                        FusedMoEArgs{.is_gated = true},
                                        quant_args,
                                        parallel_args_,
                                        options));
  } else {
    mlp_ = register_module("mlp",
                           DenseMLP(model_args.hidden_size(),
                                    model_args.intermediate_size(),
                                    true,
                                    false,
                                    model_args.hidden_act(),
                                    /*enable_result_reduction=*/true,
                                    quant_args,
                                    parallel_args_.tp_group_,
                                    options));
  }
}

void Qwen3_5DecoderLayerImpl::load_state_dict(const StateDict& state_dict) {
  if (is_full_attention_) {
    qkv_proj_->load_state_dict(state_dict.get_dict_with_prefix("self_attn."),
                               {"q_proj.", "k_proj.", "v_proj."});
    o_proj_->load_state_dict(
        state_dict.get_dict_with_prefix("self_attn.o_proj."));
    q_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("self_attn.q_norm."));
    k_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("self_attn.k_norm."));
  } else {
    linear_attn_->load_state_dict(
        state_dict.get_dict_with_prefix("linear_attn."));
  }
  input_norm_->load_state_dict(
      state_dict.get_dict_with_prefix("input_layernorm."));
  post_norm_->load_state_dict(
      state_dict.get_dict_with_prefix("post_attention_layernorm."));
  if (moe_mlp_) {
    moe_mlp_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));
  } else {
    mlp_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));
  }
}

torch::Tensor Qwen3_5DecoderLayerImpl::forward(
    torch::Tensor& x,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params) {
  // Pre-attention norm (with residual)
  if (!residual.has_value()) {
    residual = x;
    x = std::get<0>(input_norm_->forward(x));
  } else {
    auto orig_dtype = x.dtype();
    x = x + residual.value();
    residual = x;
    x = x.to(orig_dtype);
    x = std::get<0>(input_norm_->forward(x));
  }

  // ---- Attention ----
  if (is_full_attention_) {
    auto qkv = qkv_proj_->forward(x);
    torch::Tensor q, k, v;
    torch::Tensor gate;

    if (attn_output_gate_) {
      auto q_gate = qkv.slice(/*dim=*/-1, 0, q_size_ * 2);
      k = qkv.slice(/*dim=*/-1, q_size_ * 2, q_size_ * 2 + kv_size_);
      v = qkv.slice(
          /*dim=*/-1, q_size_ * 2 + kv_size_, q_size_ * 2 + kv_size_ * 2);

      std::vector<int64_t> orig_shape;
      for (int64_t i = 0; i < q_gate.dim() - 1; i++) {
        orig_shape.push_back(q_gate.size(i));
      }
      std::vector<int64_t> new_shape = orig_shape;
      new_shape.push_back(num_heads_);
      new_shape.push_back(-1);
      auto q_gate_reshaped = q_gate.reshape(new_shape);
      auto chunks = torch::chunk(q_gate_reshaped, 2, /*dim=*/-1);
      q = chunks[0];
      gate = chunks[1];

      std::vector<int64_t> q_new_shape = orig_shape;
      q_new_shape.push_back(-1);
      q = q.reshape(q_new_shape);
    } else {
      q = qkv.slice(/*dim=*/-1, 0, q_size_);
      k = qkv.slice(/*dim=*/-1, q_size_, q_size_ + kv_size_);
      v = qkv.slice(/*dim=*/-1, q_size_ + kv_size_, q_size_ + 2 * kv_size_);
    }

    const int64_t T = q.size(0);

    auto q_reshaped = q.reshape({T, num_heads_, head_dim_});
    q = std::get<0>(q_norm_->forward(q_reshaped)).view({T, q_size_});
    auto k_reshaped = k.reshape({T, num_kv_heads_, head_dim_});
    k = std::get<0>(k_norm_->forward(k_reshaped)).view({T, kv_size_});

    rotary_emb_->forward(q, k, positions, attn_metadata);
    q = q.view({T, q_size_});
    k = k.view({T, kv_size_});

    x = std::get<0>(attn_->forward(attn_metadata, q, k, v, kv_cache));

    if (attn_output_gate_) {
      gate = torch::sigmoid(gate).reshape({T, q_size_});
      x = x * gate;
    }

    x = o_proj_->forward(x);
  } else {
    x = linear_attn_->forward(x, attn_metadata, kv_cache, input_params);
  }

  // Post-attention norm (with residual)
  std::tie(x, residual) = post_norm_->forward(x, residual);

  // ---- MLP / MoE ----
  if (moe_mlp_) {
    x = moe_mlp_->forward(x, input_params);
  } else {
    x = mlp_->forward(x);
  }

  return x;
}

}  // namespace layer
}  // namespace xllm

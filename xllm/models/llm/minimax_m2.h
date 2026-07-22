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

#pragma once

#include <unordered_set>
#include <vector>

#include "core/layers/minimax_m2_decoder_layer.h"
#include "llm_model_base.h"

namespace xllm {

class MiniMaxM2ModelImpl
    : public LlmModelImplBase<layer::MiniMaxM2DecoderLayer> {
 public:
  explicit MiniMaxM2ModelImpl(const ModelContext& context)
      : LlmModelImplBase<layer::MiniMaxM2DecoderLayer>(
            "minimax_m2",
            context.get_model_args()) {
    const auto& model_args = context.get_model_args();

    layers_.reserve(model_args.n_layers());
    embed_tokens_ =
        register_module("embed_tokens", layer::WordEmbedding(context));
    norm_ = register_module("norm", layer::RMSNorm(context));

    for (int32_t i = 0; i < model_args.n_layers(); ++i) {
      layer::MiniMaxM2DecoderLayer layer =
          layer::MiniMaxM2DecoderLayer(context, i);
      layers_.emplace_back(layer);
    }
  }
};
TORCH_MODULE(MiniMaxM2Model);

class MiniMaxM2ForCausalLMImpl : public LlmForCausalLMImplBase<MiniMaxM2Model> {
 public:
  explicit MiniMaxM2ForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<MiniMaxM2Model>(context) {}
};
TORCH_MODULE(MiniMaxM2ForCausalLM);

REGISTER_CAUSAL_MODEL(minimax_m2, MiniMaxM2ForCausalLM);

REGISTER_MODEL_ARGS(minimax_m2, [&] {
  LOAD_ARG_OR(model_type, "model_type", "minimax_m2");
  LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");
  LOAD_ARG_OR(attention_bias, "attention_bias", false);
  LOAD_ARG_OR(attention_dropout, "attention_dropout", 0.0f);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 200019);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 200020);
  LOAD_ARG_OR(head_dim, "head_dim", 128);
  LOAD_ARG_OR(rotary_dim, "rotary_dim", 64);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "hidden_size", 3072);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 1536);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 196608);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 62);
  LOAD_ARG_OR(moe_intermediate_size, "intermediate_size", 1536);
  SET_ARG(n_shared_experts, 0);
  LOAD_ARG_OR(norm_topk_prob, "norm_topk_prob", true);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 48);
  LOAD_ARG_OR(num_experts, "num_local_experts", 256);
  LOAD_ARG_OR(n_routed_experts, "num_local_experts", 256);
  LOAD_ARG_OR(num_experts_per_tok, "num_experts_per_tok", 8);
  if (json.contains("num_expert_group")) {
    LOAD_ARG_OR(n_group, "num_expert_group", 1);
  } else if (json.contains("n_group")) {
    LOAD_ARG_OR(n_group, "n_group", 1);
  } else {
    SET_ARG(n_group, 8);
  }
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 62);
  if (json.contains("attn_type_list")) {
    const std::vector<int32_t> attn_type_list =
        json.value_or<std::vector<int32_t>>("attn_type_list",
                                            std::vector<int32_t>());
    if (!attn_type_list.empty() &&
        args->n_layers() != static_cast<int32_t>(attn_type_list.size())) {
      LOG(WARNING) << "MiniMax config mismatch: num_hidden_layers="
                   << args->n_layers()
                   << ", attn_type_list size=" << attn_type_list.size()
                   << ". Using attn_type_list size.";
      args->n_layers() = static_cast<int32_t>(attn_type_list.size());
    }
  }
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 8);
  LOAD_ARG_OR(output_router_logits, "output_router_logits", false);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(rope_theta, "rope_theta", 5000000.0f);
  LOAD_ARG_OR(scoring_func, "scoring_func", "sigmoid");
  if (json.contains("topk_group")) {
    LOAD_ARG_OR(topk_group, "topk_group", 1);
  } else {
    SET_ARG(topk_group, args->n_group());
  }
  LOAD_ARG_OR(routed_scaling_factor, "routed_scaling_factor", 1.0f);
  LOAD_ARG_OR(router_aux_loss_coef, "router_aux_loss_coef", 0.0f);
  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_ARG_OR(vocab_size, "vocab_size", 200064);
  LOAD_ARG_OR(mlp_only_layers, "mlp_only_layers", std::vector<int32_t>());

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
  SET_ARG(topk_method, "noaux_tc");
});

}  // namespace xllm

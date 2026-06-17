/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/layers/qwen3_5_decoder_layer.h"
#include "models/model_registry.h"
#include "qwen3_next.h"

namespace xllm {

class Qwen3_5ModelImpl : public Qwen3NextModelImpl {
 public:
  explicit Qwen3_5ModelImpl(const ModelContext& context)
      : Qwen3NextModelImpl(context, /*init_decoder_layers=*/false) {
    const int32_t n_layers = context.get_model_args().n_layers();
    for (int32_t layer_id = 0; layer_id < n_layers; ++layer_id) {
      add_decoder_layer(
          std::make_shared<layer::Qwen3_5DecoderLayerImpl>(context, layer_id));
    }
  }
};
TORCH_MODULE(Qwen3_5Model);

class Qwen3_5ForCausalLMImpl : public Qwen3NextForCausalLMImpl {
 public:
  explicit Qwen3_5ForCausalLMImpl(const ModelContext& context)
      : Qwen3NextForCausalLMImpl(context, /*init_model=*/false) {
    set_model_module(std::make_shared<Qwen3_5ModelImpl>(context));
  }

  torch::Tensor get_input_embeddings(torch::Tensor input_ids) {
    return get_word_embedding()(input_ids);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    Qwen3NextForCausalLMImpl::load_model(
        std::move(loader), "model.language_model.", "lm_head.");
  }

  void load_model(std::unique_ptr<ModelLoader> loader,
                  const std::string& model_prefix) {
    Qwen3NextForCausalLMImpl::load_model(
        std::move(loader), model_prefix, "lm_head.");
  }
};
TORCH_MODULE(Qwen3_5ForCausalLM);

#define LOAD_ARG_TEXT_OR_ROOT(arg_name, json_key, default_value) \
  LOAD_ARG_OR(arg_name, "text_config." json_key, default_value); \
  LOAD_ARG_OR(arg_name, json_key, args->arg_name())

#define LOAD_ARG_TEXT_OR_ROOT_CHAIN(arg_name, json_key, default_value) \
  LOAD_ARG_TEXT_OR_ROOT(arg_name, json_key, default_value)

#define LOAD_QWEN3_5_ROPE_ARG(arg_name, default_value)                       \
  LOAD_ARG_OR(arg_name, "text_config." #arg_name, default_value);            \
  LOAD_ARG_OR(arg_name, #arg_name, args->arg_name());                        \
  LOAD_ARG_OR(                                                               \
      arg_name, "text_config.rope_scaling." #arg_name, args->arg_name());    \
  LOAD_ARG_OR(arg_name, "rope_scaling." #arg_name, args->arg_name());        \
  LOAD_ARG_OR(                                                               \
      arg_name, "text_config.rope_parameters." #arg_name, args->arg_name()); \
  LOAD_ARG_OR(arg_name, "rope_parameters." #arg_name, args->arg_name())

#define LOAD_QWEN3_5_NEXT_COMPAT_ARGS(default_moe_intermediate_size,           \
                                      default_num_experts,                     \
                                      default_num_experts_per_tok,             \
                                      default_shared_expert_intermediate_size) \
  LOAD_ARG_TEXT_OR_ROOT(attention_bias, "attention_bias", false);              \
  LOAD_ARG_TEXT_OR_ROOT(attention_dropout, "attention_dropout", 0.0f);         \
  LOAD_ARG_TEXT_OR_ROOT(bos_token_id, "bos_token_id", 151643);                 \
  LOAD_ARG_TEXT_OR_ROOT(decoder_sparse_step, "decoder_sparse_step", 1);        \
  LOAD_ARG_TEXT_OR_ROOT(eos_token_id, "eos_token_id", 151645);                 \
  LOAD_ARG_TEXT_OR_ROOT(head_dim, "head_dim", 256);                            \
  LOAD_ARG_TEXT_OR_ROOT(hidden_act, "hidden_act", "silu");                     \
  LOAD_ARG_TEXT_OR_ROOT(hidden_size, "hidden_size", 2048);                     \
  LOAD_ARG_TEXT_OR_ROOT(initializer_range, "initializer_range", 0.02f);        \
  LOAD_ARG_TEXT_OR_ROOT(intermediate_size, "intermediate_size", 5120);         \
  LOAD_ARG_TEXT_OR_ROOT(                                                       \
      max_position_embeddings, "max_position_embeddings", 262144);             \
  LOAD_ARG_TEXT_OR_ROOT(max_window_layers, "max_window_layers", 28);           \
  LOAD_ARG_TEXT_OR_ROOT(moe_intermediate_size,                                 \
                        "moe_intermediate_size",                               \
                        default_moe_intermediate_size);                        \
  LOAD_ARG_TEXT_OR_ROOT(norm_topk_prob, "norm_topk_prob", true);               \
  LOAD_ARG_TEXT_OR_ROOT(n_heads, "num_attention_heads", 16);                   \
  LOAD_ARG_TEXT_OR_ROOT(num_experts, "num_experts", default_num_experts);      \
  LOAD_ARG_TEXT_OR_ROOT(num_experts_per_tok,                                   \
                        "num_experts_per_tok",                                 \
                        default_num_experts_per_tok);                          \
  LOAD_ARG_TEXT_OR_ROOT(n_layers, "num_hidden_layers", 48);                    \
  LOAD_ARG_OR(n_kv_heads, "text_config.num_key_value_heads", 2);               \
  LOAD_ARG_OR(                                                                 \
      n_kv_heads, "num_key_value_heads", args->n_kv_heads().value_or(2));      \
  LOAD_ARG_TEXT_OR_ROOT(output_router_logits, "output_router_logits", false);  \
  LOAD_ARG_TEXT_OR_ROOT(rms_norm_eps, "rms_norm_eps", 1e-6);                   \
  LOAD_QWEN3_5_ROPE_ARG(rope_theta, 10000000.0f);                              \
  LOAD_ARG_TEXT_OR_ROOT(router_aux_loss_coef, "router_aux_loss_coef", 0.001f); \
  LOAD_ARG_TEXT_OR_ROOT(use_sliding_window, "use_sliding_window", false);      \
  LOAD_ARG_TEXT_OR_ROOT(sliding_window, "sliding_window", 4096);               \
  LOAD_ARG_TEXT_OR_ROOT(tie_word_embeddings, "tie_word_embeddings", false);    \
  LOAD_ARG_TEXT_OR_ROOT(vocab_size, "vocab_size", 151936);                     \
  LOAD_ARG_TEXT_OR_ROOT(                                                       \
      mlp_only_layers, "mlp_only_layers", std::vector<int32_t>());             \
  LOAD_ARG_TEXT_OR_ROOT(attn_output_gate, "attn_output_gate", true);           \
  LOAD_ARG_TEXT_OR_ROOT(                                                       \
      full_attention_interval, "full_attention_interval", 4);                  \
  LOAD_ARG_TEXT_OR_ROOT(linear_conv_kernel_dim, "linear_conv_kernel_dim", 4);  \
  LOAD_ARG_TEXT_OR_ROOT(linear_key_head_dim, "linear_key_head_dim", 128);      \
  LOAD_ARG_TEXT_OR_ROOT(linear_num_key_heads, "linear_num_key_heads", 16);     \
  LOAD_ARG_TEXT_OR_ROOT(linear_num_value_heads, "linear_num_value_heads", 32); \
  LOAD_ARG_TEXT_OR_ROOT(linear_value_head_dim, "linear_value_head_dim", 128);  \
  LOAD_QWEN3_5_ROPE_ARG(partial_rotary_factor, 0.25f);                         \
  LOAD_ARG_OR(rope_scaling_mrope_section,                                      \
              "text_config.rope_scaling.mrope_section",                        \
              std::vector<int64_t>());                                         \
  LOAD_ARG_OR(rope_scaling_mrope_section,                                      \
              "text_config.rope_parameters.mrope_section",                     \
              args->rope_scaling_mrope_section());                             \
  LOAD_ARG_OR(rope_scaling_mrope_section,                                      \
              "rope_parameters.mrope_section",                                 \
              args->rope_scaling_mrope_section());                             \
  LOAD_ARG_OR(rope_scaling_mrope_interleaved,                                  \
              "text_config.rope_scaling.mrope_interleaved",                    \
              false);                                                          \
  LOAD_ARG_OR(rope_scaling_mrope_interleaved,                                  \
              "text_config.rope_parameters.mrope_interleaved",                 \
              args->rope_scaling_mrope_interleaved());                         \
  LOAD_ARG_OR(rope_scaling_mrope_interleaved,                                  \
              "rope_parameters.mrope_interleaved",                             \
              args->rope_scaling_mrope_interleaved());                         \
  LOAD_ARG_TEXT_OR_ROOT(shared_expert_intermediate_size,                       \
                        "shared_expert_intermediate_size",                     \
                        default_shared_expert_intermediate_size);              \
  LOAD_ARG_OR(                                                                 \
      num_nextn_predict_layers, "text_config.mtp_num_hidden_layers", 0);       \
  LOAD_ARG_OR(num_nextn_predict_layers,                                        \
              "mtp_num_hidden_layers",                                         \
              args->num_nextn_predict_layers());                               \
  LOAD_ARG_OR(num_nextn_predict_layers,                                        \
              "text_config.num_nextn_predict_layers",                          \
              args->num_nextn_predict_layers());                               \
  LOAD_ARG_OR(num_nextn_predict_layers,                                        \
              "num_nextn_predict_layers",                                      \
              args->num_nextn_predict_layers());                               \
  LOAD_ARG_OR(                                                                 \
      layer_types, "text_config.layer_types", std::vector<std::string>());     \
  LOAD_ARG_OR(layer_types, "layer_types", args->layer_types());                \
  LOAD_ARG_OR(                                                                 \
      layer_types, "text_config.layers_block_type", args->layer_types());      \
  LOAD_ARG_OR(layer_types, "layers_block_type", args->layer_types());          \
  LOAD_ARG_OR(                                                                 \
      n_routed_experts, "text_config.n_routed_experts", args->num_experts());  \
  LOAD_ARG_OR(n_routed_experts, "n_routed_experts", args->num_experts());      \
  SET_ARG(n_shared_experts,                                                    \
          args->shared_expert_intermediate_size() > 0 ? 1 : 0);                \
  SET_ARG(scoring_func, "softmax");                                            \
  SET_ARG(topk_method, "");                                                    \
  SET_ARG(n_group, -1);                                                        \
  SET_ARG(topk_group, 0);                                                      \
  SET_ARG(routed_scaling_factor, 1.0f);                                        \
  SET_ARG(stop_token_ids,                                                      \
          std::unordered_set<int32_t>({args->eos_token_id(), 248046}));        \
  LOAD_ARG_TEXT_OR_ROOT(mamba_ssm_dtype, "mamba_ssm_dtype", "float32")

#define LOAD_QWEN3_5_TEXT_TYPE_AND_DTYPE(default_model_type)    \
  SET_ARG(model_type, default_model_type);                      \
  LOAD_ARG_OR(dtype, "text_config.dtype", "bfloat16");          \
  LOAD_ARG_OR(dtype, "dtype", args->dtype());                   \
  LOAD_ARG_OR(dtype, "text_config.torch_dtype", args->dtype()); \
  LOAD_ARG_OR(dtype, "torch_dtype", args->dtype())

REGISTER_CAUSAL_MODEL(qwen3_5_text, Qwen3_5ForCausalLM);
REGISTER_MODEL_ARGS(qwen3_5_text, [&] {
  LOAD_QWEN3_5_TEXT_TYPE_AND_DTYPE("qwen3_5_text");
  LOAD_QWEN3_5_NEXT_COMPAT_ARGS(/*moe_intermediate_size=*/0,
                                /*num_experts=*/0,
                                /*num_experts_per_tok=*/0,
                                /*shared_expert_intermediate_size=*/0);
});

REGISTER_CAUSAL_MODEL(qwen3_5_moe_text, Qwen3_5ForCausalLM);
REGISTER_MODEL_ARGS(qwen3_5_moe_text, [&] {
  LOAD_QWEN3_5_TEXT_TYPE_AND_DTYPE("qwen3_5_moe_text");
  LOAD_QWEN3_5_NEXT_COMPAT_ARGS(/*moe_intermediate_size=*/512,
                                /*num_experts=*/512,
                                /*num_experts_per_tok=*/10,
                                /*shared_expert_intermediate_size=*/512);
});

#undef LOAD_QWEN3_5_TEXT_TYPE_AND_DTYPE
#undef LOAD_QWEN3_5_NEXT_COMPAT_ARGS
#undef LOAD_QWEN3_5_ROPE_ARG
#undef LOAD_ARG_TEXT_OR_ROOT_CHAIN
#undef LOAD_ARG_TEXT_OR_ROOT

}  // namespace xllm

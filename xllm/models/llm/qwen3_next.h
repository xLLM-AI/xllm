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

#include <string>
#include <unordered_set>
#include <vector>

#if defined(USE_MUSA)
#include "core/layers/musa/qwen3_next_decoder_layer_impl.h"
#else
#include "core/layers/npu_torch/qwen3_next_decoder_layer_impl.h"
#endif
#include "models/model_registry.h"
#include "qwen3_next_hybrid_base.h"

namespace xllm {

class Qwen3NextModelImpl : public Qwen3HybridModelImplBase {
 public:
  explicit Qwen3NextModelImpl(const ModelContext& context)
      : Qwen3NextModelImpl(context, /*init_decoder_layers=*/true) {}

 protected:
  explicit Qwen3NextModelImpl(const ModelContext& context,
                              bool init_decoder_layers)
      : Qwen3HybridModelImplBase(context) {
    if (init_decoder_layers) {
      const int32_t n_layers = context.get_model_args().n_layers();
      for (int32_t layer_id = 0; layer_id < n_layers; ++layer_id) {
        add_decoder_layer(std::make_shared<layer::Qwen3NextDecoderLayerImpl>(
            context, layer_id));
      }
    }
  }
};
TORCH_MODULE(Qwen3NextModel);

class Qwen3NextForCausalLMImpl : public Qwen3HybridForCausalLMImplBase {
 public:
  explicit Qwen3NextForCausalLMImpl(const ModelContext& context)
      : Qwen3NextForCausalLMImpl(context, /*init_model=*/true) {}

 protected:
  explicit Qwen3NextForCausalLMImpl(const ModelContext& context,
                                    bool init_model)
      : Qwen3HybridForCausalLMImplBase(context) {
    if (init_model) {
      set_model_module(std::make_shared<Qwen3NextModelImpl>(context));
    }
  }
};
TORCH_MODULE(Qwen3NextForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL(qwen3_next, Qwen3NextForCausalLM);

// register the model args
REGISTER_MODEL_ARGS(qwen3_next, [&] {
  LOAD_ARG_OR(model_type, "model_type", "qwen3_next");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(attention_bias, "attention_bias", false);
  LOAD_ARG_OR(attention_dropout, "attention_dropout", 0.0f);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 151643);
  LOAD_ARG_OR(decoder_sparse_step, "decoder_sparse_step", 1);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 151645);
  LOAD_ARG_OR(head_dim, "head_dim", 256);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "hidden_size", 2048);
  LOAD_ARG_OR(initializer_range, "initializer_range", 0.02f);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 5120);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 262144);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 28);
  LOAD_ARG_OR(moe_intermediate_size, "moe_intermediate_size", 512);
  LOAD_ARG_OR(norm_topk_prob, "norm_topk_prob", true);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 16);
  LOAD_ARG_OR(num_experts, "num_experts", 512);
  LOAD_ARG_OR(num_experts_per_tok, "num_experts_per_tok", 10);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 48);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 2);
  LOAD_ARG_OR(output_router_logits, "output_router_logits", false);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(rope_theta, "rope_theta", 10000000.0f);
  LOAD_ARG_OR(router_aux_loss_coef, "router_aux_loss_coef", 0.001f);
  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(sliding_window, "sliding_window", 4096);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_ARG_OR(vocab_size, "vocab_size", 151936);
  LOAD_ARG_OR(mlp_only_layers, "mlp_only_layers", std::vector<int>());

  // Additional parameters for Qwen3-Next architecture
  LOAD_ARG_OR(attn_output_gate, "attn_output_gate", true);
  LOAD_ARG_OR(full_attention_interval, "full_attention_interval", 4);
  LOAD_ARG_OR(linear_conv_kernel_dim, "linear_conv_kernel_dim", 4);
  LOAD_ARG_OR(linear_key_head_dim, "linear_key_head_dim", 128);
  LOAD_ARG_OR(linear_num_key_heads, "linear_num_key_heads", 16);
  LOAD_ARG_OR(linear_num_value_heads, "linear_num_value_heads", 32);
  LOAD_ARG_OR(linear_value_head_dim, "linear_value_head_dim", 128);
  LOAD_ARG_OR(partial_rotary_factor, "partial_rotary_factor", 0.25f);
  LOAD_ARG_OR(
      shared_expert_intermediate_size, "shared_expert_intermediate_size", 512);
  LOAD_ARG_OR(layer_types, "layer_types", std::vector<std::string>());

  // MoE compatibility with fused_moe implementation.
  LOAD_ARG_OR(n_routed_experts, "n_routed_experts", args->num_experts());
  SET_ARG(n_shared_experts,
          args->shared_expert_intermediate_size() > 0 ? 1 : 0);
  SET_ARG(scoring_func, "softmax");
  SET_ARG(topk_method, "");
  SET_ARG(n_group, -1);
  SET_ARG(topk_group, 0);
  SET_ARG(routed_scaling_factor, 1.0);

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});

}  // namespace xllm

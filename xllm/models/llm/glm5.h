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

#include "deepseek_v32.h"

namespace xllm {

class Glm5ModelImpl : public DeepseekV32ModelImpl {
 public:
  explicit Glm5ModelImpl(const ModelContext& context)
      : DeepseekV32ModelImpl(context) {}

 protected:
  Glm5ModelImpl(const ModelContext& context,
                const DecoderLayerFactory& decoder_layer_factory)
      : DeepseekV32ModelImpl(context, decoder_layer_factory) {}
};
TORCH_MODULE(Glm5Model);

class Glm5ForCausalLMImpl : public LlmForCausalLMImplBase<Glm5Model> {
 public:
  Glm5ForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<Glm5Model>(context) {}

  void load_model(
      std::unique_ptr<ModelLoader> loader,
      std::string prefix = "model." /*llm model weight prefix*/) override {
    LlmForCausalLMImplBase<Glm5Model>::load_model(std::move(loader), prefix);
    model_->verify_loaded_weights();
  }
};
TORCH_MODULE(Glm5ForCausalLM);

// GLM5.0/5.1 and GLM5.2 share model_type "glm_moe_dsa". Register it once in
// glm52.h; Glm52ForCausalLM falls back to the legacy GLM5 behavior when the
// resolved DSA top-k share plan contains no shared layers.

// register the model args
// example config:
// https://huggingface.co/zai-org/GLM-5/blob/main/config.json
REGISTER_MODEL_ARGS(
    glm_moe_dsa,
    ([&] {
      LOAD_ARG_OR(model_type, "model_type", "glm_moe_dsa");
      LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");
      LOAD_ARG_OR(vocab_size, "vocab_size", 154880);
      LOAD_ARG_OR(hidden_size, "hidden_size", 6144);
      LOAD_ARG_OR(n_layers, "num_hidden_layers", 78);
      LOAD_ARG_OR(n_heads, "num_attention_heads", 64);
      LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 64);
      LOAD_ARG_OR(intermediate_size, "intermediate_size", 12288);
      LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 202752);
      LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-5);
      LOAD_ARG_OR_FUNC(eos_token_id_vec, "eos_token_id", [&] {
        return std::vector<int>{154820, 154827, 154829};
      });
      LOAD_ARG_OR(bos_token_id, "bos_token_id", 0);
      LOAD_ARG_OR(rope_theta, "rope_parameters.rope_theta", 1000000.0f);
      LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
      LOAD_ARG_OR(sliding_window, "sliding_window", 4096);
      LOAD_ARG_OR(max_window_layers, "max_window_layers", 78);

      // MoE parameters
      LOAD_ARG_OR(first_k_dense_replace, "first_k_dense_replace", 3);
      LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
      LOAD_ARG_OR(moe_layer_freq, "moe_layer_freq", 1);
      LOAD_ARG_OR(topk_method, "topk_method", "noaux_tc");
      LOAD_ARG_OR(n_routed_experts, "n_routed_experts", 256);
      LOAD_ARG_OR(n_shared_experts, "n_shared_experts", 1);
      LOAD_ARG_OR(num_experts_per_tok, "num_experts_per_tok", 8);
      LOAD_ARG_OR(moe_intermediate_size, "moe_intermediate_size", 2048);
      LOAD_ARG_OR(routed_scaling_factor, "routed_scaling_factor", 2.5f);
      LOAD_ARG_OR(norm_topk_prob, "norm_topk_prob", true);
      LOAD_ARG_OR(n_group, "n_group", 1);
      LOAD_ARG_OR(topk_group, "topk_group", 1);
      LOAD_ARG_OR(scoring_func, "scoring_func", "sigmoid");

      // Attention & Indexer related parameters
      LOAD_ARG_OR(qk_nope_head_dim, "qk_nope_head_dim", 192);
      LOAD_ARG_OR(qk_rope_head_dim, "qk_rope_head_dim", 64);
      LOAD_ARG_OR(v_head_dim, "v_head_dim", 256);
      LOAD_ARG_OR(q_lora_rank, "q_lora_rank", 2048);
      LOAD_ARG_OR(kv_lora_rank, "kv_lora_rank", 512);
      LOAD_ARG_OR(num_nextn_predict_layers, "num_nextn_predict_layers", 1);
      LOAD_ARG_OR(index_head_dim, "index_head_dim", 128);
      LOAD_ARG_OR(index_n_heads, "index_n_heads", 32);
      LOAD_ARG_OR(index_topk, "index_topk", 2048);
      LOAD_ARG_OR(index_topk_freq, "index_topk_freq", 1);
      LOAD_ARG_OR(index_topk_pattern, "index_topk_pattern", "");
      LOAD_ARG_OR(index_skip_topk_offset, "index_skip_topk_offset", 0);
      LOAD_ARG_OR(index_share_for_mtp_iteration,
                  "index_share_for_mtp_iteration",
                  false);

      // Computed parameters
      // the original head_dim in glm5 config seem useless
      // here we use args->qk_nope_head_dim() + args->qk_rope_head_dim()
      SET_ARG(head_dim, 256);
      LOAD_ARG_OR_FUNC(
          rotary_dim, "rotary_dim", [&] { return args->qk_rope_head_dim(); });

      // GLM-5 uses default rope_type, no deepseek_yarn scaling
      SET_ARG(rope_scaling_rope_type, "default");
      LOAD_ARG_OR(indexer_rope_interleave, "indexer_rope_interleave", true);
      LOAD_ARG_OR(rope_theta, "rope_parameters.rope_theta", 1000000.0f);

      SET_ARG(stop_token_ids,
              std::unordered_set<int32_t>(args->eos_token_id_vec().begin(),
                                          args->eos_token_id_vec().end()));
    }));

}  // namespace xllm

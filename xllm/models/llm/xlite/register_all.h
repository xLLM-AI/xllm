/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Register all xlite-backed models as "<ModelType>_xlite".

#pragma once

#include <string>
#include <unordered_set>

#include "core/layers/xlite/xlite_register_macros.h"
#include "models/llm/xlite/adapters/deepseek_v3_adapter.h"
#include "models/llm/xlite/adapters/glm4_moe_adapter.h"
#include "models/llm/xlite/adapters/glm5_moe_adapter.h"
#include "models/llm/xlite/adapters/qwen3_adapter.h"
#include "models/llm/xlite/adapters/qwen3_moe_adapter.h"

namespace xllm::xlite {

// Qwen3 dense (MHA). xlite-specific fields appended for XliteConfigBuilder.
XLITE_REGISTER_MODEL(qwen3, Qwen3Adapter, [&] {
  LOAD_ARG_OR(model_type, "model_type", "qwen3");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 152064);
  LOAD_ARG_OR(hidden_size, "hidden_size", 3584);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 28);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 28);
  LOAD_ARG(n_kv_heads, "num_key_value_heads");
  // head_dim required by KV cache estimation; derive if absent.
  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 18944);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 32768);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 151643);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);

  // qwen3 < 7B: tie_word_embeddings = true
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);

  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 28);

  // Eagle3: layer ids to capture, e.g. "layers_to_capture": [2, 14, 25].
  LOAD_ARG_OR(layers_to_capture, "layers_to_capture", std::vector<int32_t>{});

  // xlite-specific fields for XliteConfigBuilder.
  LOAD_ARG_OR(use_qk_norm, "use_qk_norm", true);
  LOAD_ARG_OR(qkv_bias, "qkv_bias", false);
  SET_ARG(enable_mla, false);
  SET_ARG(use_moe, false);
  SET_ARG(first_k_dense_replace, 0);

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});

// DeepSeek-V3/R1 (MLA + sigmoid MoE + shared expert). enable_mla/use_moe via
// SET_ARG.
XLITE_REGISTER_MODEL(deepseek_v3, DeepseekV3Adapter, [&] {
  LOAD_ARG_OR(model_type, "model_type", "deepseek_v3");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 129280);
  LOAD_ARG_OR(hidden_size, "hidden_size", 7168);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 61);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 128);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 128);  // MLA: = n_heads
  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->qk_nope_head_dim() +
           args->qk_rope_head_dim();  // MLA: nope+rope
  });
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 18432);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 163840);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 1);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 0);
  LOAD_ARG_OR(rope_theta, "rope_theta", 10000.0f);
  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(sliding_window, "sliding_window", 4096);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 61);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);

  // MLA fields
  LOAD_ARG_OR(qk_nope_head_dim, "qk_nope_head_dim", 128);
  LOAD_ARG_OR(qk_rope_head_dim, "qk_rope_head_dim", 64);
  LOAD_ARG_OR(v_head_dim, "v_head_dim", 128);
  LOAD_ARG_OR(q_lora_rank, "q_lora_rank", 1536);
  LOAD_ARG_OR(kv_lora_rank, "kv_lora_rank", 512);

  // MoE fields
  LOAD_ARG_OR(first_k_dense_replace, "first_k_dense_replace", 3);
  LOAD_ARG_OR(moe_intermediate_size, "moe_intermediate_size", 2048);
  LOAD_ARG_OR(num_experts, "n_routed_experts", 256);
  LOAD_ARG_OR(n_shared_experts, "n_shared_experts", 1);
  LOAD_ARG_OR(num_experts_per_tok, "num_experts_per_tok", 8);
  LOAD_ARG_OR(norm_topk_prob, "norm_topk_prob", true);
  LOAD_ARG_OR(routed_scaling_factor, "routed_scaling_factor", 2.5f);
  LOAD_ARG_OR(n_group, "n_group", 8);
  LOAD_ARG_OR(topk_group, "topk_group", 4);
  LOAD_ARG_OR(scoring_func, "scoring_func", "sigmoid");

  // xlite-specific fields (SET_ARG)
  SET_ARG(enable_mla, true);
  SET_ARG(use_moe, true);
  LOAD_ARG_OR(
      use_qk_norm, "use_qk_norm", false);  // MLA uses mlaQNorm, not MHA qkNorm
  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));

  // rope_scaling (deepseek_yarn): YaRN freq correction + mscale in
  // XliteFreqsCis.
  SET_ARG(rope_scaling_rope_type, "deepseek_yarn");
  LOAD_ARG_OR(rope_scaling_factor, "rope_scaling.factor", 40.0f);
  LOAD_ARG_OR(rope_scaling_beta_fast, "rope_scaling.beta_fast", 32);
  LOAD_ARG_OR(rope_scaling_beta_slow, "rope_scaling.beta_slow", 1);
  LOAD_ARG_OR(rope_scaling_original_max_position_embeddings,
              "rope_scaling.original_max_position_embeddings",
              4096);
  LOAD_ARG_OR(rope_scaling_mscale, "rope_scaling.mscale", 1.0f);
  LOAD_ARG_OR(rope_scaling_mscale_all_dim, "rope_scaling.mscale_all_dim", 1.0f);
});

// GLM-5/5.1 (MLA + DSA indexer + sigmoid MoE + shared, default rope).
// default rope: no rope_scaling_* -> FromGlm5 skips YaRN/mscale.
XLITE_REGISTER_MODEL(glm_moe_dsa, Glm5MoeAdapter, [&] {
  LOAD_ARG_OR(model_type, "model_type", "glm_moe_dsa");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 154880);
  LOAD_ARG_OR(hidden_size, "hidden_size", 6144);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 78);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 64);
  LOAD_ARG_OR(n_kv_heads,
              "num_key_value_heads",
              64);  // MLA: c.nKvHeads=1 (MQA); framework uses this
  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->qk_nope_head_dim() +
           args->qk_rope_head_dim();  // MLA: nope+rope
  });
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 12288);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 202752);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-5);
  // eos_token_id is a list [154820,154827,154829]; read from config with the
  // list as fallback (same as glm5.h).
  LOAD_ARG_OR_FUNC(eos_token_id_vec, "eos_token_id", [&] {
    return std::vector<int32_t>({154820, 154827, 154829});
  });
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 0);
  // rope_theta under rope_parameters (GLM5 has no top-level rope_theta).
  LOAD_ARG_OR(rope_theta, "rope_parameters.rope_theta", 1000000.0f);

  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(sliding_window, "sliding_window", 4096);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 61);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);

  // MLA fields (GLM5: nope=192/rope=64/v=256/q_lora=2048/kv_lora=512)
  LOAD_ARG_OR(qk_nope_head_dim, "qk_nope_head_dim", 192);
  LOAD_ARG_OR(qk_rope_head_dim, "qk_rope_head_dim", 64);
  LOAD_ARG_OR(v_head_dim, "v_head_dim", 256);
  LOAD_ARG_OR(q_lora_rank, "q_lora_rank", 2048);
  LOAD_ARG_OR(kv_lora_rank, "kv_lora_rank", 512);

  // DSA indexer fields
  LOAD_ARG_OR(index_head_dim, "index_head_dim", 128);
  LOAD_ARG_OR(index_n_heads, "index_n_heads", 32);
  LOAD_ARG_OR(index_topk, "index_topk", 2048);
  LOAD_ARG_OR(indexer_rope_interleave, "indexer_rope_interleave", true);
  // DSA top-k sharing (GLM-5.2): shared layers skip indexer, reuse prev full
  // layer's topkIndices. freq>1 (periodic) or pattern (F/S per-layer). Empty +
  // freq<=1 => all layers run indexer (GLM-5.1).
  LOAD_ARG_OR(index_topk_freq, "index_topk_freq", 1);
  LOAD_ARG_OR(index_topk_pattern, "index_topk_pattern", "");
  LOAD_ARG_OR(index_skip_topk_offset, "index_skip_topk_offset", 0);

  // MoE fields (GLM5: 3 dense / 256 expert / 8 act / 1 shared / group n=1
  // topk=1)
  LOAD_ARG_OR(first_k_dense_replace, "first_k_dense_replace", 3);
  LOAD_ARG_OR(moe_intermediate_size, "moe_intermediate_size", 2048);
  LOAD_ARG_OR(num_experts, "n_routed_experts", 256);
  LOAD_ARG_OR(n_shared_experts, "n_shared_experts", 1);
  LOAD_ARG_OR(num_experts_per_tok, "num_experts_per_tok", 8);
  LOAD_ARG_OR(norm_topk_prob, "norm_topk_prob", true);
  LOAD_ARG_OR(routed_scaling_factor, "routed_scaling_factor", 2.5f);
  LOAD_ARG_OR(n_group, "n_group", 1);
  LOAD_ARG_OR(topk_group, "topk_group", 1);
  LOAD_ARG_OR(scoring_func, "scoring_func", "sigmoid");

  // xlite-specific fields (SET_ARG)
  SET_ARG(enable_mla, true);  // FromGlm5 sets attnType=DSA (MLA+indexer)
  SET_ARG(use_moe, true);
  LOAD_ARG_OR(
      use_qk_norm, "use_qk_norm", false);  // MLA uses mlaQNorm, not MHA qkNorm
  SET_ARG(stop_token_ids,
          std::unordered_set<int32_t>(args->eos_token_id_vec().begin(),
                                      args->eos_token_id_vec().end()));
});

// Qwen3-MoE. xlite-specific fields appended for
// XliteConfigBuilder::FromQwen3Moe.
XLITE_REGISTER_MODEL(qwen3_moe, Qwen3MoeAdapter, [&] {
  LOAD_ARG_OR(model_type, "model_type", "qwen3_moe");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 151936);
  LOAD_ARG_OR(hidden_size, "hidden_size", 2048);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 48);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 32);
  LOAD_ARG(n_kv_heads, "num_key_value_heads");
  // head_dim required by KV cache estimation; derive if absent.
  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 6144);
  // MoE fields (consumed by FromQwen3Moe).
  LOAD_ARG_OR(moe_intermediate_size, "moe_intermediate_size", 768);
  LOAD_ARG_OR(num_experts, "num_experts", 128);
  LOAD_ARG_OR(num_experts_per_tok, "num_experts_per_tok", 8);
  LOAD_ARG_OR(norm_topk_prob, "norm_topk_prob", true);
  LOAD_ARG_OR(decoder_sparse_step, "decoder_sparse_step", 1);
  // first_k_dense_replace: drives XModelConfig.nDenseLayers.
  LOAD_ARG_OR(first_k_dense_replace, "first_k_dense_replace", 0);

  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 40960);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 151645);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);

  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 48);

  // Framework-parity fields (harmless if xlite ignores).
  LOAD_ARG_OR(output_router_logits, "output_router_logits", false);
  LOAD_ARG_OR(router_aux_loss_coef, "router_aux_loss_coef", 0.001f);
  LOAD_ARG_OR(mlp_only_layers, "mlp_only_layers", std::vector<int>());

  // Eagle3: layer ids to capture, e.g. "layers_to_capture": [2, 14, 25].
  LOAD_ARG_OR(layers_to_capture, "layers_to_capture", std::vector<int32_t>{});

  // xlite-specific fields for XliteConfigBuilder.
  LOAD_ARG_OR(use_qk_norm, "use_qk_norm", true);
  LOAD_ARG_OR(qkv_bias, "qkv_bias", false);
  SET_ARG(enable_mla, false);
  SET_ARG(use_moe, true);

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});

// GLM-4 MoE (MHA + partial rotary + sigmoid MoE + shared expert, W8A8).
// Resolved name: "glm4_moe_xlite". Near Qwen3-MoE but has
// shared/partial_rotary/sigmoid. GLM-4.7 = glm4_moe (GQA MHA + qk_norm +
// qkv_bias + partial rotary + sigmoid MoE + shared). attention_bias=true ->
// qkv_bias. partial_rotary_factor=0.5 -> ropeHeadDim=64 (front 64 dims).
// eos_token_id is a list [151329,151336,151338] -> eos_token_id_vec (same as
// glm5; multi-element vector default brace list breaks macro, use SET_ARG
// parentheses).
XLITE_REGISTER_MODEL(glm4_moe, Glm4MoeAdapter, [&] {
  LOAD_ARG_OR(model_type, "model_type", "glm4_moe");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 151552);
  LOAD_ARG_OR(hidden_size, "hidden_size", 5120);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 92);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 96);
  LOAD_ARG(n_kv_heads, "num_key_value_heads");
  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });
  LOAD_ARG_OR(intermediate_size,
              "intermediate_size",
              12288);  // dense FFN (first 3 layers)
  // MoE fields (consumed by FromGlm4Moe).
  LOAD_ARG_OR(moe_intermediate_size, "moe_intermediate_size", 1536);
  LOAD_ARG_OR(first_k_dense_replace, "first_k_dense_replace", 3);
  LOAD_ARG_OR(
      num_experts, "n_routed_experts", 160);  // GLM config: n_routed_experts
  LOAD_ARG_OR(n_shared_experts, "n_shared_experts", 1);
  LOAD_ARG_OR(num_experts_per_tok, "num_experts_per_tok", 8);
  LOAD_ARG_OR(norm_topk_prob, "norm_topk_prob", true);
  LOAD_ARG_OR(routed_scaling_factor, "routed_scaling_factor", 2.5f);
  LOAD_ARG_OR(n_group, "n_group", 1);
  LOAD_ARG_OR(topk_group, "topk_group", 1);
  LOAD_ARG_OR(partial_rotary_factor, "partial_rotary_factor", 0.5f);

  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 202752);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-5);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);

  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 92);

  // GLM-4.7: qk_norm=true, attention_bias=true (qkv_bias). config field name is
  // attention_bias.
  LOAD_ARG_OR(use_qk_norm, "use_qk_norm", true);
  LOAD_ARG_OR(qkv_bias, "attention_bias", false);

  // eos_token_id is a list [151329,151336,151338] (generation_config); read
  // from config with the list as fallback (same as glm5.h).
  LOAD_ARG_OR_FUNC(eos_token_id_vec, "eos_token_id", [&] {
    return std::vector<int32_t>({151329, 151336, 151338});
  });
  SET_ARG(stop_token_ids,
          std::unordered_set<int32_t>(args->eos_token_id_vec().begin(),
                                      args->eos_token_id_vec().end()));

  // xlite-specific fields (ATB does not load; SET_ARG). enable_mla=false (GQA
  // MHA, not MLA).
  SET_ARG(enable_mla, false);
  SET_ARG(use_moe, true);
});

// Returns true if the model_type has an xlite implementation.
inline bool is_xlite_model_type(const std::string& model_type) {
  static const std::unordered_set<std::string> kXliteModelTypes = {
      "qwen3", "qwen3_moe", "deepseek_v3", "glm4_moe", "glm_moe_dsa"};
  return kXliteModelTypes.count(model_type) > 0;
}

}  // namespace xllm::xlite

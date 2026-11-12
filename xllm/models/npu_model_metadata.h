/* Copyright 2026 The xLLM Authors.

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

#pragma once

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/framework/multimodal/mposition.h"
#include "models/model_registry.h"
#include "processors/glm4v_image_processor.h"
#include "processors/glm4v_prompt_processor.h"
#include "processors/glm4v_video_processor.h"
#include "processors/kimi25_image_processor.h"
#include "processors/minicpmv_image_processor.h"
#include "processors/minicpmv_prompt_processor.h"
#include "processors/mistral3_image_processor.h"
#include "processors/mistral3_prompt_processor.h"
#include "processors/multimodal_processor_pipeline.h"
#include "processors/qwen2_vl_image_processor.h"
#include "processors/qwen2_vl_prompt_processor.h"
#include "processors/qwen2_vl_video_processor.h"
#include "processors/qwen3_vl_prompt_processor.h"
#include "processors/qwen3_vl_video_processor.h"

namespace xllm::npu::model {

// Backend discovery remains available for Python model implementations.
REGISTER_MODEL_BACKEND(DFlashDraftModel, "llm");
REGISTER_MODEL_BACKEND(DSparkDraftModel, "llm");
REGISTER_MODEL_BACKEND(deepseek_v2, "llm");
REGISTER_MODEL_BACKEND(deepseek_v3, "llm");
REGISTER_MODEL_BACKEND(deepseek_v32, "llm");
REGISTER_MODEL_BACKEND(deepseek_v32_mtp, "llm");
REGISTER_MODEL_BACKEND(deepseek_v3_mtp, "llm");
REGISTER_MODEL_BACKEND(glm4, "llm");
REGISTER_MODEL_BACKEND(glm4_moe, "llm");
REGISTER_MODEL_BACKEND(glm4_moe_lite, "llm");
REGISTER_MODEL_BACKEND(glm4_moe_mtp, "llm");
REGISTER_MODEL_BACKEND(glm4v, "vlm");
REGISTER_MODEL_BACKEND(glm4v_moe, "vlm");
REGISTER_MODEL_BACKEND(glm_moe_dsa, "llm");
REGISTER_MODEL_BACKEND(glm_moe_dsa_mtp, "llm");
REGISTER_MODEL_BACKEND(joyai_llm_flash, "llm");
REGISTER_MODEL_BACKEND(kimi_k2, "llm");
REGISTER_MODEL_BACKEND(kimi_k25, "vlm");
REGISTER_MODEL_BACKEND(llama, "llm");
REGISTER_MODEL_BACKEND(llama3, "llm");
REGISTER_MODEL_BACKEND(minicpmv, "vlm");
REGISTER_MODEL_BACKEND(mistral, "llm");
REGISTER_MODEL_BACKEND(mistral3, "vlm");
REGISTER_MODEL_BACKEND(oxygenvlm, "vlm");
REGISTER_MODEL_BACKEND(oxygenvlm_text, "llm");
REGISTER_MODEL_BACKEND(qwen2_vl, "vlm");
REGISTER_MODEL_BACKEND(qwen3_eagle3, "llm");
REGISTER_MODEL_BACKEND(qwen3_vl_moe, "vlm");

REGISTER_MODEL_ARGS(deepseek_v3_mtp, [&] {
  LOAD_ARG_OR(model_type, "model_type", "deepseek_v3_mtp");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 129280);
  LOAD_ARG_OR(hidden_size, "hidden_size", 7168);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 61);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 128);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 128);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 18432);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 163840);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 1);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 0);
  LOAD_ARG_OR(rope_theta, "rope_theta", 10000.0f);
  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(sliding_window, "sliding_window", 4096);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 61);

  LOAD_ARG_OR(first_k_dense_replace, "first_k_dense_replace", 0);
  LOAD_ARG_OR(moe_layer_freq, "moe_layer_freq", 1);
  LOAD_ARG_OR(topk_method, "topk_method", "noaux_tc");
  LOAD_ARG_OR(n_routed_experts, "n_routed_experts", 256);
  LOAD_ARG_OR(n_shared_experts, "n_shared_experts", 1);
  LOAD_ARG_OR(num_experts_per_tok, "num_experts_per_tok", 8);
  LOAD_ARG_OR(moe_intermediate_size, "moe_intermediate_size", 2048);
  LOAD_ARG_OR(routed_scaling_factor, "routed_scaling_factor", 2.5f);
  LOAD_ARG_OR(norm_topk_prob, "norm_topk_prob", true);
  LOAD_ARG_OR(n_group, "n_group", 8);
  LOAD_ARG_OR(topk_group, "topk_group", 4);
  LOAD_ARG_OR(qk_nope_head_dim, "qk_nope_head_dim", 128);
  LOAD_ARG_OR(qk_rope_head_dim, "qk_rope_head_dim", 64);
  LOAD_ARG_OR(v_head_dim, "v_head_dim", 128);
  LOAD_ARG_OR(q_lora_rank, "q_lora_rank", 1536);
  LOAD_ARG_OR(kv_lora_rank, "kv_lora_rank", 512);

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return 256;  // args->qk_nope_head_dim() + args->qk_rope_head_dim();
  });
  LOAD_ARG_OR_FUNC(
      rotary_dim, "rotary_dim", [&] { return args->qk_rope_head_dim(); });

  SET_ARG(rope_scaling_rope_type, "deepseek_yarn");
  LOAD_ARG(rope_scaling_beta_fast, "rope_scaling.beta_fast");
  LOAD_ARG(rope_scaling_beta_slow, "rope_scaling.beta_slow");
  LOAD_ARG(rope_scaling_factor, "rope_scaling.factor");
  LOAD_ARG_OR(
      rope_extrapolation_factor, "rope_scaling.extrapolation_factor", 1.0f);
  LOAD_ARG(rope_scaling_mscale, "rope_scaling.mscale");
  LOAD_ARG(rope_scaling_mscale_all_dim, "rope_scaling.mscale_all_dim");
  LOAD_ARG(rope_scaling_original_max_position_embeddings,
           "rope_scaling.original_max_position_embeddings");
  LOAD_ARG_OR(rope_scaling_attn_factor, "rope_scaling.attn_factor", 1.0f);

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({1}));
});

REGISTER_MODEL_ARGS(deepseek_v2, [&] {
  LOAD_ARG_OR(model_type, "model_type", "deepseek_v2");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 102400);
  LOAD_ARG_OR(hidden_size, "hidden_size", 2048);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 27);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 16);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 16);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 10944);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 163840);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 100001);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 100000);
  LOAD_ARG_OR(rope_theta, "rope_theta", 10000.0f);
  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(sliding_window, "sliding_window", 4096);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 27);

  LOAD_ARG_OR(first_k_dense_replace, "first_k_dense_replace", 1);
  LOAD_ARG_OR(moe_layer_freq, "moe_layer_freq", 1);
  LOAD_ARG_OR(topk_method, "topk_method", "greedy");
  LOAD_ARG_OR(n_routed_experts, "n_routed_experts", 64);
  LOAD_ARG_OR(n_shared_experts, "n_shared_experts", 2);
  LOAD_ARG_OR(num_experts_per_tok, "num_experts_per_tok", 6);
  LOAD_ARG_OR(moe_intermediate_size, "moe_intermediate_size", 1408);
  LOAD_ARG_OR(routed_scaling_factor, "routed_scaling_factor", 1.0f);
  LOAD_ARG_OR(norm_topk_prob, "norm_topk_prob", false);
  LOAD_ARG_OR(n_group, "n_group", 1);
  LOAD_ARG_OR(topk_group, "topk_group", 1);
  LOAD_ARG_OR(qk_nope_head_dim, "qk_nope_head_dim", 128);
  LOAD_ARG_OR(qk_rope_head_dim, "qk_rope_head_dim", 64);
  LOAD_ARG_OR(v_head_dim, "v_head_dim", 128);
  LOAD_ARG_OR(q_lora_rank, "q_lora_rank", 0);
  LOAD_ARG_OR(kv_lora_rank, "kv_lora_rank", 512);

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return 256;  // args->qk_nope_head_dim() + args->qk_rope_head_dim();
  });
  LOAD_ARG_OR_FUNC(
      rotary_dim, "rotary_dim", [&] { return args->qk_rope_head_dim(); });

  SET_ARG(rope_scaling_rope_type, "deepseek_yarn");
  LOAD_ARG(rope_scaling_beta_fast, "rope_scaling.beta_fast");
  LOAD_ARG(rope_scaling_beta_slow, "rope_scaling.beta_slow");
  LOAD_ARG(rope_scaling_factor, "rope_scaling.factor");
  LOAD_ARG_OR(
      rope_extrapolation_factor, "rope_scaling.extrapolation_factor", 1.0f);
  LOAD_ARG(rope_scaling_mscale, "rope_scaling.mscale");
  LOAD_ARG(rope_scaling_mscale_all_dim, "rope_scaling.mscale_all_dim");
  LOAD_ARG(rope_scaling_original_max_position_embeddings,
           "rope_scaling.original_max_position_embeddings");
  LOAD_ARG_OR(rope_scaling_attn_factor, "rope_scaling.attn_factor", 1.0f);

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({100001}));
});

REGISTER_MODEL_ARGS(deepseek_v3, [&] {
  LOAD_ARG_OR(model_type, "model_type", "deepseek_v3");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 129280);
  LOAD_ARG_OR(hidden_size, "hidden_size", 7168);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 61);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 128);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 128);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 18432);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 163840);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 1);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 0);
  LOAD_ARG_OR(rope_theta, "rope_theta", 10000.0f);
  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(sliding_window, "sliding_window", 4096);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 61);

  LOAD_ARG_OR(first_k_dense_replace, "first_k_dense_replace", 3);
  LOAD_ARG_OR(moe_layer_freq, "moe_layer_freq", 1);
  LOAD_ARG_OR(topk_method, "topk_method", "noaux_tc");
  LOAD_ARG_OR(n_routed_experts, "n_routed_experts", 256);
  LOAD_ARG_OR(n_shared_experts, "n_shared_experts", 1);
  LOAD_ARG_OR(num_experts_per_tok, "num_experts_per_tok", 8);
  LOAD_ARG_OR(moe_intermediate_size, "moe_intermediate_size", 2048);
  LOAD_ARG_OR(routed_scaling_factor, "routed_scaling_factor", 2.5f);
  LOAD_ARG_OR(norm_topk_prob, "norm_topk_prob", true);
  LOAD_ARG_OR(n_group, "n_group", 8);
  LOAD_ARG_OR(topk_group, "topk_group", 4);
  LOAD_ARG_OR(qk_nope_head_dim, "qk_nope_head_dim", 128);
  LOAD_ARG_OR(qk_rope_head_dim, "qk_rope_head_dim", 64);
  LOAD_ARG_OR(v_head_dim, "v_head_dim", 128);
  LOAD_ARG_OR(q_lora_rank, "q_lora_rank", 1536);
  LOAD_ARG_OR(kv_lora_rank, "kv_lora_rank", 512);

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return 256;  // args->qk_nope_head_dim() + args->qk_rope_head_dim();
  });
  LOAD_ARG_OR_FUNC(
      rotary_dim, "rotary_dim", [&] { return args->qk_rope_head_dim(); });

  SET_ARG(rope_scaling_rope_type, "deepseek_yarn");
  LOAD_ARG(rope_scaling_beta_fast, "rope_scaling.beta_fast");
  LOAD_ARG(rope_scaling_beta_slow, "rope_scaling.beta_slow");
  LOAD_ARG(rope_scaling_factor, "rope_scaling.factor");
  LOAD_ARG_OR(
      rope_extrapolation_factor, "rope_scaling.extrapolation_factor", 1.0f);
  LOAD_ARG(rope_scaling_mscale, "rope_scaling.mscale");
  LOAD_ARG(rope_scaling_mscale_all_dim, "rope_scaling.mscale_all_dim");
  LOAD_ARG(rope_scaling_original_max_position_embeddings,
           "rope_scaling.original_max_position_embeddings");
  LOAD_ARG_OR(rope_scaling_attn_factor, "rope_scaling.attn_factor", 1.0f);

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({1}));
});

REGISTER_MODEL_ARGS(deepseek_v32, [&] {
  LOAD_ARG_OR(model_type, "model_type", "deepseek_v32");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 129280);
  LOAD_ARG_OR(hidden_size, "hidden_size", 7168);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 61);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 128);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 128);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 18432);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 163840);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 1);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 0);
  LOAD_ARG_OR(rope_theta, "rope_theta", 10000.0f);
  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(sliding_window, "sliding_window", 4096);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 61);

  LOAD_ARG_OR(first_k_dense_replace, "first_k_dense_replace", 3);
  LOAD_ARG_OR(moe_layer_freq, "moe_layer_freq", 1);
  LOAD_ARG_OR(topk_method, "topk_method", "noaux_tc");
  LOAD_ARG_OR(n_routed_experts, "n_routed_experts", 256);
  LOAD_ARG_OR(n_shared_experts, "n_shared_experts", 1);
  LOAD_ARG_OR(num_experts_per_tok, "num_experts_per_tok", 8);
  LOAD_ARG_OR(moe_intermediate_size, "moe_intermediate_size", 2048);
  LOAD_ARG_OR(routed_scaling_factor, "routed_scaling_factor", 2.5f);
  LOAD_ARG_OR(norm_topk_prob, "norm_topk_prob", true);
  LOAD_ARG_OR(n_group, "n_group", 8);
  LOAD_ARG_OR(topk_group, "topk_group", 4);
  LOAD_ARG_OR(qk_nope_head_dim, "qk_nope_head_dim", 128);
  LOAD_ARG_OR(qk_rope_head_dim, "qk_rope_head_dim", 64);
  LOAD_ARG_OR(v_head_dim, "v_head_dim", 128);
  LOAD_ARG_OR(q_lora_rank, "q_lora_rank", 1536);
  LOAD_ARG_OR(kv_lora_rank, "kv_lora_rank", 512);
  LOAD_ARG_OR(index_head_dim, "index_head_dim", 128);
  LOAD_ARG_OR(index_n_heads, "index_n_heads", 0);
  LOAD_ARG_OR(index_topk, "index_topk", 2048);
  LOAD_ARG_OR(index_topk_freq, "index_topk_freq", 1);
  LOAD_ARG_OR(index_topk_pattern, "index_topk_pattern", "");
  LOAD_ARG_OR(index_skip_topk_offset, "index_skip_topk_offset", 0);

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return 256;  // args->qk_nope_head_dim() + args->qk_rope_head_dim();
  });
  LOAD_ARG_OR_FUNC(
      rotary_dim, "rotary_dim", [&] { return args->qk_rope_head_dim(); });

  SET_ARG(rope_scaling_rope_type, "deepseek_yarn");
  LOAD_ARG(rope_scaling_beta_fast, "rope_scaling.beta_fast");
  LOAD_ARG(rope_scaling_beta_slow, "rope_scaling.beta_slow");
  LOAD_ARG(rope_scaling_factor, "rope_scaling.factor");
  LOAD_ARG_OR(
      rope_extrapolation_factor, "rope_scaling.extrapolation_factor", 1.0f);
  LOAD_ARG(rope_scaling_mscale, "rope_scaling.mscale");
  LOAD_ARG(rope_scaling_mscale_all_dim, "rope_scaling.mscale_all_dim");
  LOAD_ARG(rope_scaling_original_max_position_embeddings,
           "rope_scaling.original_max_position_embeddings");
  LOAD_ARG_OR(rope_scaling_attn_factor, "rope_scaling.attn_factor", 1.0f);

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({1}));
});

REGISTER_MODEL_ARGS(deepseek_v32_mtp, [&] {
  LOAD_ARG_OR(model_type, "model_type", "deepseek_v32_mtp");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 129280);
  LOAD_ARG_OR(hidden_size, "hidden_size", 7168);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 61);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 128);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 128);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 18432);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 163840);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 1);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 0);
  LOAD_ARG_OR(rope_theta, "rope_theta", 10000.0f);
  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(sliding_window, "sliding_window", 4096);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 61);

  LOAD_ARG_OR(first_k_dense_replace, "first_k_dense_replace", 3);
  LOAD_ARG_OR(moe_layer_freq, "moe_layer_freq", 1);
  LOAD_ARG_OR(topk_method, "topk_method", "noaux_tc");
  LOAD_ARG_OR(n_routed_experts, "n_routed_experts", 256);
  LOAD_ARG_OR(n_shared_experts, "n_shared_experts", 1);
  LOAD_ARG_OR(num_experts_per_tok, "num_experts_per_tok", 8);
  LOAD_ARG_OR(moe_intermediate_size, "moe_intermediate_size", 2048);
  LOAD_ARG_OR(routed_scaling_factor, "routed_scaling_factor", 2.5f);
  LOAD_ARG_OR(norm_topk_prob, "norm_topk_prob", true);
  LOAD_ARG_OR(n_group, "n_group", 8);
  LOAD_ARG_OR(topk_group, "topk_group", 4);
  LOAD_ARG_OR(qk_nope_head_dim, "qk_nope_head_dim", 128);
  LOAD_ARG_OR(qk_rope_head_dim, "qk_rope_head_dim", 64);
  LOAD_ARG_OR(v_head_dim, "v_head_dim", 128);
  LOAD_ARG_OR(q_lora_rank, "q_lora_rank", 1536);
  LOAD_ARG_OR(kv_lora_rank, "kv_lora_rank", 512);
  LOAD_ARG_OR(index_head_dim, "index_head_dim", 128);
  LOAD_ARG_OR(index_n_heads, "index_n_heads", 64);
  LOAD_ARG_OR(index_topk, "index_topk", 2048);
  LOAD_ARG_OR(index_topk_freq, "index_topk_freq", 1);
  LOAD_ARG_OR(index_topk_pattern, "index_topk_pattern", "");
  LOAD_ARG_OR(index_skip_topk_offset, "index_skip_topk_offset", 0);
  LOAD_ARG_OR(
      index_share_for_mtp_iteration, "index_share_for_mtp_iteration", false);

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return 256;  // args->qk_nope_head_dim() + args->qk_rope_head_dim();
  });
  LOAD_ARG_OR_FUNC(
      rotary_dim, "rotary_dim", [&] { return args->qk_rope_head_dim(); });

  SET_ARG(rope_scaling_rope_type, "deepseek_yarn");
  LOAD_ARG(rope_scaling_beta_fast, "rope_scaling.beta_fast");
  LOAD_ARG(rope_scaling_beta_slow, "rope_scaling.beta_slow");
  LOAD_ARG(rope_scaling_factor, "rope_scaling.factor");
  LOAD_ARG_OR(
      rope_extrapolation_factor, "rope_scaling.extrapolation_factor", 1.0f);
  LOAD_ARG(rope_scaling_mscale, "rope_scaling.mscale");
  LOAD_ARG(rope_scaling_mscale_all_dim, "rope_scaling.mscale_all_dim");
  LOAD_ARG(rope_scaling_original_max_position_embeddings,
           "rope_scaling.original_max_position_embeddings");
  LOAD_ARG_OR(rope_scaling_attn_factor, "rope_scaling.attn_factor", 1.0f);

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({1}));
});

REGISTER_MODEL_ARGS(glm4, [&] {
  LOAD_ARG_OR(model_type, "model_type", "glm4");

  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(attention_bias, "attention_bias", true);
  LOAD_ARG_OR(attention_dropout, "attention_dropout", 0.0f);
  LOAD_ARG_OR(eos_token_id_vec, "eos_token_id", std::vector<int32_t>{151329});
  LOAD_ARG_OR(head_dim, "head_dim", 128);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "hidden_size", 4096);
  LOAD_ARG_OR(initializer_range, "initializer_range", 0.02f);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 13696);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 32768);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 32);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 40);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 2);
  LOAD_ARG_OR(pad_token_id, "pad_token_id", 151329);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-5);
  LOAD_ARG_OR(rope_theta, "rope_theta", 10000.0f);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_ARG_OR(vocab_size, "vocab_size", 151552);

  SET_ARG(stop_token_ids,
          std::unordered_set<int32_t>(args->eos_token_id_vec().begin(),
                                      args->eos_token_id_vec().end()));
});

REGISTER_MODEL_ARGS(glm4_moe, [&] {
  LOAD_ARG_OR(model_type, "model_type", "glm4_moe");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(attention_bias, "attention_bias", false);
  LOAD_ARG_OR(attention_dropout, "attention_dropout", 0.0f);
  LOAD_ARG_OR(decoder_sparse_step, "decoder_sparse_step", 1);
  LOAD_ARG_OR(eos_token_id_vec, "eos_token_id", std::vector<int32_t>{151329});
  LOAD_ARG_OR(head_dim, "head_dim", 128);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "hidden_size", 2048);
  LOAD_ARG_OR(initializer_range, "initializer_range", 0.02f);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 6144);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 40960);
  LOAD_ARG_OR(moe_intermediate_size, "moe_intermediate_size", 1536);
  LOAD_ARG_OR(routed_scaling_factor, "routed_scaling_factor", 2.5);
  LOAD_ARG_OR(norm_topk_prob, "norm_topk_prob", true);
  LOAD_ARG_OR(n_shared_experts, "n_shared_experts", 1);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 96);
  LOAD_ARG_OR(num_experts, "n_routed_experts", 160);
  LOAD_ARG_OR(n_group, "n_group", 1);
  LOAD_ARG_OR(topk_group, "topk_group", 1);
  LOAD_ARG_OR(num_experts_per_tok, "num_experts_per_tok", 8);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 48);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 4);
  LOAD_ARG_OR(use_qk_norm, "use_qk_norm", true);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_ARG_OR(vocab_size, "vocab_size", 151552);
  LOAD_ARG_OR(first_k_dense_replace, "first_k_dense_replace", 1);

  SET_ARG(stop_token_ids,
          std::unordered_set<int32_t>(args->eos_token_id_vec().begin(),
                                      args->eos_token_id_vec().end()));
});

REGISTER_MODEL_ARGS(glm4_moe_lite, [&] {
  LOAD_ARG_OR(model_type, "model_type", "glm4_moe_lite");
  // LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(dtype, "dtype", "");
  LOAD_ARG_OR(attention_bias, "attention_bias", false);
  LOAD_ARG_OR(attention_dropout, "attention_dropout", 0.0f);
  LOAD_ARG_OR(decoder_sparse_step, "decoder_sparse_step", 1);
  // LOAD_ARG_OR(eos_token_id, "eos_token_id", 154820);
  LOAD_ARG_OR(eos_token_id_vec, "eos_token_id", std::vector<int32_t>{154820});
  LOAD_ARG_OR(n_group, "n_group", 8);
  LOAD_ARG_OR(topk_group, "topk_group", 4);
  LOAD_ARG_OR(qk_nope_head_dim, "qk_nope_head_dim", 192);
  LOAD_ARG_OR(qk_rope_head_dim, "qk_rope_head_dim", 64);
  LOAD_ARG_OR(v_head_dim, "v_head_dim", 256);
  LOAD_ARG_OR(q_lora_rank, "q_lora_rank", 768);
  LOAD_ARG_OR(kv_lora_rank, "kv_lora_rank", 512);

  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "hidden_size", 2048);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 10240);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 202752);
  LOAD_ARG_OR(moe_intermediate_size, "moe_intermediate_size", 1536);
  LOAD_ARG_OR(routed_scaling_factor, "routed_scaling_factor", 1.8);
  LOAD_ARG_OR(norm_topk_prob, "norm_topk_prob", true);
  LOAD_ARG_OR(n_shared_experts, "n_shared_experts", 1);

  LOAD_ARG_OR(n_heads, "num_attention_heads", 20);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 20);

  int32_t headnum_p = 1 << (32 - __builtin_clz(args->n_heads() - 1));
  if (headnum_p != args->n_heads()) {
    LOG(INFO) << "--mock-padding-headnum from " << args->n_heads() << " to "
              << headnum_p;
    SET_ARG(actual_n_heads, args->n_heads());
    SET_ARG(n_heads, headnum_p);
    SET_ARG(n_kv_heads, headnum_p);
  }

  LOAD_ARG_OR(topk_group, "topk_group", 1);
  LOAD_ARG_OR(num_experts_per_tok, "num_experts_per_tok", 4);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 47);
  LOAD_ARG_OR(use_qk_norm, "use_qk_norm", true);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-05);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_ARG_OR(vocab_size, "vocab_size", 154880);
  LOAD_ARG_OR(first_k_dense_replace, "first_k_dense_replace", 1);

  LOAD_ARG_OR(topk_method, "topk_method", "noaux_tc");
  LOAD_ARG_OR(n_routed_experts, "n_routed_experts", 64);  // ep
  LOAD_ARG_OR(num_experts, "n_routed_experts", 64);

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->qk_nope_head_dim() + args->qk_rope_head_dim();
  });
  LOAD_ARG_OR_FUNC(
      rotary_dim, "rotary_dim", [&] { return args->qk_rope_head_dim(); });

  SET_ARG(stop_token_ids,
          std::unordered_set<int32_t>(args->eos_token_id_vec().begin(),
                                      args->eos_token_id_vec().end()));
});

REGISTER_MODEL_ARGS(glm4_moe_mtp, [&] {
  LOAD_ARG_OR(model_type, "model_type", "glm4_moe_mtp");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(attention_bias, "attention_bias", false);
  LOAD_ARG_OR(attention_dropout, "attention_dropout", 0.0f);
  LOAD_ARG_OR(decoder_sparse_step, "decoder_sparse_step", 1);
  LOAD_ARG_OR(eos_token_id_vec, "eos_token_id", std::vector<int32_t>{151329});
  LOAD_ARG_OR(head_dim, "head_dim", 128);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "hidden_size", 2048);
  LOAD_ARG_OR(initializer_range, "initializer_range", 0.02f);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 6144);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 40960);
  LOAD_ARG_OR(moe_intermediate_size, "moe_intermediate_size", 1536);
  LOAD_ARG_OR(routed_scaling_factor, "routed_scaling_factor", 2.5);
  LOAD_ARG_OR(norm_topk_prob, "norm_topk_prob", true);
  LOAD_ARG_OR(n_shared_experts, "n_shared_experts", 1);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 96);
  LOAD_ARG_OR(num_experts, "n_routed_experts", 160);
  LOAD_ARG_OR(n_group, "n_group", 1);
  LOAD_ARG_OR(topk_group, "topk_group", 1);
  LOAD_ARG_OR(num_experts_per_tok, "num_experts_per_tok", 8);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 48);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 4);
  LOAD_ARG_OR(use_qk_norm, "use_qk_norm", true);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_ARG_OR(vocab_size, "vocab_size", 151552);
  LOAD_ARG_OR(first_k_dense_replace, "first_k_dense_replace", 1);

  SET_ARG(stop_token_ids,
          std::unordered_set<int32_t>(args->eos_token_id_vec().begin(),
                                      args->eos_token_id_vec().end()));
});

REGISTER_MODEL_ARGS(
    glm_moe_dsa,
    ([&] {
      LOAD_ARG_OR(model_type, "model_type", "glm_moe_dsa");
      LOAD_ARG_OR(dtype, "dtype", "");
      LOAD_ARG_OR(attention_bias, "attention_bias", false);
      LOAD_ARG_OR(attention_dropout, "attention_dropout", 0.0f);

      LOAD_ARG_OR(vocab_size, "vocab_size", 154880);
      LOAD_ARG_OR(hidden_size, "hidden_size", 6144);
      LOAD_ARG_OR(n_layers, "num_hidden_layers", 78);
      LOAD_ARG_OR(n_heads, "num_attention_heads", 64);
      LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 64);
      LOAD_ARG_OR(intermediate_size, "intermediate_size", 12288);
      LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 202752);
      LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-5);
      LOAD_ARG_OR_FUNC(eos_token_id_vec, "eos_token_id", [&] {
        return std::vector<int32_t>{154820, 154827, 154829};
      });
      LOAD_ARG_OR(bos_token_id, "bos_token_id", 0);
      LOAD_ARG_OR(rope_theta, "rope_parameters.rope_theta", 1000000.0f);

      LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
      LOAD_ARG_OR(sliding_window, "sliding_window", 4096);
      LOAD_ARG_OR(max_window_layers, "max_window_layers", 61);

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
      LOAD_ARG_OR(qk_nope_head_dim, "qk_nope_head_dim", 192);
      LOAD_ARG_OR(qk_rope_head_dim, "qk_rope_head_dim", 64);
      LOAD_ARG_OR(v_head_dim, "v_head_dim", 256);
      LOAD_ARG_OR(q_lora_rank, "q_lora_rank", 2048);
      LOAD_ARG_OR(kv_lora_rank, "kv_lora_rank", 512);
      LOAD_ARG_OR(index_head_dim, "index_head_dim", 128);
      LOAD_ARG_OR(num_nextn_predict_layers, "num_nextn_predict_layers", 1);
      LOAD_ARG_OR(index_n_heads, "index_n_heads", 32);
      LOAD_ARG_OR(index_topk, "index_topk", 2048);
      LOAD_ARG_OR(index_topk_freq, "index_topk_freq", 1);
      LOAD_ARG_OR(index_topk_pattern, "index_topk_pattern", "");
      LOAD_ARG_OR(index_skip_topk_offset, "index_skip_topk_offset", 0);
      LOAD_ARG_OR(indexer_types, "indexer_types", std::vector<std::string>());
      LOAD_ARG_OR(
          mlp_layer_types, "mlp_layer_types", std::vector<std::string>());
      LOAD_ARG_OR(index_share_for_mtp_iteration,
                  "index_share_for_mtp_iteration",
                  false);

      LOAD_ARG_OR(use_qk_norm, "use_qk_norm", true);
      LOAD_ARG_OR(indexer_rope_interleave, "indexer_rope_interleave", true);
      LOAD_ARG_OR(rope_theta, "rope_theta", args->rope_theta());
      LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);

      SET_ARG(head_dim, args->qk_nope_head_dim() + args->qk_rope_head_dim());
      LOAD_ARG_OR_FUNC(
          rotary_dim, "rotary_dim", [&] { return args->qk_rope_head_dim(); });

      SET_ARG(stop_token_ids,
              std::unordered_set<int32_t>(args->eos_token_id_vec().begin(),
                                          args->eos_token_id_vec().end()));
    }));

REGISTER_TOKENIZER_ARGS(glm_moe_dsa, [&] {
  const std::vector<std::string> parser_visible_special_tokens({"<tool_call>",
                                                                "</tool_call>",
                                                                "<arg_key>",
                                                                "</arg_key>",
                                                                "<arg_value>",
                                                                "</arg_value>",
                                                                "<think>",
                                                                "</think>"});
  SET_ARG(visible_special_tokens, parser_visible_special_tokens);
});

REGISTER_MODEL_ARGS(
    glm_moe_dsa_mtp,
    ([&] {
      LOAD_ARG_OR(model_type, "model_type", "glm_moe_dsa_mtp");
      LOAD_ARG_OR(dtype, "dtype", "");
      LOAD_ARG_OR(attention_bias, "attention_bias", false);
      LOAD_ARG_OR(attention_dropout, "attention_dropout", 0.0f);

      LOAD_ARG_OR(vocab_size, "vocab_size", 154880);
      LOAD_ARG_OR(hidden_size, "hidden_size", 6144);
      LOAD_ARG_OR(n_layers, "num_hidden_layers", 1);
      LOAD_ARG_OR(n_heads, "num_attention_heads", 64);
      LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 64);
      LOAD_ARG_OR(intermediate_size, "intermediate_size", 12288);
      LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 202752);
      LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-5);
      LOAD_ARG_OR_FUNC(eos_token_id_vec, "eos_token_id", [&] {
        return std::vector<int32_t>{154820, 154827, 154829};
      });
      LOAD_ARG_OR(bos_token_id, "bos_token_id", 0);
      LOAD_ARG_OR(rope_theta, "rope_parameters.rope_theta", 1000000.0f);

      LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
      LOAD_ARG_OR(sliding_window, "sliding_window", 4096);
      LOAD_ARG_OR(max_window_layers, "max_window_layers", 61);

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
      LOAD_ARG_OR(qk_nope_head_dim, "qk_nope_head_dim", 192);
      LOAD_ARG_OR(qk_rope_head_dim, "qk_rope_head_dim", 64);
      LOAD_ARG_OR(v_head_dim, "v_head_dim", 256);
      LOAD_ARG_OR(q_lora_rank, "q_lora_rank", 2048);
      LOAD_ARG_OR(kv_lora_rank, "kv_lora_rank", 512);
      LOAD_ARG_OR(index_head_dim, "index_head_dim", 128);
      LOAD_ARG_OR(num_nextn_predict_layers, "num_nextn_predict_layers", 1);
      LOAD_ARG_OR(index_n_heads, "index_n_heads", 32);
      LOAD_ARG_OR(index_topk, "index_topk", 2048);
      LOAD_ARG_OR(index_topk_freq, "index_topk_freq", 1);
      LOAD_ARG_OR(index_topk_pattern, "index_topk_pattern", "");
      LOAD_ARG_OR(index_skip_topk_offset, "index_skip_topk_offset", 0);
      LOAD_ARG_OR(indexer_types, "indexer_types", std::vector<std::string>());
      LOAD_ARG_OR(
          mlp_layer_types, "mlp_layer_types", std::vector<std::string>());
      LOAD_ARG_OR(index_share_for_mtp_iteration,
                  "index_share_for_mtp_iteration",
                  false);

      LOAD_ARG_OR(use_qk_norm, "use_qk_norm", true);
      LOAD_ARG_OR(indexer_rope_interleave, "indexer_rope_interleave", true);
      LOAD_ARG_OR(rope_theta, "rope_theta", args->rope_theta());
      LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);

      SET_ARG(head_dim, args->qk_nope_head_dim() + args->qk_rope_head_dim());
      LOAD_ARG_OR_FUNC(
          rotary_dim, "rotary_dim", [&] { return args->qk_rope_head_dim(); });

      SET_ARG(stop_token_ids,
              std::unordered_set<int32_t>(args->eos_token_id_vec().begin(),
                                          args->eos_token_id_vec().end()));
    }));

REGISTER_TOKENIZER_ARGS(glm_moe_dsa_mtp, [&] {
  const std::vector<std::string> parser_visible_special_tokens({"<tool_call>",
                                                                "</tool_call>",
                                                                "<arg_key>",
                                                                "</arg_key>",
                                                                "<arg_value>",
                                                                "</arg_value>",
                                                                "<think>",
                                                                "</think>"});
  SET_ARG(visible_special_tokens, parser_visible_special_tokens);
});

REGISTER_MODEL_ARGS(joyai_llm_flash, [&] {
  LOAD_ARG_OR(model_type, "model_type", "joyai_llm_flash");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 129280);
  LOAD_ARG_OR(hidden_size, "hidden_size", 2048);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 40);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 32);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 32);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 7168);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 131072);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 1);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 0);
  LOAD_ARG_OR(rope_theta, "rope_theta", 32000000.0f);

  LOAD_ARG_OR(first_k_dense_replace, "first_k_dense_replace", 1);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(moe_layer_freq, "moe_layer_freq", 1);
  LOAD_ARG_OR(topk_method, "topk_method", "noaux_tc");
  LOAD_ARG_OR(n_routed_experts, "n_routed_experts", 256);
  LOAD_ARG_OR(n_shared_experts, "n_shared_experts", 1);
  LOAD_ARG_OR(num_experts_per_tok, "num_experts_per_tok", 8);
  LOAD_ARG_OR(moe_intermediate_size, "moe_intermediate_size", 768);
  LOAD_ARG_OR(routed_scaling_factor, "routed_scaling_factor", 2.5f);
  LOAD_ARG_OR(norm_topk_prob, "norm_topk_prob", true);
  LOAD_ARG_OR(n_group, "n_group", 1);
  LOAD_ARG_OR(topk_group, "topk_group", 1);
  LOAD_ARG_OR(scoring_func, "scoring_func", "sigmoid");
  LOAD_ARG_OR(qk_nope_head_dim, "qk_nope_head_dim", 128);
  LOAD_ARG_OR(qk_rope_head_dim, "qk_rope_head_dim", 64);
  LOAD_ARG_OR(v_head_dim, "v_head_dim", 128);
  LOAD_ARG_OR(q_lora_rank, "q_lora_rank", 1536);
  LOAD_ARG_OR(kv_lora_rank, "kv_lora_rank", 512);
  LOAD_ARG_OR(num_nextn_predict_layers, "num_nextn_predict_layers", 1);

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return 256;  // args->qk_nope_head_dim() + args->qk_rope_head_dim();
  });
  LOAD_ARG_OR_FUNC(
      rotary_dim, "rotary_dim", [&] { return args->qk_rope_head_dim(); });

  // uses default rope_type, no deepseek_yarn scaling
  SET_ARG(rope_scaling_rope_type, "default");
  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({1}));
});

REGISTER_MODEL_ARGS(kimi_k2, [&] {
  LOAD_ARG_OR(model_type, "model_type", "kimi_k2");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 163840);
  LOAD_ARG_OR(hidden_size, "hidden_size", 7168);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 61);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 64);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 64);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 18432);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 131072);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 163585);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 163584);
  LOAD_ARG_OR(rope_theta, "rope_theta", 50000.0f);
  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(sliding_window, "sliding_window", 4096);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 61);
  LOAD_ARG_OR(first_k_dense_replace, "first_k_dense_replace", 1);
  LOAD_ARG_OR(moe_layer_freq, "moe_layer_freq", 1);
  LOAD_ARG_OR(topk_method, "topk_method", "noaux_tc");
  LOAD_ARG_OR(n_routed_experts, "n_routed_experts", 384);
  LOAD_ARG_OR(n_shared_experts, "n_shared_experts", 1);
  LOAD_ARG_OR(num_experts_per_tok, "num_experts_per_tok", 8);
  LOAD_ARG_OR(moe_intermediate_size, "moe_intermediate_size", 2048);
  LOAD_ARG_OR(routed_scaling_factor, "routed_scaling_factor", 2.827f);
  LOAD_ARG_OR(norm_topk_prob, "norm_topk_prob", true);
  LOAD_ARG_OR(n_group, "n_group", 1);
  LOAD_ARG_OR(topk_group, "topk_group", 1);
  LOAD_ARG_OR(qk_nope_head_dim, "qk_nope_head_dim", 128);
  LOAD_ARG_OR(qk_rope_head_dim, "qk_rope_head_dim", 64);
  LOAD_ARG_OR(v_head_dim, "v_head_dim", 128);
  LOAD_ARG_OR(q_lora_rank, "q_lora_rank", 1536);
  LOAD_ARG_OR(kv_lora_rank, "kv_lora_rank", 512);
  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return 256;  // args->qk_nope_head_dim() + args->qk_rope_head_dim();
  });
  LOAD_ARG_OR_FUNC(
      rotary_dim, "rotary_dim", [&] { return args->qk_rope_head_dim(); });
  SET_ARG(rope_scaling_rope_type, "deepseek_yarn");
  LOAD_ARG(rope_scaling_beta_fast, "rope_scaling.beta_fast");
  LOAD_ARG(rope_scaling_beta_slow, "rope_scaling.beta_slow");
  LOAD_ARG(rope_scaling_factor, "rope_scaling.factor");
  LOAD_ARG_OR(
      rope_extrapolation_factor, "rope_scaling.extrapolation_factor", 1.0f);
  LOAD_ARG(rope_scaling_mscale, "rope_scaling.mscale");
  LOAD_ARG(rope_scaling_mscale_all_dim, "rope_scaling.mscale_all_dim");
  LOAD_ARG(rope_scaling_original_max_position_embeddings,
           "rope_scaling.original_max_position_embeddings");
  LOAD_ARG_OR(rope_scaling_attn_factor, "rope_scaling.attn_factor", 1.0f);
  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({163585, 163586}));
});

REGISTER_TOKENIZER_ARGS(kimi_k2, [&] {
  SET_ARG(tokenizer_type, "tiktoken");
  SET_ARG(vocab_file, "tiktoken.model");
  // set special tokens
  // ref to:
  // https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/tokenizer_config.json
  const std::vector<SpecialToken> special_tokens(
      {{"<|im_end|>", 163586},
       {"<|im_user|>", 163587},
       {"<|im_assistant|>", 163588},
       {"<|start_header_id|>", 163590},
       {"<|end_header_id|>", 163591},
       {"[EOT]", 163593},
       {"<|im_system|>", 163594},
       {"<|im_middle|>", 163601}});
  SET_ARG(special_tokens, special_tokens);
  // set regex pattern for tiktoken tokenizer.
  // ref to:
  // https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/tokenization_kimi.py#L58
  // N.B. replaced '\s+(?!\S)' with '\s+[^\s]' to avoid regex error
  const std::string pattern_str =
      R"([\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+[^\s]|\s+)";
  SET_ARG(pattern, pattern_str);
});

REGISTER_MODEL_ARGS(llama, [&] {
  LOAD_ARG_OR(model_type, "model_type", "llama");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG(n_kv_heads, "num_key_value_heads");
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");

  // decide model type based on vocab size
  LOAD_ARG_OR(vocab_size, "vocab_size", 128256);
  if (args->vocab_size() == 128256) {
    // choose the right chat template
    SET_ARG(model_type, "llama3");

    LOAD_ARG_OR(hidden_size, "hidden_size", 8192);
    LOAD_ARG_OR(n_layers, "num_hidden_layers", 80);
    LOAD_ARG_OR(n_heads, "num_attention_heads", 64);
    LOAD_ARG_OR(intermediate_size, "intermediate_size", 28672);
    LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 8192);
    LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-5);
    LOAD_ARG_OR(bos_token_id, "bos_token_id", 128000);
    // TODO: support a list of eos token ids
    LOAD_ARG_OR(eos_token_id, "eos_token_id", 128001);
    LOAD_ARG_OR(rope_theta, "rope_theta", 500000.0f);
    // load rope scaling parameters
    LOAD_ARG(rope_scaling_rope_type, "rope_scaling.rope_type");
    LOAD_ARG(rope_scaling_factor, "rope_scaling.factor");
    LOAD_ARG(rope_scaling_low_freq_factor, "rope_scaling.low_freq_factor");
    LOAD_ARG(rope_scaling_high_freq_factor, "rope_scaling.high_freq_factor");
    LOAD_ARG(rope_scaling_original_max_position_embeddings,
             "rope_scaling.original_max_position_embeddings");
    // stop token ids: "<|eom_id|>", "<|eot_id|>"
    SET_ARG(stop_token_ids, std::unordered_set<int32_t>({128008, 128009}));
  } else {
    // llama 2
    LOAD_ARG_OR(hidden_size, "hidden_size", 4096);
    LOAD_ARG_OR(n_layers, "num_hidden_layers", 32);
    LOAD_ARG_OR(n_heads, "num_attention_heads", 32);
    LOAD_ARG_OR(intermediate_size, "intermediate_size", 11008);
    LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 2048);
    LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-5);
    LOAD_ARG_OR(bos_token_id, "bos_token_id", 1);
    LOAD_ARG_OR(eos_token_id, "eos_token_id", 2);
    LOAD_ARG_OR(rope_theta, "rope_theta", 10000.0f);
    // LOAD_ARG_OR(rope_scaling, "rope_scaling", 1.0f);
  }

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });
});

REGISTER_MODEL_ARGS(mistral, [&] {
  LOAD_ARG_OR(model_type, "model_type", "mistral");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 32000);
  LOAD_ARG_OR(hidden_size, "hidden_size", 4096);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 32);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 32);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 8);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 14336);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 4096 * 32);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-5);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 1);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 2);
  LOAD_ARG_OR(rope_theta, "rope_theta", 10000.0f);

  LOAD_ARG_OR(rope_scaling_rope_type, "rope_scaling_rope_type", "default");
  LOAD_ARG_OR(rope_scaling_factor, "rope_scaling_factor", 1.0f);
  LOAD_ARG_OR(rope_scaling_original_max_position_embeddings,
              "rope_scaling_original_max_position_embeddings",
              4096);
  LOAD_ARG_OR(rope_extrapolation_factor, "rope_extrapolation_factor", 1.0f);
  LOAD_ARG_OR(rope_scaling_attn_factor, "rope_scaling_attn_factor", 1.0f);
  LOAD_ARG_OR(rope_scaling_beta_fast, "rope_scaling_beta_fast", 32.0f);
  LOAD_ARG_OR(rope_scaling_beta_slow, "rope_scaling_beta_slow", 1.0f);
  LOAD_ARG_OR(rope_scaling_mscale, "rope_scaling_mscale", 1.0f);
  LOAD_ARG_OR(rope_scaling_mscale_all_dim, "rope_scaling_mscale_all_dim", 1.0f);

  // head_dim needs to be calculated from hidden_size and n_heads, must use
  // LOAD_ARG_OR_FUNC
  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });
});

REGISTER_MODEL_ARGS(qwen3_eagle3, [&] {
  LOAD_ARG_OR(model_type, "model_type", "qwen3_eagle3");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 152064);
  LOAD_ARG_OR(draft_vocab_size, "draft_vocab_size", 0);
  if (args->draft_vocab_size() > 0) {
    args->vocab_size(args->draft_vocab_size());
  }
  LOAD_ARG_OR(hidden_size, "hidden_size", 3584);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 1);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 28);
  LOAD_ARG(n_kv_heads, "num_key_value_heads");
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 18944);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 32768);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 151643);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);
  LOAD_ARG_OR(rope_scaling_mrope_section,
              "text_config.rope_scaling.mrope_section",
              std::vector<int64_t>{});
  LOAD_ARG_OR(rope_scaling_mrope_section,
              "text_config.rope_parameters.mrope_section",
              args->rope_scaling_mrope_section());
  LOAD_ARG_OR(rope_scaling_mrope_section,
              "rope_parameters.mrope_section",
              args->rope_scaling_mrope_section());
  LOAD_ARG_OR(rope_scaling_mrope_section,
              "rope_scaling.mrope_section",
              args->rope_scaling_mrope_section());

  // For qwen3/2.5 model < 7B, tie_word_embeddings = true
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);

  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 28);

  // Eagle3.1 (speculators-format) draft variants, read straight from the
  // checkpoint's own config keys. normalize_speculators_config only sets
  // model_type and infers use_qk_norm.
  LOAD_ARG_OR(use_qk_norm, "use_qk_norm", false);
  LOAD_ARG_OR(norm_before_fc, "norm_before_fc", false);
  LOAD_ARG_OR(norm_before_residual, "norm_before_residual", false);
  LOAD_ARG_OR(norm_output, "norm_output", false);
  LOAD_ARG_OR(fc_norm, "fc_norm", false);

  // Only the element count is consumed (fc_norm_.size() in the ctor), so the
  // speculators boundary indices are stored as-is; the 0-based post-layer shift
  // is applied on the target capture side.
  LOAD_ARG_OR(layers_to_capture,
              "eagle_aux_hidden_state_layer_ids",
              std::vector<int32_t>{});

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});

using Glm4vMultimodalProcessor = MultimodalProcessor<GLM4VPromptProcessor,
                                                     Glm4VImageProcessor,
                                                     Glm4VVideoProcessor>;

REGISTER_MULTIMODAL_PROCESSOR(glm4v, Glm4vMultimodalProcessor);

REGISTER_MPOSITION_GENERATOR(glm4v, xllm::Glm4VMPositionGenerator);

REGISTER_MODEL_ARGS(glm4v, [&] {
  LOAD_ARG_OR(model_type, "model_type", "glm4v");
  LOAD_ARG_OR(image_start_token_id, "image_start_token_id", 151339);
  LOAD_ARG_OR(image_end_token_id, "image_end_token_id", 151340);
  LOAD_ARG_OR(video_start_token_id, "video_start_token_id", 151341);
  LOAD_ARG_OR(video_end_token_id, "video_end_token_id", 151342);
  LOAD_ARG_OR(image_token_id, "image_token_id", 151363);
  LOAD_ARG_OR(video_token_id, "video_token_id", 151364);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);

  // text config
  LOAD_ARG_OR(vocab_size, "text_config.vocab_size", 151552);
  // LOAD_ARG_OR(pad_token_id, "text_config.pad_token_id", 151329);
  LOAD_ARG_OR(eos_token_id_vec,
              "text_config.eos_token_id",
              std::vector<int32_t>{151329});
  LOAD_ARG_OR(attention_bias, "text_config.attention_bias", true);
  LOAD_ARG_OR(attention_dropout, "text_config.attention_dropout", 0.0f);
  LOAD_ARG_OR(first_k_dense_replace, "text_config.first_k_dense_replace", 1);
  LOAD_ARG_OR(hidden_act, "text_config.hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "text_config.hidden_size", 4096);
  LOAD_ARG_OR(initializer_range, "text_config.initializer_range", 0.02);
  LOAD_ARG_OR(intermediate_size, "text_config.intermediate_size", 10944);
  LOAD_ARG_OR(
      max_position_embeddings, "text_config.max_position_embeddings", 131072);
  LOAD_ARG_OR(n_heads, "text_config.num_attention_heads", 96);
  LOAD_ARG_OR_FUNC(head_dim, "text_config.head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });
  LOAD_ARG_OR(num_experts_per_tok, "text_config.num_experts_per_tok", 8);
  LOAD_ARG_OR(n_layers, "text_config.num_hidden_layers", 46);
  LOAD_ARG_OR(n_kv_heads, "text_config.num_key_value_heads", 8);
  // LOAD_ARG_OR(partial_rotary_factor, "text_config.partial_rotary_factor",
  // 0.5);
  LOAD_ARG_OR(rms_norm_eps, "text_config.rms_norm_eps", 1e-05);
  LOAD_ARG_OR(dtype, "text_config.dtype", "bfloat16");
  LOAD_ARG_OR(
      rope_scaling_rope_type, "text_config.rope_parameters.type", "mrope");
  LOAD_ARG(rope_scaling_mrope_section,
           "text_config.rope_parameters.mrope_section");
  LOAD_ARG_OR(rope_theta, "text_config.rope_parameters.rope_theta", 500000.0f);
  LOAD_ARG_OR(routed_scaling_factor, "text_config.routed_scaling_factor", 1.0);
  LOAD_ARG_OR(topk_group, "text_config.topk_group", 1);
  // LOAD_ARG_OR(use_cache, "text_config.use_cache", true);
  LOAD_ARG_OR(use_qk_norm, "text_config.use_qk_norm", false);

  // vision config
  // LOAD_ARG_OR(mm_attention_bias, "vision_config.attention_bias", false);
  // LOAD_ARG_OR(mm_attention_dropout, "vision_config.attention_dropout", 0.0f);
  LOAD_ARG_OR(mm_num_hidden_layers, "vision_config.depth", 24);
  LOAD_ARG_OR(mm_hidden_act, "vision_config.hidden_act", "silu");
  LOAD_ARG_OR(mm_hidden_size, "vision_config.hidden_size", 1536);
  LOAD_ARG_OR(mm_image_size, "vision_config.image_size", 336);
  LOAD_ARG_OR(mm_num_channels, "vision_config.in_channels", 3);
  LOAD_ARG_OR(mm_initializer_range, "vision_config.initializer_range", 0.02);
  LOAD_ARG_OR(mm_intermediate_size, "vision_config.intermediate_size", 10944);
  LOAD_ARG_OR(mm_num_attention_heads, "vision_config.num_heads", 12);
  LOAD_ARG_OR(mm_projection_dim, "vision_config.out_hidden_size", 4096);
  LOAD_ARG_OR(mm_patch_size, "vision_config.patch_size", 14);
  // LOAD_ARG_OR(mm_rms_norm_eps, "vision_config.rms_norm_eps", 1e-05);
  LOAD_ARG_OR(mm_spatial_merge_size, "vision_config.spatial_merge_size", 2);
  LOAD_ARG_OR(mm_temporal_patch_size, "vision_config.temporal_patch_size", 2);
  // Vision-scoped key so JsonReader's text_config fallback cannot resolve
  // "head_dim" against text_config.head_dim (which is 0 for GLM-5.3-Flash and
  // 128 for future GLM4V variants that adopt the nested HF multimodal layout).
  // vision_config has no head_dim; the fallback lambda derives it from
  // mm_hidden_size / mm_num_attention_heads.
  LOAD_ARG_OR_FUNC(mm_head_dim, "vision_config.head_dim", [&] {
    return args->mm_hidden_size() / args->mm_num_attention_heads();
  });

  SET_ARG(stop_token_ids,
          std::unordered_set<int32_t>(args->eos_token_id_vec().begin(),
                                      args->eos_token_id_vec().end()));
  SET_ARG(rope_scaling_rope_type, "mrope");
});

using Glm4vMoeMultimodalProcessor = MultimodalProcessor<GLM4VPromptProcessor,
                                                        Glm4VImageProcessor,
                                                        Glm4VVideoProcessor>;

REGISTER_MULTIMODAL_PROCESSOR(glm4v_moe, Glm4vMoeMultimodalProcessor);

REGISTER_MPOSITION_GENERATOR(glm4v_moe, xllm::Glm4VMPositionGenerator);

REGISTER_MODEL_ARGS(glm4v_moe, [&] {
  LOAD_ARG_OR(model_type, "model_type", "glm4v_moe");
  LOAD_ARG_OR(image_start_token_id, "image_start_token_id", 151339);
  LOAD_ARG_OR(image_end_token_id, "image_end_token_id", 151340);
  LOAD_ARG_OR(video_start_token_id, "video_start_token_id", 151341);
  LOAD_ARG_OR(video_end_token_id, "video_end_token_id", 151342);
  LOAD_ARG_OR(image_token_id, "image_token_id", 151363);
  LOAD_ARG_OR(video_token_id, "video_token_id", 151364);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);

  // text config
  LOAD_ARG_OR(vocab_size, "text_config.vocab_size", 151552);
  // LOAD_ARG_OR(pad_token_id, "text_config.pad_token_id", 151329);
  LOAD_ARG_OR(eos_token_id_vec,
              "text_config.eos_token_id",
              std::vector<int32_t>{151329});
  LOAD_ARG_OR_FUNC(head_dim, "text_config.head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });
  LOAD_ARG_OR(attention_bias, "text_config.attention_bias", true);
  LOAD_ARG_OR(attention_dropout, "text_config.attention_dropout", 0.0f);
  LOAD_ARG_OR(first_k_dense_replace, "text_config.first_k_dense_replace", 1);
  LOAD_ARG_OR(hidden_act, "text_config.hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "text_config.hidden_size", 4096);
  LOAD_ARG_OR(initializer_range, "text_config.initializer_range", 0.02);
  LOAD_ARG_OR(intermediate_size, "text_config.intermediate_size", 10944);
  LOAD_ARG_OR(
      max_position_embeddings, "text_config.max_position_embeddings", 131072);
  LOAD_ARG_OR(moe_intermediate_size, "text_config.moe_intermediate_size", 1408);
  LOAD_ARG_OR(n_group, "text_config.n_group", 1);
  LOAD_ARG_OR(num_experts, "text_config.n_routed_experts", 128);
  LOAD_ARG_OR(n_shared_experts, "text_config.n_shared_experts", 1);
  LOAD_ARG_OR(norm_topk_prob, "text_config.norm_topk_prob", true);
  LOAD_ARG_OR(n_heads, "text_config.num_attention_heads", 96);
  LOAD_ARG_OR(num_experts_per_tok, "text_config.num_experts_per_tok", 8);
  LOAD_ARG_OR(n_layers, "text_config.num_hidden_layers", 46);
  LOAD_ARG_OR(n_kv_heads, "text_config.num_key_value_heads", 8);
  // LOAD_ARG_OR(partial_rotary_factor, "text_config.partial_rotary_factor",
  // 0.5);
  LOAD_ARG_OR(rms_norm_eps, "text_config.rms_norm_eps", 1e-05);
  LOAD_ARG_OR(dtype, "text_config.dtype", "bfloat16");
  LOAD_ARG_OR(
      rope_scaling_rope_type, "text_config.rope_parameters.type", "mrope");
  LOAD_ARG(rope_scaling_mrope_section,
           "text_config.rope_parameters.mrope_section");
  LOAD_ARG_OR(rope_theta, "text_config.rope_parameters.rope_theta", 500000.0f);
  LOAD_ARG_OR(routed_scaling_factor, "text_config.routed_scaling_factor", 1.0);
  LOAD_ARG_OR(topk_group, "text_config.topk_group", 1);
  // LOAD_ARG_OR(use_cache, "text_config.use_cache", true);
  LOAD_ARG_OR(use_qk_norm, "text_config.use_qk_norm", false);

  // vision config
  // LOAD_ARG_OR(mm_attention_bias, "vision_config.attention_bias", false);
  // LOAD_ARG_OR(mm_attention_dropout, "vision_config.attention_dropout", 0.0f);
  LOAD_ARG_OR(mm_num_hidden_layers, "vision_config.depth", 24);
  LOAD_ARG_OR(mm_hidden_act, "vision_config.hidden_act", "silu");
  LOAD_ARG_OR(mm_hidden_size, "vision_config.hidden_size", 1536);
  LOAD_ARG_OR(mm_image_size, "vision_config.image_size", 336);
  LOAD_ARG_OR(mm_num_channels, "vision_config.in_channels", 3);
  LOAD_ARG_OR(mm_initializer_range, "vision_config.initializer_range", 0.02);
  LOAD_ARG_OR(mm_intermediate_size, "vision_config.intermediate_size", 10944);
  LOAD_ARG_OR(mm_num_attention_heads, "vision_config.num_heads", 12);
  LOAD_ARG_OR(mm_projection_dim, "vision_config.out_hidden_size", 4096);
  LOAD_ARG_OR(mm_patch_size, "vision_config.patch_size", 14);
  // LOAD_ARG_OR(mm_rms_norm_eps, "vision_config.rms_norm_eps", 1e-05);
  LOAD_ARG_OR(mm_spatial_merge_size, "vision_config.spatial_merge_size", 2);
  LOAD_ARG_OR(mm_temporal_patch_size, "vision_config.temporal_patch_size", 2);
  LOAD_ARG_OR_FUNC(mm_head_dim, "head_dim", [&] {
    return args->mm_hidden_size() / args->mm_num_attention_heads();
  });

  SET_ARG(stop_token_ids,
          std::unordered_set<int32_t>(args->eos_token_id_vec().begin(),
                                      args->eos_token_id_vec().end()));
  SET_ARG(rope_scaling_rope_type, "mrope");
});

using KimiK25MultimodalProcessor = MultimodalProcessor<KimiK25PromptProcessor,
                                                       KimiK25ImageProcessor,
                                                       KimiK25VideoProcessor>;

REGISTER_MULTIMODAL_PROCESSOR(kimi_k25, KimiK25MultimodalProcessor);

REGISTER_MODEL_ARGS(kimi_k25, [&] {
  // text config (Kimi-K2.5): args are under text_config.* in HF config.
  LOAD_ARG_OR(model_type, "model_type", "kimi_k25");
  LOAD_ARG_OR_FUNC(dtype, "dtype", [&] {
    return json.value_or<std::string>("torch_dtype", "bfloat16");
  });
  LOAD_ARG_OR(vocab_size, "text_config.vocab_size", 163840);
  LOAD_ARG_OR(hidden_act, "text_config.hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "text_config.hidden_size", 7168);
  LOAD_ARG_OR(initializer_range, "text_config.initializer_range", 0.02f);
  LOAD_ARG_OR(intermediate_size, "text_config.intermediate_size", 18432);
  LOAD_ARG_OR(n_layers, "text_config.num_hidden_layers", 61);
  LOAD_ARG_OR(n_heads, "text_config.num_attention_heads", 64);
  LOAD_ARG_OR(n_kv_heads, "text_config.num_key_value_heads", 64);
  LOAD_ARG_OR(
      max_position_embeddings, "text_config.max_position_embeddings", 262144);
  LOAD_ARG_OR(rms_norm_eps, "text_config.rms_norm_eps", 1e-05);
  LOAD_ARG_OR(rope_theta, "text_config.rope_theta", 50000.0f);

  LOAD_ARG_OR(attention_bias, "text_config.attention_bias", false);
  LOAD_ARG_OR(attention_dropout, "text_config.attention_dropout", 0.0f);

  // [Kimi-K2.5 config missing, keep DeepSeek-V3-compatible default]
  LOAD_ARG_OR(max_window_layers, "text_config.max_window_layers", 61);
  LOAD_ARG_OR(tie_word_embeddings, "text_config.tie_word_embeddings", false);

  // DeepSeek-V3 style MoE / MLA args.
  LOAD_ARG_OR(first_k_dense_replace, "text_config.first_k_dense_replace", 1);
  LOAD_ARG_OR(moe_layer_freq, "text_config.moe_layer_freq", 1);
  LOAD_ARG_OR(topk_method, "text_config.topk_method", "noaux_tc");
  LOAD_ARG_OR(n_routed_experts, "text_config.n_routed_experts", 384);
  LOAD_ARG_OR(n_shared_experts, "text_config.n_shared_experts", 1);
  LOAD_ARG_OR(num_experts_per_tok, "text_config.num_experts_per_tok", 8);
  LOAD_ARG_OR(moe_intermediate_size, "text_config.moe_intermediate_size", 2048);
  LOAD_ARG_OR(
      routed_scaling_factor, "text_config.routed_scaling_factor", 2.827f);
  LOAD_ARG_OR(norm_topk_prob, "text_config.norm_topk_prob", true);
  LOAD_ARG_OR(n_group, "text_config.n_group", 1);
  LOAD_ARG_OR(topk_group, "text_config.topk_group", 1);
  LOAD_ARG_OR(qk_nope_head_dim, "text_config.qk_nope_head_dim", 128);
  LOAD_ARG_OR(qk_rope_head_dim, "text_config.qk_rope_head_dim", 64);
  LOAD_ARG_OR(v_head_dim, "text_config.v_head_dim", 128);
  LOAD_ARG_OR(q_lora_rank, "text_config.q_lora_rank", 1536);
  LOAD_ARG_OR(kv_lora_rank, "text_config.kv_lora_rank", 512);
  LOAD_ARG_OR(scoring_func, "text_config.scoring_func", "sigmoid");

  SET_ARG(head_dim, args->qk_nope_head_dim() + args->qk_rope_head_dim());
  LOAD_ARG_OR_FUNC(
      rotary_dim, "rotary_dim", [&] { return args->qk_rope_head_dim(); });

  LOAD_ARG(rope_scaling_rope_type, "text_config.rope_scaling.type");
  if (args->rope_scaling_rope_type().empty() ||
      args->rope_scaling_rope_type() == "default" ||
      args->rope_scaling_rope_type() == "yarn") {
    // Kimi-K2.5 text config uses HF rope_scaling.type="yarn", while the
    // DeepSeek MLA implementation in xLLM keys off "deepseek_yarn".
    args->rope_scaling_rope_type() = "deepseek_yarn";
  }
  LOAD_ARG(rope_scaling_beta_fast, "text_config.rope_scaling.beta_fast");
  LOAD_ARG(rope_scaling_beta_slow, "text_config.rope_scaling.beta_slow");
  LOAD_ARG(rope_scaling_factor, "text_config.rope_scaling.factor");
  LOAD_ARG_OR(rope_extrapolation_factor,
              "text_config.rope_scaling.extrapolation_factor",
              1.0f);
  LOAD_ARG(rope_scaling_mscale, "text_config.rope_scaling.mscale");
  LOAD_ARG(rope_scaling_mscale_all_dim,
           "text_config.rope_scaling.mscale_all_dim");
  LOAD_ARG(rope_scaling_original_max_position_embeddings,
           "text_config.rope_scaling.original_max_position_embeddings");
  LOAD_ARG_OR(
      rope_scaling_attn_factor, "text_config.rope_scaling.attn_factor", 1.0f);
  LOAD_ARG_OR(
      num_nextn_predict_layers, "text_config.num_nextn_predict_layers", 1);
  SET_ARG(max_concurrent_prefills_per_dp, 2);
  LOAD_ARG_OR(layers_to_capture, "layers_to_capture", std::vector<int32_t>{});
  LOAD_ARG_OR(layers_to_capture,
              "text_config.layers_to_capture",
              args->layers_to_capture());

  LOAD_ARG_OR(bos_token_id, "text_config.bos_token_id", 163584);
  LOAD_ARG_OR(eos_token_id, "text_config.eos_token_id", 163585);
  LOAD_ARG_OR(pad_token_id, "text_config.pad_token_id", 163839);

  // Kimi-K2.5 uses media_* special tokens instead of Qwen's vision/image/video
  // token family. The official config only exposes media_placeholder_token_id;
  // the begin/content/end ids come from the tokenizer special token table
  // below.

  LOAD_ARG_OR_FUNC(vision_start_token_id, "vision_start_token_id", [&] {
    return int32_t(163602);  // <|media_begin|>
  });
  LOAD_ARG_OR_FUNC(vision_end_token_id, "vision_end_token_id", [&] {
    return int32_t(163604);  // <|media_end|>
  });
  LOAD_ARG_OR_FUNC(vision_token_id, "vision_token_id", [&] {
    return int32_t(163603);  // <|media_content|>
  });
  LOAD_ARG_OR_FUNC(image_token_id, "image_token_id", [&] {
    return int32_t(163605);  // <|media_pad|>
  });
  LOAD_ARG_OR_FUNC(video_token_id, "video_token_id", [&] {
    return int32_t(163605);  // <|media_pad|>
  });

  // vision_config inferred mapping (Kimi-K2.5):
  LOAD_ARG_OR(mm_num_hidden_layers, "vision_config.vt_num_hidden_layers", 27);
  // vt_hidden_act is not provided in Kimi-K2.5 config, keep default "silu".
  LOAD_ARG_OR(mm_hidden_act, "vision_config.vt_hidden_act", "silu");
  LOAD_ARG_OR(mm_hidden_size, "vision_config.mm_hidden_size", 1152);
  LOAD_ARG_OR(mm_intermediate_size, "vision_config.vt_intermediate_size", 4304);
  LOAD_ARG_OR(
      mm_num_attention_heads, "vision_config.vt_num_attention_heads", 16);

  // Projector-related args from Kimi-K2.5 vision_config.
  LOAD_ARG_OR(mm_projection_dim, "vision_config.text_hidden_size", 7168);
  LOAD_ARG_OR(
      mm_projector_type, "vision_config.mm_projector_type", "patchmerger");
  LOAD_ARG_OR(
      mm_projector_hidden_act, "vision_config.projector_hidden_act", "gelu");
  // NOTE: projector_ln_eps is mapped to mm_layer_norm_eps by inference.
  LOAD_ARG_OR(mm_layer_norm_eps, "vision_config.projector_ln_eps", 1e-05f);

  LOAD_ARG_OR(mm_patch_size, "vision_config.patch_size", 14);
  // Kimi-K2.5 uses merge_kernel_size (e.g. [2,2]); map first dim by inference.
  LOAD_ARG_OR_FUNC(
      mm_spatial_merge_size, "vision_config.spatial_merge_size", [&] {
        if (auto merge_kernel_size = json.value<std::vector<int64_t>>(
                "vision_config.merge_kernel_size");
            merge_kernel_size.has_value() && !merge_kernel_size->empty()) {
          return (*merge_kernel_size)[0];
        }
        return int64_t(2);
      });
  LOAD_ARG_OR_FUNC(mm_image_merge_size, "vision_config.image_merge_size", [&] {
    return args->mm_spatial_merge_size() > 0 ? args->mm_spatial_merge_size()
                                             : int64_t(2);
  });

  LOAD_ARG_OR_FUNC(mm_head_dim, "vision_config.head_dim", [&] {
    return args->mm_hidden_size() / args->mm_num_attention_heads();
  });
  // No explicit spatial_patch_size in Kimi-K2.5; fallback to patch_size.
  LOAD_ARG_OR_FUNC(mm_spatial_patch_size,
                   "vision_config.spatial_patch_size",
                   [&] { return args->mm_patch_size(); });

  // Original qwen2_5_vl vision args not found in Kimi-K2.5 vision_config.
  // Keep defaults and mark as unmapped:
  // mm_num_channels is required by patch_embed in_features (in_dim * patch^2).
  LOAD_ARG_OR(mm_num_channels, "vision_config.in_chans", 3);
  // LOAD_ARG_OR(mm_window_size, "vision_config.window_size", 112);
  // LOAD_ARG_OR(mm_fullatt_block_indexes,
  //             "vision_config.fullatt_block_indexes",
  //             std::vector<int64_t>({7, 15, 23, 31}));
  // LOAD_ARG_OR(mm_tokens_per_second, "vision_config.tokens_per_second", 2);
  // LOAD_ARG_OR(mm_temporal_patch_size, "vision_config.temporal_patch_size",
  // 2);

  // [Compared with qwen2_5_vl] qwen mrope args are absent in Kimi-K2.5 config.
  // LOAD_ARG(rope_scaling_mrope_section, "rope_scaling.mrope_section");

  // Add Args For Kimi-K2.5
  LOAD_ARG_OR(mm_init_pos_emb_time, "vision_config.init_pos_emb_time", 4);
  LOAD_ARG_OR(mm_init_pos_emb_width, "vision_config.init_pos_emb_width", 64);
  LOAD_ARG_OR(mm_init_pos_emb_height, "vision_config.init_pos_emb_height", 64);

  // New Kimi-K2.5 vision_config keys currently not mapped to ModelArgs:
  // - _attn_implementation
  // - merge_type
  // - pos_emb_type
  // - video_attn_type
  // New Kimi-K2.5 config keys not registered in ModelArgs:
  // - architectures
  // - aux_loss_alpha
  // - seq_aux
  // - transformers_version
  // - use_cache
  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({163585, 163586}));
});

REGISTER_TOKENIZER_ARGS(kimi_k25, [&] {
  // Kimi-K2.5 uses TikTokenTokenizer with tiktoken.model.
  // HF files:
  // - tokenizer_config.json (tokenizer_class=TikTokenTokenizer)
  // - tiktoken.model
  SET_ARG(tokenizer_type, "tiktoken");
  SET_ARG(vocab_file, "tiktoken.model");

  // ref:
  // https://huggingface.co/moonshotai/Kimi-K2.5/blob/main/tokenizer_config.json
  const std::vector<SpecialToken> special_tokens(
      {{"[BOS]", 163584},
       {"[EOS]", 163585},
       {"<|im_end|>", 163586},
       {"<|im_user|>", 163587},
       {"<|im_assistant|>", 163588},
       {"<|start_header_id|>", 163590},
       {"<|end_header_id|>", 163591},
       {"[EOT]", 163593},
       {"<|im_system|>", 163594},
       {"<|tool_calls_section_begin|>", 163595},
       {"<|tool_calls_section_end|>", 163596},
       {"<|tool_call_begin|>", 163597},
       {"<|tool_call_argument_begin|>", 163598},
       {"<|tool_call_end|>", 163599},
       {"<|im_middle|>", 163601},
       {"<|media_begin|>", 163602},
       {"<|media_content|>", 163603},
       {"<|media_end|>", 163604},
       {"<|media_pad|>", 163605},
       {"<think>", 163606},
       {"</think>", 163607},
       {"[UNK]", 163838},
       {"[PAD]", 163839}});
  SET_ARG(special_tokens, special_tokens);

  // Keep parser tokens visible in decoded output instead of stripping them.
  const std::vector<std::string> parser_visible_special_tokens(
      {"<|tool_calls_section_begin|>",
       "<|tool_calls_section_end|>",
       "<|tool_call_begin|>",
       "<|tool_call_argument_begin|>",
       "<|tool_call_end|>",
       "<think>",
       "</think>"});
  SET_ARG(visible_special_tokens, parser_visible_special_tokens);

  // ref:
  // https://huggingface.co/moonshotai/Kimi-K2.5/blob/main/tokenization_kimi.py#L53-L62
  // N.B. re2 doesn't support character class intersection (&&) or subtraction
  // (--). Since [\p{Han}]+ is the first branch and matches all Han characters
  // first, we can safely remove the '&&[^\p{Han}]' part from subsequent
  // branches. N.B. replaced '\s+(?!\S)' with '\s+[^\s]' - re2 doesn't support
  // negative lookahead.
  const std::string pattern_str =
      R"([\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+[^\s]|\s+)";
  SET_ARG(pattern, pattern_str);
});

using MiniCPMVMultimodalProcessor =
    MultimodalProcessor<MiniCPMPromptProcessor, MiniCPMVImageProcessor>;

REGISTER_MULTIMODAL_PROCESSOR(minicpmv, MiniCPMVMultimodalProcessor);

REGISTER_MODEL_ARGS(minicpmv, [&] {
  // text config
  LOAD_ARG_OR(model_type, "model_type", "minicpmv");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vision_custom_adapter, "vision_adapter_type", "");
  LOAD_ARG_OR(vision_max_slice_nums, "slice_config.max_slice_nums", 9);
  LOAD_ARG_OR(hidden_size, "hidden_size", 3584);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 28);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 28);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 18944);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 32768);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-06);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 151643);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 151645);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);
  LOAD_ARG_OR(rope_scaling_factor, "rope_scaling_factor", 1.0f);

  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(sliding_window, "sliding_window", 131072);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 28);
  LOAD_ARG_OR(query_num, "query_num", 64);
  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });
  LOAD_ARG_OR(vocab_size, "vocab_size", 151666);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 4);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");

  LOAD_ARG_OR(mm_hidden_size, "vision_config.hidden_size", 1152);
  LOAD_ARG_OR(mm_image_size, "vision_config.image_size", 980);
  LOAD_ARG_OR(mm_intermediate_size, "vision_config.intermediate_size", 4304);
  LOAD_ARG_OR(mm_num_attention_heads, "vision_config.num_attention_heads", 16);
  LOAD_ARG_OR(mm_num_hidden_layers, "vision_config.num_hidden_layers", 27);
  LOAD_ARG_OR(mm_patch_size, "vision_config.patch_size", 14);
  LOAD_ARG_OR(mm_num_channels, "vision_config.num_channels", 3);
  LOAD_ARG_OR(mm_dropout, "attention_dropout", 0.0);
  LOAD_ARG_OR(mm_hidden_act, "vision_config.hidden_act", "gelu_pytorch_tanh");
  LOAD_ARG_OR(mm_layer_norm_eps, "vision_config.layer_norm_eps", 1e-06);
  LOAD_ARG_OR_FUNC(mm_head_dim, "mm_head_dim", [&] {
    return args->mm_hidden_size() / args->mm_num_attention_heads();
  });
});

using Mistral3MultimodalProcessor =
    MultimodalProcessor<Mistral3PromptProcessor, Mistral3ImageProcessor>;

REGISTER_MODEL_ARGS(mistral3, [&] {
  // Text config (from top-level JSON)
  LOAD_ARG_OR(model_type, "model_type", "mistral3");
  LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");
  LOAD_ARG_OR(vocab_size, "vocab_size", 131072);
  LOAD_ARG_OR(hidden_size, "hidden_size", 5120);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 32768);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 40);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 32);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 8);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 131072);
  LOAD_ARG_OR(head_dim, "head_dim", 128);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-5);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1e9);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 1);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 2);

  // Vision config (Pixtral ViT, from vision_config.* in JSON)
  LOAD_ARG_OR(mm_hidden_size, "vision_config.hidden_size", 1024);
  LOAD_ARG_OR(mm_intermediate_size, "vision_config.intermediate_size", 4096);
  LOAD_ARG_OR(mm_num_attention_heads, "vision_config.num_attention_heads", 16);
  LOAD_ARG_OR(mm_num_hidden_layers, "vision_config.num_hidden_layers", 24);
  LOAD_ARG_OR(mm_num_channels, "vision_config.num_channels", 3);
  LOAD_ARG_OR(mm_patch_size, "vision_config.patch_size", 14);
  LOAD_ARG_OR(mm_image_size, "vision_config.image_size", 1540);
  LOAD_ARG_OR(mm_projection_dim, "text_config.hidden_size", 5120);
  LOAD_ARG_OR(mm_spatial_merge_size, "spatial_merge_size", 2);

  LOAD_ARG_OR_FUNC(mm_head_dim, "vision_config.head_dim", [&] {
    return args->mm_hidden_size() / args->mm_num_attention_heads();
  });

  // Multimodal token IDs
  LOAD_ARG_OR(image_token_id, "image_token_index", 10);

  // Image processor config (from preprocessor_config.json)
  LOAD_ARG_OR(mm_image_do_normalize, "do_normalize", true);
  LOAD_ARG_OR(mm_image_do_rescale, "do_rescale", true);
  LOAD_ARG_OR(mm_image_do_resize, "do_resize", true);
});

REGISTER_MULTIMODAL_PROCESSOR(mistral3, Mistral3MultimodalProcessor);

using OxygenVLMultimodalProcessor = MultimodalProcessor<Qwen2VLPromptProcessor,
                                                        Qwen2VLImageProcessor,
                                                        Qwen2VLVideoProcessor>;

REGISTER_MULTIMODAL_PROCESSOR(oxygenvlm, OxygenVLMultimodalProcessor);

REGISTER_MPOSITION_GENERATOR(oxygenvlm, xllm::QwenVLMPositionGenerator);

REGISTER_MODEL_ARGS(oxygenvlm, [&] {
  LOAD_ARG_OR(model_type, "model_type", "oxygenvlm");
  LOAD_ARG_OR(vision_start_token_id, "vision_start_token_id", 151652);
  LOAD_ARG_OR(vision_end_token_id, "vision_end_token_id", 151653);
  LOAD_ARG_OR(vision_token_id, "vision_token_id", 151654);
  LOAD_ARG_OR(video_token_id, "video_token_id", 151656);
  LOAD_ARG_OR(image_token_id, "image_token_id", 151655);

  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);

  // text config
  LOAD_ARG_OR(vocab_size, "text_config.vocab_size", 151936);
  LOAD_ARG_OR(eos_token_id, "text_config.eos_token_id", 151645);
  LOAD_ARG_OR(attention_bias, "text_config.attention_bias", false);
  LOAD_ARG_OR(attention_dropout, "text_config.attention_dropout", 0.0f);
  LOAD_ARG_OR(hidden_act, "text_config.hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "text_config.hidden_size", 5120);
  LOAD_ARG_OR(initializer_range, "text_config.initializer_range", 0.02);
  LOAD_ARG_OR(intermediate_size, "text_config.intermediate_size", 25600);
  LOAD_ARG_OR(
      max_position_embeddings, "text_config.max_position_embeddings", 40960);
  LOAD_ARG_OR(n_heads, "text_config.num_attention_heads", 64);
  LOAD_ARG_OR(head_dim, "text_config.head_dim", 128);

  LOAD_ARG_OR(n_layers, "text_config.num_hidden_layers", 64);
  LOAD_ARG_OR(n_kv_heads, "text_config.num_key_value_heads", 8);
  LOAD_ARG_OR(rms_norm_eps, "text_config.rms_norm_eps", 1e-05);
  LOAD_ARG_OR(dtype, "text_config.dtype", "bfloat16");
  LOAD_ARG_OR(rope_scaling_rope_type, "text_config.rope_scaling.type", "mrope");
  LOAD_ARG(rope_scaling_mrope_section,
           "text_config.rope_scaling.mrope_section");
  LOAD_ARG_OR(rope_theta, "text_config.rope_theta", 1000000);

  // vision config
  LOAD_ARG_OR(mm_num_hidden_layers, "vision_config.depth", 24);
  LOAD_ARG_OR(mm_hidden_act, "vision_config.hidden_act", "silu");
  LOAD_ARG_OR(mm_hidden_size, "vision_config.hidden_size", 1536);
  LOAD_ARG_OR(mm_image_size, "vision_config.image_size", 336);
  LOAD_ARG_OR(mm_num_channels, "vision_config.in_channels", 3);
  LOAD_ARG_OR(
      mm_intermediate_size, "vision_config.projector_hidden_size", 4096);
  LOAD_ARG_OR(mm_num_attention_heads, "vision_config.num_heads", 12);
  LOAD_ARG_OR(mm_projection_dim, "vision_config.out_hidden_size", 5120);
  LOAD_ARG_OR(mm_patch_size, "vision_config.patch_size", 14);
  LOAD_ARG_OR(mm_spatial_merge_size, "vision_config.spatial_merge_size", 2);
  LOAD_ARG_OR(mm_temporal_patch_size, "vision_config.temporal_patch_size", 2);

  // Vision-scoped key so JsonReader's text_config fallback cannot resolve
  // "head_dim" against text_config.head_dim. vision_config has no head_dim;
  // the fallback lambda derives it from mm_hidden_size /
  // mm_num_attention_heads.
  LOAD_ARG_OR_FUNC(mm_head_dim, "vision_config.head_dim", [&] {
    return args->mm_hidden_size() / args->mm_num_attention_heads();
  });
  if (args->rope_scaling_rope_type() == "default")
    args->rope_scaling_rope_type() = "mrope";
});

using Qwen3VLMoeMultimodalProcessor =
    MultimodalProcessor<Qwen3VLPromptProcessor,
                        Qwen2VLImageProcessor,
                        Qwen3VLVideoProcessor>;

REGISTER_MULTIMODAL_PROCESSOR(qwen3_vl_moe, Qwen3VLMoeMultimodalProcessor);

REGISTER_MPOSITION_GENERATOR(qwen3_vl_moe, xllm::Qwen3VLMPositionGenerator);

REGISTER_MODEL_ARGS(qwen3_vl_moe, [&] {
  // text config
  LOAD_ARG_OR(model_type, "model_type", "qwen3_vl_moe");
  LOAD_ARG_OR(attention_bias, "text_config.attention_bias", false);
  LOAD_ARG_OR(attention_dropout, "attention_dropout", 0.0f);
  LOAD_ARG_OR(bos_token_id, "text_config.bos_token_id", 151643);
  LOAD_ARG_OR(decoder_sparse_step, "text_config.decoder_sparse_step", 1);
  LOAD_ARG_OR(dtype, "text_config.dtype", "bfloat16");
  LOAD_ARG_OR(eos_token_id, "text_config.eos_token_id", 151645);
  LOAD_ARG_OR_FUNC(head_dim, "text_config.head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });
  LOAD_ARG_OR(hidden_act, "text_config.hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "text_config.hidden_size", 2048);
  LOAD_ARG_OR(initializer_range, "text_config.initializer_range", 0.02);
  LOAD_ARG_OR(intermediate_size, "text_config.intermediate_size", 5632);
  LOAD_ARG_OR(
      max_position_embeddings, "text_config.max_position_embeddings", 128000);
  // LOAD_ARG(mlp_only_layers, "text_config.mlp_only_layers");
  LOAD_ARG_OR(moe_intermediate_size, "text_config.moe_intermediate_size", 1408);
  LOAD_ARG_OR(norm_topk_prob, "text_config.norm_topk_prob", true);
  LOAD_ARG_OR(n_heads, "text_config.num_attention_heads", 16);
  LOAD_ARG_OR(num_experts, "text_config.num_experts", 128);
  LOAD_ARG_OR(num_experts_per_tok, "text_config.num_experts_per_tok", 8);
  LOAD_ARG_OR(n_layers, "text_config.num_hidden_layers", 24);
  LOAD_ARG_OR(n_kv_heads, "text_config.num_key_value_heads", 16);
  LOAD_ARG_OR(rms_norm_eps, "text_config.rms_norm_eps", 1e-06);
  LOAD_ARG_OR(rope_scaling_rope_type, "text_config.rope_scaling.type", "mrope");
  LOAD_ARG(rope_scaling_mrope_section,
           "text_config.rope_scaling.mrope_section");
  // LOAD_ARG_OR(rope_scaling_mrope_interleaved,"text_config.rope_scaling.mrope_interleaved",true);
  LOAD_ARG_OR(rope_theta, "text_config.rope_theta", 5000000.0f);
  LOAD_ARG_OR(vocab_size, "text_config.vocab_size", 151936);

  // vision config
  LOAD_ARG(mm_deepstack_visual_indexes,
           "vision_config.deepstack_visual_indexes");
  LOAD_ARG_OR(mm_num_hidden_layers, "vision_config.depth", 27);
  LOAD_ARG_OR(mm_hidden_act, "vision_config.hidden_act", "gelu_pytorch_tanh");
  LOAD_ARG_OR(mm_hidden_size, "vision_config.hidden_size", 1152);
  LOAD_ARG_OR(mm_num_channels, "vision_config.in_channels", 3);
  LOAD_ARG_OR(mm_initializer_range, "vision_config.initializer_range", 0.02);
  LOAD_ARG_OR(mm_intermediate_size, "vision_config.intermediate_size", 4304);
  LOAD_ARG_OR(mm_num_attention_heads, "vision_config.num_heads", 16);
  LOAD_ARG_OR(mm_num_position_embeddings,
              "vision_config.num_position_embeddings",
              2304);
  LOAD_ARG_OR(mm_projection_dim, "vision_config.out_hidden_size", 3584);
  LOAD_ARG_OR(mm_patch_size, "vision_config.patch_size", 16);
  LOAD_ARG_OR(mm_spatial_merge_size, "vision_config.spatial_merge_size", 2);
  LOAD_ARG_OR(mm_temporal_patch_size, "vision_config.temporal_patch_size", 2);
  // Read the vision-scoped key so the top-level fallback in JsonReader does
  // not resolve "head_dim" against text_config: Qwen3-VL keeps the text
  // head_dim (128) under text_config and derives the vision head_dim from
  // vision_config.hidden_size / vision_config.num_heads (1152 / 16 = 72).
  LOAD_ARG_OR_FUNC(mm_head_dim, "vision_config.head_dim", [&] {
    return args->mm_hidden_size() / args->mm_num_attention_heads();
  });

  LOAD_ARG_OR(image_token_id, "image_token_id", 151655);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_ARG_OR(video_token_id, "video_token_id", 151656);
  LOAD_ARG_OR(vision_end_token_id, "vision_end_token_id", 151653);
  LOAD_ARG_OR(vision_start_token_id, "vision_start_token_id", 151652);
});

}  // namespace xllm::npu::model

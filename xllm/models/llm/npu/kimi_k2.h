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

#pragma once

#include "deepseek_v2.h"

namespace xllm::npu::model {
// register the causal model
REGISTER_CAUSAL_MODEL(kimi_k2, DeepseekV2ForCausalLM);
// register the model args
// example config:
// example config:
// https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/config.json
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
}  // namespace xllm::npu::model

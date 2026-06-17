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

#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/scheduler_config.h"
#include "deepseek_v2.h"

namespace xllm {

class DeepseekV3ForCausalLMImpl
    : public LlmForCausalLMImplBase<DeepseekV2Model> {
 public:
  DeepseekV3ForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<DeepseekV2Model>(context) {
    // Check if prefix cache or chunked prefill is enabled for unsupported
    // models
    CHECK(!::xllm::KVCacheConfig::get_instance().enable_prefix_cache())
        << "deepseek_v3 have not supported "
           "enable_prefix_cache yet. Please disable it.";
    CHECK(!::xllm::SchedulerConfig::get_instance().enable_chunked_prefill())
        << "deepseek_v3 have not supported "
           "enable_chunked_prefill yet. Please disable it.";
  }
};
TORCH_MODULE(DeepseekV3ForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL(deepseek_v3, DeepseekV3ForCausalLM);
// register the model args
// example config:
// https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json
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
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
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
}  // namespace xllm

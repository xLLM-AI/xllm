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

#include "core/framework/model_loader/model_loader.h"
#include "models/model_registry.h"

namespace xllm {

// Compute is provided by the Python executor. Do not route this checkpoint to
// the Qwen/GQA DSpark native factory or the quantized GLM target model.
REGISTER_MODEL_ARGS(glm5_dspark, [&] {
  LOAD_ARG_OR(model_type, "model_type", "glm5_dspark");
  LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");
  LOAD_ARG_OR(vocab_size, "vocab_size", 154880);
  LOAD_ARG_OR(hidden_size, "hidden_size", 6144);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 5);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 64);
  SET_ARG(n_kv_heads, 1);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 12288);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 1048576);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-5);
  LOAD_ARG_OR(rope_theta, "rope_parameters.rope_theta", 8000000.0f);
  LOAD_ARG_OR(rope_theta, "rope_theta", args->rope_theta());
  LOAD_ARG_OR(q_lora_rank, "q_lora_rank", 2048);
  LOAD_ARG_OR(kv_lora_rank, "kv_lora_rank", 512);
  LOAD_ARG_OR(qk_nope_head_dim, "qk_nope_head_dim", 192);
  LOAD_ARG_OR(qk_rope_head_dim, "qk_rope_head_dim", 64);
  LOAD_ARG_OR(v_head_dim, "v_head_dim", 256);
  SET_ARG(head_dim, args->qk_nope_head_dim() + args->qk_rope_head_dim());
  SET_ARG(rotary_dim, args->qk_rope_head_dim());
  SET_ARG(index_topk, 0);
  SET_ARG(num_nextn_predict_layers, 0);
  SET_ARG(use_sliding_window, false);
  SET_ARG(sliding_window, -1);
  SET_ARG(tie_word_embeddings, false);
  LOAD_ARG_OR(markov_rank, "markov_rank", 256);
  LOAD_ARG_OR(enable_confidence_head, "enable_confidence_head", false);
  LOAD_ARG_OR(
      confidence_head_with_markov, "confidence_head_with_markov", false);
});

}  // namespace xllm

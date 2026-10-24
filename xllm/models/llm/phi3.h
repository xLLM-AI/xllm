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

#include <cmath>
#include <cstdint>
#include <unordered_set>

#include "models/model_registry.h"

namespace xllm {

// Phi-3 executes through PyCausalLM; no native causal-model factory is needed.
REGISTER_MODEL_BACKEND(phi3, "llm");

REGISTER_MODEL_ARGS_LOADER(phi3, [](const JsonReader& json, ModelArgs* args) {
  LOAD_ARG_OR(model_type, "model_type", "phi3");
  LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");
  LOAD_ARG_OR(dtype, "dtype", args->dtype());
  LOAD_ARG_OR(vocab_size, "vocab_size", 32064);
  LOAD_ARG_OR(hidden_size, "hidden_size", 3072);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 32);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 32);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", args->n_heads());
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 8192);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 4096);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-5);
  LOAD_ARG_OR(rope_theta, "rope_theta", 10000.0f);
  LOAD_ARG_OR(sliding_window, "sliding_window", -1);
  LOAD_ARG_OR(attention_bias, "attention_bias", false);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 1);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 32000);
  LOAD_ARG_OR(pad_token_id, "pad_token_id", 32000);

  if (args->hidden_size() <= 0 || args->n_heads() <= 0 ||
      args->n_layers() <= 0 || args->intermediate_size() <= 0 ||
      args->vocab_size() <= 0 || args->max_position_embeddings() <= 0 ||
      args->n_kv_heads().value() <= 0 ||
      args->hidden_size() % args->n_heads() != 0 ||
      args->n_heads() % args->n_kv_heads().value() != 0) {
    LOG(ERROR) << "Invalid Phi-3 model dimensions";
    return false;
  }
  LOAD_ARG_OR(head_dim, "head_dim", args->hidden_size() / args->n_heads());
  if (args->head_dim() != args->hidden_size() / args->n_heads() ||
      args->head_dim() % 2 != 0 || args->hidden_act() != "silu" ||
      args->attention_bias() || args->tie_word_embeddings() ||
      !std::isfinite(args->rms_norm_eps()) || args->rms_norm_eps() <= 0 ||
      !std::isfinite(args->rope_theta()) || args->rope_theta() <= 0 ||
      args->sliding_window() == 0 || args->sliding_window() < -1) {
    LOG(ERROR) << "Unsupported Phi-3 mini 4K configuration";
    return false;
  }
  const auto* rope_scaling = json.resolve("rope_scaling");
  if (rope_scaling != nullptr && !rope_scaling->is_null()) {
    LOG(ERROR) << "Phi-3 mini 4K does not support scaled RoPE";
    return false;
  }
  args->use_sliding_window() = args->sliding_window() > 0;
  args->eos_token_id_vec() = {args->eos_token_id()};
  // mini-4k generation_config also stops on <|assistant|> and <|end|>.
  args->stop_token_ids() =
      std::unordered_set<int32_t>({args->eos_token_id(), 32001, 32007});
  return true;
});

}  // namespace xllm

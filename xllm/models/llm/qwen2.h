/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "core/layers/qwen2_decoder_layer.h"
#include "llm_model_base.h"

namespace xllm {

class QWen2ModelImpl : public LlmModelImplBase<layer::Qwen2DecoderLayer> {
 public:
  QWen2ModelImpl(const ModelContext& context)
      : LlmModelImplBase<layer::Qwen2DecoderLayer>("qwen2",
                                                   context.get_model_args()) {
    // register submodules
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    if (!mrope_section_.empty()) {
      cos_sin_ = layer::rotary::get_concat_rotary_embedding(
          model_args.hidden_size() / model_args.n_heads(),
          model_args.max_position_embeddings(),
          model_args.rope_theta(),
          options);
    }

    layers_.reserve(model_args.n_layers());
    norm_ = register_module("norm", layer::RMSNorm(context));
    embed_tokens_ =
        register_module("embed_tokens", layer::WordEmbedding(context));

    for (int32_t i = 0; i < model_args.n_layers(); i++) {
      auto layer = layer::Qwen2DecoderLayer(context, i);
      layers_.push_back(layer);
    }
  }
  std::pair<torch::Tensor, torch::Tensor> apply_mrope(
      const torch::Tensor positions) override {
    auto target_cos_sin = cos_sin_.index({positions});
    auto target_cos_sin_chunks = target_cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
    auto cos_pos = target_cos_sin_chunks[0].contiguous();
    auto sin_pos = target_cos_sin_chunks[1].contiguous();
    auto apply = [this](torch::Tensor x) {
      auto sections = mrope_section_;
      sections.insert(sections.end(), sections.begin(), sections.end());

      auto vec = x.split(sections, -1);
      std::vector<torch::Tensor> selects;
      selects.reserve(vec.size());

      for (int64_t i = 0; i < vec.size(); ++i) {
        auto m = vec[i];
        selects.push_back(m[i % mrope_section_.size()]);
      }
      return torch::cat(selects, -1);
    };
    cos_pos = apply(cos_pos.reshape({positions.size(0), -1, cos_pos.size(-1)}));
    sin_pos = apply(sin_pos.reshape({positions.size(0), -1, sin_pos.size(-1)}));
    return std::make_pair(cos_pos, sin_pos);
  }
};
TORCH_MODULE(QWen2Model);

class QWen2ForCausalLMImpl : public LlmForCausalLMImplBase<QWen2Model> {
 public:
  QWen2ForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<QWen2Model>(context) {}
};
TORCH_MODULE(QWen2ForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL(qwen2, QWen2ForCausalLM);

// register the model args
// example config:
// https://huggingface.co/Qwen/Qwen2-7B-Instruct/blob/main/config.json
REGISTER_MODEL_ARGS(qwen2, [&] {
  LOAD_ARG_OR(model_type, "model_type", "qwen2");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 152064);
  LOAD_ARG_OR(hidden_size, "hidden_size", 3584);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 28);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 28);
  LOAD_ARG(n_kv_heads, "num_key_value_heads");
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(attention_bias, "attention_bias", true);
  // LOAD_ARG_OR(no_bias, "no_bias", true);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 18944);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 32768);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 151643);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);

  // For Qwen2/2.5 model < 7B,  tie_word_embeddings = true
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);

  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR_FUNC(sliding_window, "sliding_window", [&] {
    return args->max_position_embeddings();
  });
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 28);

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});

}  // namespace xllm

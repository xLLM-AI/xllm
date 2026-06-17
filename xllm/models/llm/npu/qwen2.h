/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "core/framework/config/scheduler_config.h"
#include "core/layers/npu/npu_qwen2_decoder_layer_impl.h"
#include "layers/common/rotary_embedding_util.h"
#include "llm_model_base.h"

// QWen2 model compatible with huggingface weights
// ref to:
// https://github.com/huggingface/transformers/blob/v4.43.3/src/transformers/models/qwen2/modeling_qwen2.py
namespace xllm::npu::model {

class QWen2DecoderLayerImpl
    : public LlmDecoderLayerImplBase<layer::NpuQwen2DecoderLayer> {
 public:
  QWen2DecoderLayerImpl(const ModelContext& context, const int32_t layer_id)
      : LlmDecoderLayerImplBase<layer::NpuQwen2DecoderLayer>(context,
                                                             layer_id) {}
};
TORCH_MODULE(QWen2DecoderLayer);

class QWen2ModelImpl : public LlmModelImplBase<QWen2DecoderLayer> {
 public:
  QWen2ModelImpl(const ModelContext& context)
      : LlmModelImplBase<QWen2DecoderLayer>("qwen2", context.get_model_args()) {
    // register submodules
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    auto parallel_args = context.get_parallel_args();
    auto dp_local_tp_size =
        parallel_args.world_size() / parallel_args.dp_size();
    dp_rank_ = parallel_args.rank() / dp_local_tp_size;

    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(model_args.n_layers());
    norm_ = register_module("norm", layer::NpuRMSNorm(context));
    npu_embed_tokens_ =
        register_module("npu_embed_tokens", layer::NpuWordEmbedding(context));
    atb_pos_emb_ = layer::NpuPosEmbedding(context);
    cos_sin_ = layer::rotary::get_concat_rotary_embedding(
        model_args.hidden_size() / model_args.n_heads(),
        model_args.max_position_embeddings(),
        model_args.rope_theta(),
        options);
    int32_t mask_value =
        ::xllm::SchedulerConfig::get_instance().enable_chunked_prefill() ? -9984
                                                                         : 1;
    attn_mask_ = layer::AttentionMask(options.device(),
                                      options.dtype().toScalarType(),
                                      /*mask_value=*/mask_value);

    for (int32_t i = 0; i < model_args.n_layers(); i++) {
      auto block = QWen2DecoderLayer(context, i);
      layers_.push_back(block);
      blocks_->push_back(block);
    }
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
  LOAD_ARG_OR(sliding_window, "sliding_window", 4096);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 28);

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});

}  // namespace xllm::npu::model

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

#include "core/framework/config/scheduler_config.h"
#include "core/framework/model/model_output.h"
#include "core/layers/common/rotary_embedding_util.h"
#include "core/layers/npu/npu_glm4_decoder_layer_impl.h"
#include "llm_model_base.h"

namespace xllm::npu::model {

class Glm4DecoderLayerImpl : public xllm::npu::model::LlmDecoderLayerImplBase<
                                 layer::NpuGlm4DecoderLayer> {
 public:
  Glm4DecoderLayerImpl(const ModelContext& context, const int32_t layer_id)
      : xllm::npu::model::LlmDecoderLayerImplBase<layer::NpuGlm4DecoderLayer>(
            context,
            layer_id) {}
};
TORCH_MODULE(Glm4DecoderLayer);

class Glm4ModelImpl
    : public xllm::npu::model::LlmModelImplBase<Glm4DecoderLayer> {
 public:
  Glm4ModelImpl(const ModelContext& context)
      : xllm::npu::model::LlmModelImplBase<Glm4DecoderLayer>(
            "glm4",
            context.get_model_args()) {
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
    cos_sin_ = layer::rotary::get_chatglm_rotary_embedding(
        64,
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
      auto block = Glm4DecoderLayer(context, i);
      layers_.push_back(block);
      blocks_->push_back(block);
    }
  }

  virtual ModelOutput forward(torch::Tensor tokens,
                              torch::Tensor positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelInputParams& input_params) {
    ModelInputParams& input_params_new =
        const_cast<ModelInputParams&>(input_params);

    if (tokens.numel() == 0) {
      tokens = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
      positions = torch::tensor({0}).to(torch::kInt32).to(tokens.device());
    }
    auto inputs_embeds = input_params.embedding.input_embedding;
    torch::Tensor h;
    if (inputs_embeds.defined()) {
      h = inputs_embeds;
    } else {
      h = npu_embed_tokens_(tokens, 0);
    }
    auto target_cos_sin = atb_pos_emb_(cos_sin_, positions, 0);
    auto target_cos_sin_chunks = target_cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
    auto cos_pos = target_cos_sin_chunks[0].contiguous();

    auto sin_pos = target_cos_sin_chunks[1].contiguous();

    if (positions.dim() == 2) {  // mrope
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
      cos_pos = apply(cos_pos.reshape(
          {positions.sizes().front(), -1, cos_pos.sizes().back()}));
      sin_pos = apply(sin_pos.reshape(
          {positions.sizes().front(), -1, sin_pos.sizes().back()}));
    }
    cos_pos = cos_pos.reshape({-1, cos_pos.sizes().back() / 2, 2});
    sin_pos = sin_pos.reshape({-1, sin_pos.sizes().back() / 2, 2});
    torch::Tensor attn_mask;
    if (::xllm::SchedulerConfig::get_instance().enable_chunked_prefill()) {
      int max_kv_seq = input_params.meta.kv_max_seq_len;
      int num_sequences = input_params.meta.num_sequences;
      if (num_sequences > 0) {
        std::vector<torch::Tensor> req_mask_vec;
        req_mask_vec.reserve(num_sequences);

        for (int j = 0; j < num_sequences; j++) {
          auto mask = attn_mask_.gen_append_mask(
              input_params.attention.host.q_seq_lens[j],
              input_params.attention.host.kv_seq_lens[j],
              max_kv_seq,
              cos_pos.dtype().toScalarType(),
              cos_pos.device());
          req_mask_vec.emplace_back(mask);
        }
        attn_mask = torch::cat(req_mask_vec, 0);
      }
    } else if (input_params.meta.batch_forward_type.is_prefill()) {
      attn_mask = attn_mask_.get_attn_mask(
          128, cos_pos.dtype().toScalarType(), cos_pos.device());
    }

    for (size_t i = 0; i < layers_.size(); i++) {
      aclrtEvent* event{nullptr};
      std::atomic<bool>* event_flag{nullptr};

      if (input_params.parallel.layer_synchronizer != nullptr) {
        event = input_params.parallel.layer_synchronizer->get_event(i);
        event_flag =
            input_params.parallel.layer_synchronizer->get_event_flag(i);
      }
      if (!input_params.synchronize_layer(i)) {
        return ModelOutput();
      }

      auto& layer = layers_[i];
      layer(h,
            cos_pos,
            sin_pos,
            attn_mask,
            kv_caches[i],
            input_params_new,
            event,
            event_flag);
    }
    auto hidden_states = norm_(h, 0);
    return ModelOutput(hidden_states);
  }

 private:
  torch::Tensor viusal_pos_mask_;
};
TORCH_MODULE(Glm4Model);

class Glm4ForCausalLMImpl
    : public xllm::npu::model::LlmForCausalLMImplBase<Glm4Model> {
 public:
  Glm4ForCausalLMImpl(const ModelContext& context)
      : xllm::npu::model::LlmForCausalLMImplBase<Glm4Model>(context) {}
};
TORCH_MODULE(Glm4ForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL(glm4, Glm4ForCausalLM);

// register the model args
REGISTER_MODEL_ARGS(glm4, [&] {
  LOAD_ARG_OR(model_type, "model_type", "glm4");

  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(attention_bias, "attention_bias", true);
  LOAD_ARG_OR(attention_dropout, "attention_dropout", 0.0f);
  LOAD_ARG_OR(eos_token_id_vec, "eos_token_id", std::vector<int>{151329});
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

}  // namespace xllm::npu::model

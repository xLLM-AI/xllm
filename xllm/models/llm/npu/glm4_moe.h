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
#include "core/framework/model_context.h"
#include "core/framework/parallel_state/npu_dp_ep_padding.h"
#include "core/layers/npu/npu_glm4_moe_decoder_layer.h"
#include "llm_model_base.h"

namespace xllm::npu::model {

using torch::indexing::None;
using ISlice = torch::indexing::Slice;

class Glm4MoeDecoderLayerImpl : public torch::nn::Module {
 public:
  Glm4MoeDecoderLayerImpl(const ModelContext& context, const int32_t i) {
    // register submodules
    decoder_layer_ =
        register_module("decoder_layer", layer::NpuGlm4MoeDecoder(context, i));
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor cos_pos,
                        torch::Tensor sin_pos,
                        torch::Tensor attn_mask,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params,
                        aclrtEvent* event,
                        std::atomic<bool>* event_flag) {
    return decoder_layer_(x,
                          cos_pos,
                          sin_pos,
                          attn_mask,
                          kv_cache,
                          input_params,
                          event,
                          event_flag);
  }

  void load_state_dict(const StateDict& state_dict) {
    decoder_layer_->load_state_dict(state_dict);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    decoder_layer_->verify_loaded_weights(prefix);
  }

  void merge_loaded_weights() { decoder_layer_->merge_loaded_weights(); }

  void merge_and_move_pinned_host() {
    decoder_layer_->merge_and_move_pinned_host();
  }

  void free_weights() { decoder_layer_->free_weights(); }

  void reload_weights() { decoder_layer_->reload_weights(); }

  void reload_weights_from_device() {
    decoder_layer_->reload_weights_from_device();
  }

  layer::BaseLoader* get_loader() { return decoder_layer_->get_loader(); }

  void refresh_rolling_weights() { decoder_layer_->refresh_rolling_weights(); }

 private:
  layer::NpuGlm4MoeDecoder decoder_layer_{nullptr};
};
TORCH_MODULE(Glm4MoeDecoderLayer);

class Glm4MoeModelImpl : public torch::nn::Module {
 public:
  Glm4MoeModelImpl(const ModelContext& context)
      : device_(context.get_tensor_options().device()) {
    auto options = context.get_tensor_options();
    auto model_args = context.get_model_args();
    auto parallel_args = context.get_parallel_args();

    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(model_args.n_layers());
    // register submodules
    device_ = options.device();
    dtype_ = options.dtype().toScalarType();
    num_speculative_tokens_ = model_args.num_speculative_tokens();
    npu_embed_tokens_ =
        register_module("npu_embed_tokens", layer::NpuWordEmbedding(context));

    atb_pos_emb_ = layer::NpuPosEmbedding(context);
    cos_sin_ = layer::rotary::get_concat_rotary_embedding(
        64,
        model_args.max_position_embeddings(),
        model_args.rope_theta(),
        options);
    mrope_section_ = model_args.rope_scaling_mrope_section();

    // int32_t mask_value = model_args.dtype() == "bfloat16" ? 1 : -9984;
    int32_t mask_value =
        ::xllm::SchedulerConfig::get_instance().enable_chunked_prefill() ? -9984
                                                                         : 1;
    attn_mask_ = layer::AttentionMask(options.device(),
                                      options.dtype().toScalarType(),
                                      /*mask_value=*/mask_value);
    for (int32_t i = 0; i < model_args.n_layers(); ++i) {
      auto block = Glm4MoeDecoderLayer(context, i);
      layers_.push_back(block);
      blocks_->push_back(block);
    }

    norm_ = register_module("norm", layer::NpuRMSNorm(context));
    dp_size_ = parallel_args.dp_size();
    std::vector<int64_t> indices;
    dp_local_tp_size_ = parallel_args.world_size() / dp_size_;
    dp_rank_ = parallel_args.rank() / dp_local_tp_size_;
    rank_ = parallel_args.rank();
    mapping_data_ = parallel_args.mapping_data();
    num_experts_per_tok_ = model_args.num_experts_per_tok();
    for (int i = 0; i < parallel_args.world_size(); i += dp_local_tp_size_) {
      indices.push_back(i);
    }
  }

  torch::Tensor get_input_embeddings(torch::Tensor input_ids) {
    return npu_embed_tokens_(input_ids, 0);
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    if (dp_size_ > 1) {
      if (tokens.sizes() == 0) {
        tokens = torch::tensor({1}).to(torch::kInt32).to(device_);
        positions = torch::tensor({0}).to(torch::kInt32).to(device_);
      }
    }

    auto inputs_embeds = input_params.embedding.input_embedding;
    torch::Tensor h;
    if (inputs_embeds.defined()) {
      h = inputs_embeds;
    } else {
      h = npu_embed_tokens_(tokens, 0);
    }
    int64_t input_length = tokens.size(0);
    torch::Tensor expert_array = torch::arange(
        0,
        input_length * num_experts_per_tok_,
        torch::TensorOptions().dtype(torch::kInt32).device(tokens.device()));
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
    cos_pos = cos_pos.view(at::IntArrayRef{-1, 2, cos_pos.size(-1) / 2});
    sin_pos = sin_pos.view(at::IntArrayRef{-1, 2, sin_pos.size(-1) / 2});
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
      attn_mask = attn_mask_.get_attn_mask(128, dtype_, device_);
    }

    ModelInputParams& input_params_new =
        const_cast<ModelInputParams&>(input_params);
    input_params_new.expert.expert_array = expert_array;

    RollingLayerGuard rolling_guard(rolling_mgr_);
    for (size_t i = 0; i < layers_.size(); i++) {
      aclrtEvent* event = nullptr;
      std::atomic<bool>* event_flag = nullptr;
      if (input_params.parallel.layer_synchronizer != nullptr) {
        event = input_params.parallel.layer_synchronizer->get_event(i);
        event_flag =
            input_params.parallel.layer_synchronizer->get_event_flag(i);
      }
      if (!input_params.synchronize_layer(i)) {
        return ModelOutput();
      }

      auto& layer = layers_[i];
      const int32_t layer_index = i;
      rolling_guard.before_layer(layer_index);
      layer(h,
            cos_pos,
            sin_pos,
            attn_mask,
            kv_caches[i],
            input_params_new,
            event,
            event_flag);
      rolling_guard.after_layer(layer_index);
    }
    auto hidden_states = norm_(h, 0);
    return ModelOutput(hidden_states);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    npu_embed_tokens_->load_state_dict(
        state_dict.get_dict_with_prefix("embed_tokens."));
    // call each layer's load_state_dict function
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    npu_embed_tokens_->verify_loaded_weights(prefix + "embed_tokens.");
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
    norm_->verify_loaded_weights(prefix + "norm.");
  }

  void merge_loaded_weights() {
    npu_embed_tokens_->merge_loaded_weights();
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->merge_loaded_weights();
    }
    norm_->merge_loaded_weights();
  }

  void merge_and_move_pinned_host() {
    npu_embed_tokens_->merge_and_move_pinned_host();
    for (size_t i = 0; i < layers_.size(); i++) {
      layers_[i]->merge_and_move_pinned_host();
    }
    norm_->merge_and_move_pinned_host();
  }

  void free_weights() {
    npu_embed_tokens_->free_weights();
    for (size_t i = 0; i < layers_.size(); i++) {
      layers_[i]->free_weights();
    }
    norm_->free_weights();
  }

  void reload_weights() {
    npu_embed_tokens_->reload_weights();
    for (size_t i = 0; i < layers_.size(); i++) {
      layers_[i]->reload_weights();
    }
    norm_->reload_weights();
  }

  void reload_non_decoder_weights() {
    npu_embed_tokens_->reload_weights();
    norm_->reload_weights();
  }

  void reload_weights_from_device() {
    npu_embed_tokens_->reload_weights_from_device();
    for (size_t i = 0; i < layers_.size(); i++) {
      layers_[i]->reload_weights_from_device();
    }
    norm_->reload_weights_from_device();
  }

  void refresh_rolling_weights() {
    for (auto& layer : layers_) {
      layer->refresh_rolling_weights();
    }
  }

  std::vector<layer::BaseLoader*> get_decoder_loaders() {
    std::vector<layer::BaseLoader*> loaders;
    loaders.reserve(layers_.size());
    for (auto& layer : layers_) {
      loaders.push_back(layer->get_loader());
    }
    return loaders;
  }

  void set_rolling_load_manager(RollingLoadManager* mgr) { rolling_mgr_ = mgr; }

  layer::NpuWordEmbedding get_npu_word_embedding() { return npu_embed_tokens_; }

  void set_npu_word_embedding(layer::NpuWordEmbedding& npu_word_embedding) {
    npu_embed_tokens_ = npu_word_embedding;
  }

 private:
  torch::nn::ModuleList blocks_{nullptr};
  std::vector<Glm4MoeDecoderLayer> layers_;
  int32_t max_seq_len_ = 0;
  int32_t dp_rank_;
  int32_t rank_;
  int32_t dp_size_;
  int32_t dp_local_tp_size_;
  nlohmann::json mapping_data_;
  int32_t num_experts_per_tok_;
  int32_t num_speculative_tokens_ = 0;
  at::Device device_;
  torch::Dtype dtype_;
  layer::NpuWordEmbedding npu_embed_tokens_{nullptr};
  layer::AttentionMask attn_mask_;
  layer::NpuRMSNorm norm_{nullptr};
  torch::Tensor cos_sin_;
  layer::NpuPosEmbedding atb_pos_emb_{nullptr};

  std::vector<int64_t> mrope_section_;
  RollingLoadManager* rolling_mgr_ = nullptr;
};
TORCH_MODULE(Glm4MoeModel);

class Glm4MoeForCausalLMImpl
    : public xllm::npu::model::LlmForCausalLMImplBase<Glm4MoeModel> {
 public:
  Glm4MoeForCausalLMImpl(const ModelContext& context)
      : xllm::npu::model::LlmForCausalLMImplBase<Glm4MoeModel>(context) {}
};
TORCH_MODULE(Glm4MoeForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL(glm4_moe, Glm4MoeForCausalLM);

// register the model args
// example config:
// https://huggingface.co/zai-org/GLM-4.5-Air/blob/main/config.json
REGISTER_MODEL_ARGS(glm4_moe, [&] {
  LOAD_ARG_OR(model_type, "model_type", "glm4_moe");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(attention_bias, "attention_bias", false);
  LOAD_ARG_OR(attention_dropout, "attention_dropout", 0.0f);
  LOAD_ARG_OR(decoder_sparse_step, "decoder_sparse_step", 1);
  LOAD_ARG_OR(eos_token_id_vec, "eos_token_id", std::vector<int>{151329});
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
}  // namespace xllm::npu::model

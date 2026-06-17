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
#include "core/framework/model/model_output.h"
#include "core/layers/npu/npu_llama_decoder_layer_impl.h"
#include "llm_model_base.h"

// llama2 model compatible with huggingface weights
namespace xllm::npu::model {

class LlamaDecoderLayerImpl : public torch::nn::Module {
 public:
  LlamaDecoderLayerImpl(const ModelContext& context) {
    // register submodules
    decoder_layer_ =
        register_module("decoder_layer", layer::NpuLlamaDecoderLayer(context));
  }

  torch::Tensor forward(torch::Tensor& x,
                        torch::Tensor& cos_pos,
                        torch::Tensor& sin_pos,
                        torch::Tensor& attn_mask,
                        KVCache& kv_cache,
                        ModelInputParams& input_params,
                        int node_id) {
    return decoder_layer_(
        x, cos_pos, sin_pos, attn_mask, kv_cache, input_params, node_id);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    decoder_layer_->load_state_dict(state_dict);
  }

  void verify_loaded_weights(const std::string& prefix) const {}

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
  layer::NpuLlamaDecoderLayer decoder_layer_{nullptr};
};
TORCH_MODULE(LlamaDecoderLayer);

std::tuple<torch::Tensor, torch::Tensor> get_llama_rotary_embedding(
    int64_t dim,
    int64_t seq_len,
    double rope_theta,
    const torch::TensorOptions& options) {
  // auto inv_freq = 1.0 / torch::pow(10000, torch::arange(0, dim, 2, options) /
  // dim);
  auto options_new =
      torch::device(options.device()).dtype(at::ScalarType::Double);
  auto inv_freq =
      1.0 / torch::pow(rope_theta, torch::arange(0, dim, 2, options_new) / dim)
                .to(at::ScalarType::Float);
  auto seq_idx = torch::arange(seq_len, options_new);

  auto freqs = torch::ger(seq_idx, inv_freq).to(torch::kFloat32);
  auto emb = torch::cat({freqs, freqs}, -1);
  auto rope_cos = torch::cos(emb);
  auto rope_sin = torch::sin(emb);

  auto dtype = options.dtype();
  if (dtype == torch::kFloat16 || dtype == torch::kBFloat16 ||
      dtype == torch::kInt8) {
    if (dtype == torch::kBFloat16) {
      rope_cos = rope_cos.to(torch::kBFloat16);
      rope_sin = rope_sin.to(torch::kBFloat16);
    } else {
      rope_cos = rope_cos.to(torch::kFloat16);
      rope_sin = rope_sin.to(torch::kFloat16);
    }
  }
  return std::make_tuple(rope_cos, rope_sin);
}

class LlamaModelImpl : public torch::nn::Module {
 public:
  LlamaModelImpl(const ModelContext& context) {
    auto options = context.get_tensor_options();
    auto model_args = context.get_model_args();
    // register submodules
    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(context.get_model_args().n_layers());
    npu_embed_tokens_ =
        register_module("npu_embed_tokens", layer::NpuWordEmbedding(context));
    norm_ = register_module("norm", layer::NpuRMSNorm(context));

    std::tie(cos_pos_, sin_pos_) =
        get_llama_rotary_embedding(128,
                                   model_args.max_position_embeddings(),
                                   model_args.rope_theta(),
                                   options);
    // encode_attn_mask_ =
    //   layer::AttentionMask(options.device(),
    //   options.dtype()).get_attn_mask(2048, options.device(),
    //   options.dtype());
    int32_t mask_value =
        ::xllm::SchedulerConfig::get_instance().enable_chunked_prefill() ? -9984
                                                                         : 1;
    attn_mask_ = layer::AttentionMask(options.device(),
                                      options.dtype().toScalarType(),
                                      /*mask_value=*/mask_value);
    max_seq_len_ = 0;

    for (int32_t i = 0; i < model_args.n_layers(); i++) {
      auto block = LlamaDecoderLayer(context);
      layers_.push_back(block);
      blocks_->push_back(block);
    }
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    torch::Tensor h = npu_embed_tokens_(tokens, 0);
    auto cos_pos = cos_pos_.index_select(0, positions);
    auto sin_pos = sin_pos_.index_select(0, positions);
    ModelInputParams& input_params_new =
        const_cast<ModelInputParams&>(input_params);
    // torch::Tensor max_of_seq =
    // torch::max(input_params.attention.device.kv_seq_lens); max_seq_len_ =
    // std::max(max_of_seq.item<int>(), max_seq_len_);
    torch::Tensor max_of_seq =
        torch::max(input_params.attention.device.kv_seq_lens);
    max_seq_len_ =
        ::xllm::SchedulerConfig::get_instance().enable_chunked_prefill()
            ? std::max(max_of_seq.item<int>(), max_seq_len_)
            : 128;
    auto attn_mask = attn_mask_.get_attn_mask(
        max_seq_len_, cos_pos.dtype().toScalarType(), cos_pos.device());

    if (::xllm::SchedulerConfig::get_instance().enable_chunked_prefill()) {
      int batch_size = input_params.attention.host.q_seq_lens.size();
      std::vector<torch::Tensor> req_mask_vec;
      req_mask_vec.reserve(batch_size);

      for (int i = 0; i < batch_size; i++) {
        int start = input_params.attention.host.kv_seq_lens[i] -
                    input_params.attention.host.q_seq_lens[i];
        int end = input_params.attention.host.kv_seq_lens[i];

        auto req_mask_slice = attn_mask.slice(0, start, end);
        req_mask_vec.emplace_back(req_mask_slice);
      }
      attn_mask = torch::cat(req_mask_vec, 0);
    }
    RollingLayerGuard rolling_guard(rolling_mgr_);
    for (size_t i = 0; i < layers_.size(); i++) {
      auto& layer = layers_[i];
      const int32_t layer_index = i;
      rolling_guard.before_layer(layer_index);
      layer(h, cos_pos, sin_pos, attn_mask, kv_caches[i], input_params_new, i);
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
    // test
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

  layer::NpuWordEmbedding get_npu_word_embedding() {
    return {npu_embed_tokens_};
  }

  void set_npu_word_embedding(layer::NpuWordEmbedding& npu_word_embedding) {
    npu_embed_tokens_ = npu_word_embedding;
  }

 private:
  torch::Tensor cos_pos_;
  torch::Tensor sin_pos_;
  int max_seq_len_ = 0;
  int device_id_ = 0;
  layer::AttentionMask attn_mask_;
  layer::NpuWordEmbedding npu_embed_tokens_{nullptr};
  layer::NpuRMSNorm norm_{nullptr};

  torch::nn::ModuleList blocks_{nullptr};
  // hold same data but different type as blocks_ to avoid type cast
  std::vector<LlamaDecoderLayer> layers_;
  RollingLoadManager* rolling_mgr_ = nullptr;
};
TORCH_MODULE(LlamaModel);

class LlamaForCausalLMImpl
    : public xllm::npu::model::LlmForCausalLMImplBase<LlamaModel> {
 public:
  LlamaForCausalLMImpl(const ModelContext& context)
      : xllm::npu::model::LlmForCausalLMImplBase<LlamaModel>(context) {}
};
TORCH_MODULE(LlamaForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL(llama, LlamaForCausalLM);

// register the model args
// example config:
// https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct/blob/main/config.json
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

}  // namespace xllm::npu::model

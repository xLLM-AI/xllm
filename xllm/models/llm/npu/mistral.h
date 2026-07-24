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

#include <torch/torch.h>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_output.h"
#include "core/layers/common/activation.h"
#include "core/layers/common/attention.h"
#include "core/layers/common/linear.h"
#include "core/layers/common/rotary_embedding.h"  // Use existing rotary_embedding
#include "core/layers/npu/npu_mistral_decoder_layer_impl.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/state_dict/state_dict.h"
#include "models/model_registry.h"

// Mistral model compatible with huggingface weights
namespace xllm {

// ==================== Mistral Decoder Layer ====================

class MistralDecoderLayerImpl : public torch::nn::Module {
 public:
  MistralDecoderLayerImpl(const ModelContext& context) {
    decoder_layer_ = register_module("decoder_layer",
                                     layer::NpuMistralDecoderLayer(context));
  }

  ModelOutput forward(torch::Tensor& x,
                      torch::Tensor& cos_pos,
                      torch::Tensor& sin_pos,
                      torch::Tensor& attn_mask,
                      KVCache& kv_cache,
                      ModelInputParams& input_params,
                      int node_id) {
    auto hidden_states = decoder_layer_(
        x, cos_pos, sin_pos, attn_mask, kv_cache, input_params, node_id);
    return ModelOutput(hidden_states);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    decoder_layer_->load_state_dict(state_dict);
  }

  void verify_loaded_weights(const std::string& prefix) const {}

  // Add missing lifecycle functions
  void merge_loaded_weights() {
    if (decoder_layer_) {
      decoder_layer_->merge_loaded_weights();
    }
  }

  void merge_and_move_pinned_host() {
    if (decoder_layer_) {
      decoder_layer_->merge_and_move_pinned_host();
    }
  }

  void free_weights() {
    if (decoder_layer_) {
      decoder_layer_->free_weights();
    }
  }

  void reload_weights() {
    if (decoder_layer_) {
      decoder_layer_->reload_weights();
    }
  }

  void reload_weights_from_device() {
    if (decoder_layer_) {
      decoder_layer_->reload_weights_from_device();
    }
  }

  torch::Tensor _create_4d_causal_attention_mask(torch::IntArrayRef input_shape,
                                                 torch::Dtype dtype,
                                                 torch::Device device) {
    const int64_t bsz = input_shape[0];
    const int64_t tgt_len = input_shape[1];

    auto options = torch::TensorOptions().dtype(dtype).device(device);
    auto causal_mask = torch::full(
        {tgt_len, tgt_len}, -std::numeric_limits<double>::infinity(), options);
    causal_mask.triu_(1);
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0);
    causal_mask = causal_mask.expand({bsz, 1, tgt_len, tgt_len});
    return causal_mask;
  }

 private:
  layer::NpuMistralDecoderLayer decoder_layer_{nullptr};
};
TORCH_MODULE(MistralDecoderLayer);

inline std::tuple<torch::Tensor, torch::Tensor> get_mistral_rotary_embedding(
    int64_t dim,
    int64_t seq_len,
    double rope_theta,
    const torch::TensorOptions& options) {
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

// ==================== Mistral Model ====================

class MistralModelImpl : public torch::nn::Module {
 public:
  MistralModelImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    auto parallel_args = context.get_parallel_args();

    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(context.get_model_args().n_layers());
    npu_embed_tokens_ =
        register_module("npu_embed_tokens", layer::NpuWordEmbedding(context));
    norm_ = register_module("norm", layer::NpuRMSNorm(context));
    std::tie(cos_pos_, sin_pos_) =
        get_mistral_rotary_embedding(128,
                                     model_args.max_position_embeddings(),
                                     model_args.rope_theta(),
                                     options);

    int32_t mask_value =
        ::xllm::SchedulerConfig::get_instance().enable_chunked_prefill() ? -9984
                                                                         : 1;
    attn_mask_ = layer::AttentionMask(options.device(),
                                      options.dtype().toScalarType(),
                                      /*mask_value=*/mask_value);
    max_seq_len_ = 0;

    for (int32_t i = 0; i < model_args.n_layers(); i++) {
      auto block = MistralDecoderLayer(context);
      layers_.push_back(block);
      blocks_->push_back(block);
    }
  }

  torch::Tensor get_input_embeddings(torch::Tensor input_ids) {
    return npu_embed_tokens_(input_ids, 0);
  }

  layer::NpuWordEmbedding get_npu_word_embedding() { return npu_embed_tokens_; }
  void set_npu_word_embedding(layer::NpuWordEmbedding& npu_word_embedding) {
    npu_embed_tokens_ = npu_word_embedding;
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    // Reuse pre-computed input embedding from VlmExecutorImpl::run() if
    // available, to avoid calling npu_embed_tokens_ twice per step. The ATB
    // word_embedding all_gather accumulates async HCCL resources and fails
    // after repeated calls (device tensor is null).
    torch::Tensor h;
    const auto& precomputed = input_params.embedding.input_embedding;
    if (precomputed.defined()) {
      h = precomputed;
    } else {
      h = npu_embed_tokens_(tokens, 0);
    }
    auto cos_pos = cos_pos_.index_select(0, positions);
    auto sin_pos = sin_pos_.index_select(0, positions);
    ModelInputParams& input_params_new =
        const_cast<ModelInputParams&>(input_params);
    torch::Tensor max_of_seq =
        torch::max(input_params.attention.device.kv_seq_lens);
    max_seq_len_ =
        ::xllm::SchedulerConfig::get_instance().enable_chunked_prefill()
            ? std::max(max_of_seq.item<int>(), max_seq_len_)
            : 128;

    torch::Tensor attn_mask;

    if (::xllm::SchedulerConfig::get_instance().enable_chunked_prefill()) {
      // LOG(FATAL)
      //     << "Flux2 text encoder (Mistral) does not support chunked_prefill.
      //     "
      //     << "Please set --enable_chunked_prefill=false and restart.";
      // Use the original logic
      int32_t max_kv_seq = input_params.meta.kv_max_seq_len;
      int32_t num_sequences = input_params.meta.num_sequences;
      if (num_sequences > 0) {
        std::vector<torch::Tensor> req_mask_vec;
        req_mask_vec.reserve(num_sequences);
        int64_t row_offset = 0;
        int64_t token_offset = 0;
        for (int32_t j = 0; j < num_sequences; j++) {
          int32_t q_len = input_params.attention.host.q_seq_lens[j];
          int32_t kv_len = input_params.attention.host.kv_seq_lens[j];
          int64_t pad_count = 0;
          int64_t token_end =
              std::min<int64_t>(tokens.size(0), token_offset + q_len);
          if (token_offset < token_end) {
            auto token_slice = tokens.slice(0, token_offset, token_end);
            auto is_pad_cpu = (token_slice == 11).cpu();
            for (int64_t i = 0; i < is_pad_cpu.size(0); ++i) {
              if (is_pad_cpu[i].item<bool>()) {
                pad_count++;
              } else {
                break;
              }
            }
          }

          auto mask = attn_mask_.gen_append_mask(q_len,
                                                 kv_len,
                                                 max_kv_seq,
                                                 cos_pos.dtype().toScalarType(),
                                                 cos_pos.device());
          if (pad_count > 0) {
            int64_t pad_cols = std::min<int64_t>(pad_count, mask.size(1));
            int64_t pad_rows = std::min<int64_t>(pad_count, mask.size(0));
            mask.slice(0, 0, pad_rows).fill_(-9984);
            mask.slice(1, 0, pad_cols).fill_(-9984);
          }
          req_mask_vec.emplace_back(mask);
          row_offset += q_len;
          token_offset += q_len;
        }
        attn_mask = torch::cat(req_mask_vec, 0);
        LOG(INFO) << "chunk prefill attn_mask dtype: " << attn_mask.dtype()
                  << ", cos_pos dtype: " << cos_pos.dtype()
                  << ", h dtype: " << h.dtype()
                  << ", attn_mask sizes: " << attn_mask.sizes();
      }
    } else if (input_params.meta.batch_forward_type.is_prefill()) {
      int64_t seq_len = h.size(0);
      float min_dtype = 1.0f;
      auto opts = torch::TensorOptions()
                      .dtype(cos_pos.dtype().toScalarType())
                      .device(cos_pos.device());

      // Detect left-padding from token ID (pad_token_id=770)
      auto is_pad = (tokens == 11);  // [seqLen]
      int64_t pad_count = 0;
      // left-padding: consecutive pad tokens starting from position 0
      auto is_pad_cpu = is_pad.cpu();
      for (int64_t i = 0; i < seq_len; ++i) {
        if (is_pad_cpu[i].item<bool>()) {
          pad_count++;
        } else {
          break;
        }
      }
      // Create causal mask [seq_len, seq_len]
      auto causal = torch::zeros({seq_len, seq_len}, opts);
      auto upper = torch::ones({seq_len, seq_len}, opts);
      upper.triu_(1);
      causal = causal + upper * min_dtype;

      if (pad_count > 0) {
        // left-padding: mask all rows and columns of the first pad_count
        causal.slice(0, 0, pad_count) = min_dtype;
        causal.slice(1, 0, pad_count) = min_dtype;
      }
      attn_mask = causal;
    }

    std::vector<torch::Tensor> all_layer_hidden_states;
    all_layer_hidden_states.reserve(layers_.size());

    for (size_t i = 0; i < layers_.size(); i++) {
      auto& layer = layers_[i];
      layer(h, cos_pos, sin_pos, attn_mask, kv_caches[i], input_params_new, i);

      all_layer_hidden_states.emplace_back(
          h.clone());  // Collect the output of each layer
    }

    auto hidden_states = norm_(h, 0);
    auto stacked = torch::stack(all_layer_hidden_states, /*dim=*/0);
    return ModelOutput(hidden_states, torch::Tensor(), stacked);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    npu_embed_tokens_->load_state_dict(
        state_dict.get_dict_with_prefix("embed_tokens."));
    // rotary_emb has no weights to load (all are buffers)
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

  // Add missing lifecycle functions
  void merge_loaded_weights() {
    LOG(INFO) << "Merging loaded weights for MistralModel";

    if (npu_embed_tokens_) {
      npu_embed_tokens_->merge_loaded_weights();
    }

    for (auto& layer : layers_) {
      if (layer) {
        layer->merge_loaded_weights();
      }
    }

    if (norm_) {
      norm_->merge_loaded_weights();
    }

    LOG(INFO) << "MistralModel merge_loaded_weights completed";
  }

  void merge_and_move_pinned_host() {
    if (npu_embed_tokens_) {
      npu_embed_tokens_->merge_and_move_pinned_host();
    }

    for (auto& layer : layers_) {
      if (layer) {
        layer->merge_and_move_pinned_host();
      }
    }

    if (norm_) {
      norm_->merge_and_move_pinned_host();
    }
  }

  void free_weights() {
    if (npu_embed_tokens_) {
      npu_embed_tokens_->free_weights();
    }

    for (auto& layer : layers_) {
      if (layer) {
        layer->free_weights();
      }
    }

    if (norm_) {
      norm_->free_weights();
    }
  }

  void reload_weights() {
    if (npu_embed_tokens_) {
      npu_embed_tokens_->reload_weights();
    }

    for (auto& layer : layers_) {
      if (layer) {
        layer->reload_weights();
      }
    }

    if (norm_) {
      norm_->reload_weights();
    }
  }

  void reload_weights_from_device() {
    if (npu_embed_tokens_) {
      npu_embed_tokens_->reload_weights_from_device();
    }

    for (auto& layer : layers_) {
      if (layer) {
        layer->reload_weights_from_device();
      }
    }

    if (norm_) {
      norm_->reload_weights_from_device();
    }
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
  std::vector<MistralDecoderLayer> layers_;
};
TORCH_MODULE(MistralModel);

// ==================== Mistral For Causal LM ====================

class MistralForCausalLMImpl : public torch::nn::Module {
 public:
  MistralForCausalLMImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    auto parallel_args = context.get_parallel_args();

    // register submodules
    model_ = register_module("model", MistralModel(context));

    lm_head_ = register_module("npu_lm_head", layer::NpuLmHead(context));
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  // returns: [num_tokens, hidden_size]
  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    auto hidden_states = model_(tokens, positions, kv_caches, input_params);
    return ModelOutput(hidden_states);
  }

  virtual torch::Tensor pooler(const torch::Tensor& hidden_states,
                               const torch::Tensor& seleted_idxes) {
    auto h = hidden_states;
    if (seleted_idxes.defined()) {
      h = h.index_select(/*dim=*/0, seleted_idxes);
    }
    return h;
  }

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  // returns: [num_tokens, vocab_size]
  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    return lm_head_(hidden_states, seleted_idxes, 0);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    model_->load_state_dict(state_dict.get_dict_with_prefix("model."));
    lm_head_->load_state_dict(state_dict.get_dict_with_prefix("lm_head."));
  }

  void verify_loaded_weights() const {
    model_->verify_loaded_weights("model.");
    lm_head_->verify_loaded_weights("lm_head.");
  }

  virtual void prepare_expert_weight(int32_t layer_id,
                                     const std::vector<int32_t>& expert_ids) {
    return;
  }
  virtual void update_expert_weight(int32_t layer_id) { return; }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    LOG(INFO) << "Loading MistralForCausalLM from ModelLoader...";
    for (const auto& state_dict : loader->get_state_dicts()) {
      model_->load_state_dict(
          state_dict->get_dict_with_prefix("language_model."));
      lm_head_->load_state_dict(
          state_dict->get_dict_with_prefix("language_model.lm_head."));
    }
    // Critical: add merge_loaded_weights call!
    if (model_) {
      model_->merge_loaded_weights();
    }
    if (lm_head_) {
      lm_head_->merge_loaded_weights();
    }
    model_->verify_loaded_weights("language_model.");
    lm_head_->verify_loaded_weights("language_model.lm_head.");
    LOG(INFO) << "MistralForCausalLM loaded successfully.";
  }

 private:
  // parameter members, must be registered
  MistralModel model_{nullptr};
  layer::NpuLmHead lm_head_{nullptr};
};
TORCH_MODULE(MistralForCausalLM);

// ==================== Registration ====================

REGISTER_CAUSAL_MODEL(mistral, MistralForCausalLM);
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

}  // namespace xllm
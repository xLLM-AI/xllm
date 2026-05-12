/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "core/common/flash_comm1_context.h"
#include "core/common/global_flags.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_context.h"
#include "core/framework/model_loader.h"
#include "core/layers/common/attention_mask.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/layers/common/lm_head.h"
#include "core/layers/common/qwen3_next_rms_norm.h"
#include "core/layers/common/word_embedding.h"
#include "core/layers/npu_torch/qwen3_next_hybrid_decoder_layer_base.h"

namespace xllm {

class Qwen3HybridModelModule : public torch::nn::Module {
 public:
  virtual ModelOutput forward(torch::Tensor tokens,
                              torch::Tensor positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelInputParams& input_params) = 0;
  virtual void load_state_dict(const StateDict& state_dict) = 0;
  virtual void verify_loaded_weights(const std::string& prefix) const = 0;
  virtual layer::WordEmbedding get_word_embedding() = 0;
  virtual void set_word_embedding(layer::WordEmbedding& word_embedding) = 0;
};

using Qwen3HybridModelModulePtr = std::shared_ptr<Qwen3HybridModelModule>;

class Qwen3HybridModelImplBase : public Qwen3HybridModelModule {
 public:
  explicit Qwen3HybridModelImplBase(const ModelContext& context)
      : device_(context.get_tensor_options().device()),
        model_args_(context.get_model_args()),
        parallel_args_(context.get_parallel_args()) {
    auto options = context.get_tensor_options();

    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(model_args_.n_layers());
    device_ = options.device();
    dtype_ = options.dtype().toScalarType();
    norm_ = register_module(
        "norm",
        xllm::layer::Qwen3NextRMSNorm(
            model_args_.hidden_size(), model_args_.rms_norm_eps(), options));
    embed_tokens_ =
        register_module("embed_tokens", layer::WordEmbedding(context));
    int32_t mask_value = FLAGS_enable_chunked_prefill ? -9984 : 1;
    attn_mask_ = layer::AttentionMask(options.device(),
                                      options.dtype().toScalarType(),
                                      /*mask_value=*/mask_value);
    dp_size_ = parallel_args_.dp_size();
  }

  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) override {
    torch::NoGradGuard no_grad;
    if (dp_size_ > 1) {
      if (tokens.sizes() == 0) {
        tokens = torch::tensor({1}).to(torch::kInt32).to(device_);
        positions = torch::tensor({0}).to(torch::kInt32).to(device_);
      }
    }

    int32_t num_tokens = static_cast<int32_t>(tokens.size(0));
    bool is_prefill = input_params.batch_forward_type.is_prefill();
    FlashComm1Context fc1_ctx = build_flash_comm1_context(
        num_tokens, is_prefill, parallel_args_);

    layer::AttentionMetadata attn_metadata =
        layer::AttentionMetadataBuilder::build(
            input_params,
            model_args_.enable_mla(),
            build_attention_mask(input_params));
    torch::Tensor h = embed_tokens_(tokens);

    if (fc1_ctx.is_sequence_sharded()) {
      h = shard_sequence(h, fc1_ctx);
    }

    torch::Tensor mrope_cos_sin;
    for (const auto& layer : layers_) {
      mrope_cos_sin = layer->build_mrope_cos_sin(positions);
      if (mrope_cos_sin.defined()) break;
    }

    std::optional<torch::Tensor> residual = std::nullopt;
    for (size_t i = 0; i < layers_.size(); i++) {
      auto& layer = layers_[i];
      h = layer->forward(h,
                         residual,
                         positions,
                         attn_metadata,
                         kv_caches[i],
                         input_params,
                         mrope_cos_sin,
                         &fc1_ctx);
    }
    auto [hidden_states, residual_out] = norm_->forward(h, residual);
    h = hidden_states;

    if (fc1_ctx.is_sequence_sharded()) {
      h = gather_and_unpad_sequence(h, fc1_ctx);
    }

    return ModelOutput(h);
  }

  void load_state_dict(const StateDict& state_dict) override {
    embed_tokens_->load_state_dict(
        state_dict.get_dict_with_prefix("embed_tokens."));
    for (int i = 0; i < static_cast<int>(layers_.size()); i++) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  }

  void verify_loaded_weights(const std::string& prefix) const override {
    for (size_t i = 0; i < layers_.size(); ++i) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
  }

  layer::WordEmbedding get_word_embedding() override { return embed_tokens_; }

  void set_word_embedding(layer::WordEmbedding& word_embedding) override {
    embed_tokens_ = word_embedding;
  }

  void add_decoder_layer(layer::Qwen3HybridDecoderLayerModulePtr layer) {
    layers_.push_back(layer);
    blocks_->push_back(layer);
  }

  int32_t num_hidden_layers() const {
    return static_cast<int32_t>(layers_.size());
  }

 protected:
  torch::Tensor build_attention_mask(const ModelInputParams& input_params) {
    max_seq_len_ = std::max(input_params.kv_max_seq_len, max_seq_len_);
    if (!FLAGS_enable_chunked_prefill) {
      return attn_mask_.get_attn_mask(max_seq_len_, dtype_, device_);
    }

    const int32_t num_sequences = input_params.num_sequences;
    if (num_sequences <= 0) {
      return attn_mask_.get_attn_mask(max_seq_len_, dtype_, device_);
    }

    std::vector<torch::Tensor> req_mask_vec;
    req_mask_vec.reserve(num_sequences);
    for (int32_t j = 0; j < num_sequences; ++j) {
      req_mask_vec.emplace_back(
          attn_mask_.gen_append_mask(input_params.q_seq_lens_vec[j],
                                     input_params.kv_seq_lens_vec[j],
                                     max_seq_len_,
                                     dtype_,
                                     device_));
    }
    return torch::cat(req_mask_vec, 0);
  }

  ModelArgs model_args_;
  ParallelArgs parallel_args_;
  torch::nn::ModuleList blocks_{nullptr};
  std::vector<layer::Qwen3HybridDecoderLayerModulePtr> layers_;
  int32_t max_seq_len_ = 0;
  int32_t dp_size_ = 1;
  torch::Device device_;
  torch::ScalarType dtype_ = torch::kFloat;
  layer::Qwen3NextRMSNorm norm_{nullptr};
  layer::AttentionMask attn_mask_;
  layer::WordEmbedding embed_tokens_{nullptr};
};

class Qwen3HybridForCausalLMImplBase : public torch::nn::Module {
 public:
  explicit Qwen3HybridForCausalLMImplBase(const ModelContext& context) {
    tie_word_embeddings_ = context.get_model_args().tie_word_embeddings();
    lm_head_ = register_module("lm_head", layer::LmHead(context));
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  // returns: [num_tokens, hidden_size]
  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    return model_->forward(tokens, positions, kv_caches, input_params);
  }

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  // returns: [num_tokens, vocab_size]
  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    auto h = hidden_states;
    if (seleted_idxes.defined()) {
      h = h.index_select(/*dim=*/0, seleted_idxes);
    }
    return lm_head_(h);
  }

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    auto h = hidden_states;
    if (seleted_idxes.defined()) {
      h = h.index_select(/*dim=*/0, seleted_idxes);
    }
    namespace F = torch::nn::functional;
    return F::normalize(h, F::NormalizeFuncOptions().p(2).dim(1));
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    auto has_model_weights = [](const StateDict& dict) {
      return dict.get_tensor("embed_tokens.weight").defined() ||
             dict.get_dict_with_prefix("layers.").size() > 0 ||
             dict.get_tensor("norm.weight").defined();
    };
    auto has_lm_head_weights = [](const StateDict& dict) {
      return dict.get_tensor("weight").defined() ||
             dict.get_tensor("qweight").defined();
    };

    for (const auto& state_dict : loader->get_state_dicts()) {
      auto model_state_dict = state_dict->get_dict_with_prefix("model.");
      if (!has_model_weights(model_state_dict)) {
        auto language_model_state_dict =
            state_dict->get_dict_with_prefix("language_model.model.");
        if (has_model_weights(language_model_state_dict)) {
          model_state_dict = language_model_state_dict;
        } else {
          auto wrapped_language_model_state_dict =
              state_dict->get_dict_with_prefix("model.language_model.");
          if (has_model_weights(wrapped_language_model_state_dict)) {
            model_state_dict = wrapped_language_model_state_dict;
          }
        }
      }
      model_->load_state_dict(model_state_dict);

      auto lm_head_state_dict = state_dict->get_dict_with_prefix("lm_head.");
      if (!has_lm_head_weights(lm_head_state_dict)) {
        auto language_model_lm_head_state_dict =
            state_dict->get_dict_with_prefix("language_model.lm_head.");
        if (has_lm_head_weights(language_model_lm_head_state_dict)) {
          lm_head_state_dict = language_model_lm_head_state_dict;
        } else {
          auto wrapped_language_model_lm_head_state_dict =
              state_dict->get_dict_with_prefix("model.language_model.lm_head.");
          if (has_lm_head_weights(wrapped_language_model_lm_head_state_dict)) {
            lm_head_state_dict = wrapped_language_model_lm_head_state_dict;
          } else {
            auto wrapped_lm_head_state_dict =
                state_dict->get_dict_with_prefix("model.lm_head.");
            if (has_lm_head_weights(wrapped_lm_head_state_dict)) {
              lm_head_state_dict = wrapped_lm_head_state_dict;
            }
          }
        }
      }
      if (!has_lm_head_weights(lm_head_state_dict) && tie_word_embeddings_) {
        auto tied_lm_head_state_dict =
            model_state_dict.get_dict_with_prefix("embed_tokens.");
        if (has_lm_head_weights(tied_lm_head_state_dict)) {
          lm_head_state_dict = tied_lm_head_state_dict;
        }
      }
      lm_head_->load_state_dict(lm_head_state_dict);
    }

    model_->verify_loaded_weights("model.");
  }

  virtual void prepare_expert_weight(int32_t layer_id,
                                     const std::vector<int32_t>& expert_ids) {
    return;
  }
  virtual void update_expert_weight(int32_t layer_id) { return; }

  layer::LmHead get_lm_head() { return lm_head_; }

  void set_lm_head(layer::LmHead& head) { lm_head_ = head; }

  layer::WordEmbedding get_word_embedding() {
    return model_->get_word_embedding();
  }

  void set_word_embedding(layer::WordEmbedding& word_embedding) {
    model_->set_word_embedding(word_embedding);
  }

  void set_model_module(Qwen3HybridModelModulePtr model) {
    model_ = register_module("model", std::move(model));
  }

 protected:
  bool tie_word_embeddings_{false};
  layer::LmHead lm_head_{nullptr};
  Qwen3HybridModelModulePtr model_;
};

}  // namespace xllm

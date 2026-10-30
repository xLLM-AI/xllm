/* Copyright 2025-2026 The xLLM Authors.

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

#include <c10/core/Device.h>
#include <torch/types.h>

#include <vector>

#include "causal_lm.h"
#include "core/framework/model_context/model_context.h"
#include "core/framework/quantization/quant_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/kv_cache/storage/kv_cache.h"
#include "model_args.h"
#include "model_params_view.h"

namespace xllm {

class CausalVLM : public CausalLM {
 public:
  ~CausalVLM() override = default;
  virtual MMDict encode(const ModelParamsView& parameters) = 0;
  virtual torch::Tensor get_input_embeddings(
      const torch::Tensor& input_ids,
      const ModelParamsView& input_params) = 0;
};

template <typename ModelType>
class CausalVLMImpl : public CausalVLM {
 public:
  CausalVLMImpl(ModelType model, const ModelContext& context)
      : model_(std::move(model)), options_(context.get_tensor_options()) {
    const auto& args = context.get_model_args();
    tie_word_embeddings_ = args.tie_word_embeddings();
    embedding_mode_ = args.embedding_mode();
    // Models that own a non-standard head (e.g. mistral3's language_model
    // lm_head) provide their own logits; skip registering a framework-owned
    // lm_head for them.
    const bool owns_custom_head = detail::has_logits<ModelType>::value;
    if (!embedding_mode_ && !owns_custom_head) {
#if defined(USE_NPU)
      if constexpr (detail::has_get_npu_lm_head<ModelType>::value) {
        npu_lm_head_ =
            register_module("npu_lm_head", layer::NpuLmHead(context));
      } else
#endif
      {
        lm_head_ = register_module("lm_head", layer::LmHead(context));
      }
    }
  }

  MMDict encode(const ModelParamsView& parameters) override {
    return model_->get_multimodal_embeddings(parameters);
  }

  torch::Tensor get_input_embeddings(
      const torch::Tensor& input_ids,
      const ModelParamsView& input_params) override {
    return model_->get_input_embeddings(input_ids, input_params);
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelParamsView& parameters) override {
    return model_->forward(tokens, positions, kv_caches, parameters);
  }

  bool is_hybrid_linear_attention() override {
    if constexpr (detail::has_is_hybrid_linear_attention<ModelType>::value) {
      return model_->is_hybrid_linear_attention();
    } else {
      return CausalLM::is_hybrid_linear_attention();
    }
  }

  bool supports_mla_graph_kv_bucketing() const override {
    if constexpr (detail::has_supports_mla_graph_kv_bucketing<
                      ModelType>::value) {
      return model_->supports_mla_graph_kv_bucketing();
    }
    return CausalLM::supports_mla_graph_kv_bucketing();
  }

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override {
    if constexpr (detail::has_pooler<ModelType>::value) {
      return model_->pooler(hidden_states, seleted_idxes);
    } else {
      return CausalLM::pooler(hidden_states, seleted_idxes);
    }
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override {
    if constexpr (detail::has_logits<ModelType>::value) {
      return model_->logits(hidden_states, seleted_idxes);
    }
#if defined(USE_NPU)
    if constexpr (detail::has_get_npu_lm_head<ModelType>::value) {
      return npu_lm_head_(hidden_states, seleted_idxes, 0);
    }
#endif
    return lm_head_(hidden_states, seleted_idxes, 0);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes,
                       torch::Tensor& out_hidden) override {
    if constexpr (detail::has_logits<ModelType>::value) {
      return CausalLM::logits(hidden_states, seleted_idxes, out_hidden);
    }
#if defined(USE_NPU)
    if constexpr (detail::has_get_npu_lm_head<ModelType>::value) {
      return npu_lm_head_->forward_with_hidden(
          hidden_states, seleted_idxes, out_hidden, 0);
    }
#endif
    return lm_head_->forward_with_hidden(
        hidden_states, seleted_idxes, out_hidden, 0);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) override {
    if (!embedding_mode_ && !detail::has_logits<ModelType>::value) {
      for (const auto& state_dict : loader->get_state_dicts()) {
#if defined(USE_NPU)
        if constexpr (detail::has_get_npu_lm_head<ModelType>::value) {
          if (tie_word_embeddings_) {
            npu_lm_head_->load_state_dict(
                state_dict->get_dict_with_prefix(std::vector<std::string>{
                    "model.embed_tokens.", "embed_tokens."}));
          } else {
            npu_lm_head_->load_state_dict(
                state_dict->get_dict_with_prefix("lm_head."));
          }
        } else
#endif
        {
          if constexpr (detail::has_load_lm_head<ModelType>::value) {
            model_->load_lm_head(*state_dict, lm_head_);
          } else if (tie_word_embeddings_) {
            lm_head_->load_state_dict(state_dict->get_dict_with_prefix(
                std::vector<std::string>{"embed_tokens.", "embed."}));
          } else {
            lm_head_->load_state_dict(
                state_dict->get_dict_with_prefix("lm_head."));
          }
        }
      }
    }
    model_->load_model(std::move(loader));
    if (!embedding_mode_ && !detail::has_logits<ModelType>::value) {
#if defined(USE_NPU)
      if constexpr (detail::has_get_npu_lm_head<ModelType>::value) {
        npu_lm_head_->verify_loaded_weights(
            tie_word_embeddings_ ? "embed_tokens." : "lm_head.");
        npu_lm_head_->merge_loaded_weights();
      }
#endif
    }
  }

  virtual void prepare_expert_weight(int32_t layer_id,
                                     const std::vector<int32_t>& expert_ids) {
    return;
  }

  void update_expert_weight(int32_t layer_id) override { return; }

#if defined(USE_NPU)
  layer::NpuLmHead get_npu_lm_head() override {
    if constexpr (detail::has_get_npu_lm_head<ModelType>::value) {
      return model_->get_npu_lm_head();
    }
    return npu_lm_head_;
  }

  void set_npu_lm_head(layer::NpuLmHead& head) override {
    if constexpr (detail::has_set_npu_lm_head<ModelType>::value) {
      model_->set_npu_lm_head(head);
      return;
    }
    npu_lm_head_ = head;
  }

  layer::NpuWordEmbedding get_npu_word_embedding() override {
    if constexpr (detail::has_get_npu_word_embedding<ModelType>::value) {
      return model_->get_npu_word_embedding();
    } else {
      return CausalLM::get_npu_word_embedding();
    }
  }

  void set_npu_word_embedding(layer::NpuWordEmbedding& embedding) override {
    if constexpr (detail::has_set_npu_word_embedding<ModelType>::value) {
      model_->set_npu_word_embedding(embedding);
    } else {
      CausalLM::set_npu_word_embedding(embedding);
    }
  }
#endif
  layer::LmHead get_lm_head() override {
    if constexpr (detail::has_get_lm_head<ModelType>::value) {
      return model_->get_lm_head();
    }
    return lm_head_;
  }

  void set_lm_head(layer::LmHead& head) override {
    if constexpr (detail::has_set_lm_head<ModelType>::value) {
      model_->set_lm_head(head);
      return;
    }
    lm_head_ = head;
  }

  layer::WordEmbedding get_word_embedding() override {
    if constexpr (detail::has_get_word_embedding<ModelType>::value) {
      return model_->get_word_embedding();
    } else {
      return CausalLM::get_word_embedding();
    }
  }

  void set_word_embedding(layer::WordEmbedding& embedding) override {
    if constexpr (detail::has_set_word_embedding<ModelType>::value) {
      model_->set_word_embedding(embedding);
    } else {
      CausalLM::set_word_embedding(embedding);
    }
  }

  torch::Device device() const override { return options_.device(); }

  const torch::TensorOptions& options() const override { return options_; }

 private:
  ModelType model_;
  torch::TensorOptions options_;

  bool tie_word_embeddings_{false};
  bool embedding_mode_{false};

#if defined(USE_NPU)
  layer::NpuLmHead npu_lm_head_{nullptr};
#endif
  layer::LmHead lm_head_{nullptr};
};

}  // namespace xllm

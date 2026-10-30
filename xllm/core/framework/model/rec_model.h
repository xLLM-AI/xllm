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

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/framework/model/model.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model/model_traits.h"
#include "core/framework/model/rec_model_kind.h"
#include "core/framework/model/rec_model_params.h"
#include "core/framework/model_context/model_context.h"
#include "core/kv_cache/storage/kv_cache.h"

namespace xllm {

class RecModel : public torch::nn::Module, public Model {
 public:
  ~RecModel() override = default;

  virtual RecModelKind model_kind() const = 0;

  virtual ModelOutput forward(const torch::Tensor& tokens,
                              const torch::Tensor& positions,
                              std::vector<KVCache>& kv_caches,
                              const RecModelParams& parameters) = 0;

  virtual torch::Tensor pooler(const torch::Tensor& hidden_states,
                               const torch::Tensor& selected_idxes) {
    if (selected_idxes.defined()) {
      return hidden_states.index_select(/*dim=*/0, selected_idxes);
    }
    return hidden_states;
  }

  virtual torch::Tensor logits(const torch::Tensor& hidden_states,
                               const torch::Tensor& selected_idxes) = 0;
};

namespace detail {

template <typename ModelType, typename RecType>
class RecModelImpl : public RecType {
  static_assert(std::is_base_of_v<RecModel, RecType>);

 public:
  RecModelImpl(ModelType model, const ModelContext& context)
      : model_(std::move(model)), options_(context.get_tensor_options()) {
    const auto& args = context.get_model_args();
    tie_word_embeddings_ = args.tie_word_embeddings();
    embedding_mode_ = args.embedding_mode();
    // Models that own a non-standard head provide their own logits; skip
    // registering a framework-owned lm_head for them.
    const bool owns_custom_head = detail::has_logits<ModelType>::value;
    if (!embedding_mode_ && !owns_custom_head) {
#if defined(USE_NPU)
      if constexpr (detail::has_get_npu_lm_head<ModelType>::value) {
        npu_lm_head_ =
            this->register_module("npu_lm_head", layer::NpuLmHead(context));
      } else
#endif
      {
        lm_head_ = this->register_module("lm_head", layer::LmHead(context));
      }
    }
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const RecModelParams& parameters) override {
    return model_->forward(tokens, positions, kv_caches, parameters);
  }

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_idxes) override {
    return model_->pooler(hidden_states, selected_idxes);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_idxes) override {
    if constexpr (detail::has_logits<ModelType>::value) {
      return model_->logits(hidden_states, selected_idxes);
    }
#if defined(USE_NPU)
    if constexpr (detail::has_get_npu_lm_head<ModelType>::value) {
      return npu_lm_head_(hidden_states, selected_idxes, 0);
    }
#endif
    return lm_head_(hidden_states, selected_idxes, 0);
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
          if (tie_word_embeddings_) {
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

  void lazy_load_model(std::unique_ptr<ModelLoader> loader) override {
    if constexpr (detail::has_lazy_load_model<ModelType>::value) {
      model_->lazy_load_model(std::move(loader));
    } else {
      RecType::lazy_load_model(std::move(loader));
    }
  }

  void free_model_weights() override {
    if constexpr (detail::has_free_model_weights<ModelType>::value) {
      model_->free_model_weights();
    } else {
      RecType::free_model_weights();
    }
  }

  void reload_model_weights() override {
    if constexpr (detail::has_reload_model_weights<ModelType>::value) {
      model_->reload_model_weights();
    } else {
      RecType::reload_model_weights();
    }
  }

  void reload_model_weights_from_device() override {
    if constexpr (detail::has_reload_model_weights_from_device<
                      ModelType>::value) {
      model_->reload_model_weights_from_device();
    } else {
      RecType::reload_model_weights_from_device();
    }
  }

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
      return RecType::get_npu_word_embedding();
    }
  }

  void set_npu_word_embedding(layer::NpuWordEmbedding& embedding) override {
    if constexpr (detail::has_set_npu_word_embedding<ModelType>::value) {
      model_->set_npu_word_embedding(embedding);
    } else {
      RecType::set_npu_word_embedding(embedding);
    }
  }

  bool init_or_refresh_rolling_runtime(Stream* load_stream,
                                       Stream* compute_stream,
                                       int32_t num_cached_slots,
                                       int32_t requested_rolling_slots,
                                       const std::string& model_id) override {
    if constexpr (detail::has_init_or_refresh_rolling_runtime<
                      ModelType>::value) {
      return model_->init_or_refresh_rolling_runtime(load_stream,
                                                     compute_stream,
                                                     num_cached_slots,
                                                     requested_rolling_slots,
                                                     model_id);
    }
    return RecType::init_or_refresh_rolling_runtime(load_stream,
                                                    compute_stream,
                                                    num_cached_slots,
                                                    requested_rolling_slots,
                                                    model_id);
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
      return RecType::get_word_embedding();
    }
  }

  void set_word_embedding(layer::WordEmbedding& embedding) override {
    if constexpr (detail::has_set_word_embedding<ModelType>::value) {
      model_->set_word_embedding(embedding);
    } else {
      RecType::set_word_embedding(embedding);
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

}  // namespace detail

}  // namespace xllm

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

#include "core/framework/model/causal_lm.h"

namespace xllm {

class RecCausalLM : public CausalLM {
 public:
  ~RecCausalLM() override = default;
};

template <typename Model>
class RecCausalLMImpl : public RecCausalLM {
 public:
  RecCausalLMImpl(Model model, const torch::TensorOptions& options)
      : model_(std::move(model)), options_(options) {}

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& parameters) override {
    return model_->forward(tokens, positions, kv_caches, parameters);
  }

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override {
    return model_->pooler(hidden_states, seleted_idxes);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override {
    return model_->logits(hidden_states, seleted_idxes);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) override {
    model_->load_model(std::move(loader));
  }

  void lazy_load_model(std::unique_ptr<ModelLoader> loader) override {
    if constexpr (detail::has_lazy_load_model<Model>::value) {
      model_->lazy_load_model(std::move(loader));
    } else {
      RecCausalLM::lazy_load_model(std::move(loader));
    }
  }

  void free_model_weights() override {
    if constexpr (detail::has_free_model_weights<Model>::value) {
      model_->free_model_weights();
    } else {
      RecCausalLM::free_model_weights();
    }
  }

  void reload_model_weights() override {
    if constexpr (detail::has_reload_model_weights<Model>::value) {
      model_->reload_model_weights();
    } else {
      RecCausalLM::reload_model_weights();
    }
  }

  void reload_model_weights_from_device() override {
    if constexpr (detail::has_reload_model_weights_from_device<Model>::value) {
      model_->reload_model_weights_from_device();
    } else {
      RecCausalLM::reload_model_weights_from_device();
    }
  }

  void prepare_expert_weight(int32_t layer_id,
                             const std::vector<int32_t>& expert_ids) override {
    return model_->prepare_expert_weight(layer_id, expert_ids);
  }

  void update_expert_weight(int32_t layer_id) override {
    return model_->update_expert_weight(layer_id);
  }

#if defined(USE_NPU)
  layer::NpuLmHead get_npu_lm_head() override {
    if constexpr (detail::has_get_npu_lm_head<Model>::value) {
      return model_->get_npu_lm_head();
    } else {
      return RecCausalLM::get_npu_lm_head();
    }
  }

  void set_npu_lm_head(layer::NpuLmHead& head) override {
    if constexpr (detail::has_set_npu_lm_head<Model>::value) {
      model_->set_npu_lm_head(head);
    } else {
      RecCausalLM::set_npu_lm_head(head);
    }
  }

  layer::NpuWordEmbedding get_npu_word_embedding() override {
    if constexpr (detail::has_get_npu_word_embedding<Model>::value) {
      return model_->get_npu_word_embedding();
    } else {
      return RecCausalLM::get_npu_word_embedding();
    }
  }

  void set_npu_word_embedding(layer::NpuWordEmbedding& embedding) override {
    if constexpr (detail::has_set_npu_word_embedding<Model>::value) {
      model_->set_npu_word_embedding(embedding);
    } else {
      RecCausalLM::set_npu_word_embedding(embedding);
    }
  }

  bool init_or_refresh_rolling_runtime(Stream* load_stream,
                                       Stream* compute_stream,
                                       int32_t num_cached_slots,
                                       int32_t requested_rolling_slots,
                                       const std::string& model_id) override {
    if constexpr (detail::has_init_or_refresh_rolling_runtime<Model>::value) {
      return model_->init_or_refresh_rolling_runtime(load_stream,
                                                     compute_stream,
                                                     num_cached_slots,
                                                     requested_rolling_slots,
                                                     model_id);
    }
    return RecCausalLM::init_or_refresh_rolling_runtime(load_stream,
                                                        compute_stream,
                                                        num_cached_slots,
                                                        requested_rolling_slots,
                                                        model_id);
  }
#endif

  layer::LmHead get_lm_head() override {
    if constexpr (detail::has_get_lm_head<Model>::value) {
      return model_->get_lm_head();
    } else {
      return RecCausalLM::get_lm_head();
    }
  }

  void set_lm_head(layer::LmHead& head) override {
    if constexpr (detail::has_set_lm_head<Model>::value) {
      model_->set_lm_head(head);
    } else {
      RecCausalLM::set_lm_head(head);
    }
  }

  layer::WordEmbedding get_word_embedding() override {
    if constexpr (detail::has_get_word_embedding<Model>::value) {
      return model_->get_word_embedding();
    } else {
      return RecCausalLM::get_word_embedding();
    }
  }

  void set_word_embedding(layer::WordEmbedding& embedding) override {
    if constexpr (detail::has_set_word_embedding<Model>::value) {
      model_->set_word_embedding(embedding);
    } else {
      RecCausalLM::set_word_embedding(embedding);
    }
  }

  torch::Device device() const override { return options_.device(); }

  const torch::TensorOptions& options() const override { return options_; }

 private:
  Model model_;
  torch::TensorOptions options_;
};

}  // namespace xllm

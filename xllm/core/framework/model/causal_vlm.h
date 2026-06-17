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

#include <c10/core/Device.h>
#include <torch/torch.h>

#include <vector>

#include "causal_lm.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/quant_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "model_args.h"
#include "model_input_params.h"

namespace xllm {

class CausalVLM : public CausalLM {
 public:
  ~CausalVLM() override = default;
  virtual MMDict encode(const ModelInputParams& parameters) = 0;
  virtual torch::Tensor get_input_embeddings(
      const torch::Tensor& input_ids,
      const ModelInputParams& input_params) = 0;
};

template <typename Model>
class CausalVLMImpl : public CausalVLM {
 public:
  CausalVLMImpl(Model model, const torch::TensorOptions& options)
      : model_(std::move(model)), options_(options) {}

  MMDict encode(const ModelInputParams& parameters) override {
    return model_->get_multimodal_embeddings(parameters);
  }

  torch::Tensor get_input_embeddings(
      const torch::Tensor& input_ids,
      const ModelInputParams& input_params) override {
    return model_->get_input_embeddings(input_ids, input_params);
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& parameters) override {
    return model_->forward(tokens, positions, kv_caches, parameters);
  }

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override {
    if constexpr (detail::has_pooler<Model>::value) {
      return model_->pooler(hidden_states, seleted_idxes);
    } else {
      return CausalLM::pooler(hidden_states, seleted_idxes);
    }
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override {
    return model_->logits(hidden_states, seleted_idxes);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) override {
    model_->load_model(std::move(loader));
  }

  virtual void prepare_expert_weight(int32_t layer_id,
                                     const std::vector<int32_t>& expert_ids) {
    return;
  }

  virtual void update_expert_weight(int32_t layer_id) { return; }

#if defined(USE_NPU)
  layer::NpuLmHead get_npu_lm_head() override {
    if constexpr (detail::has_get_npu_lm_head<Model>::value) {
      return model_->get_npu_lm_head();
    } else {
      return CausalLM::get_npu_lm_head();
    }
  }

  void set_npu_lm_head(layer::NpuLmHead& head) override {
    if constexpr (detail::has_set_npu_lm_head<Model>::value) {
      model_->set_npu_lm_head(head);
    } else {
      CausalLM::set_npu_lm_head(head);
    }
  }

  layer::NpuWordEmbedding get_npu_word_embedding() override {
    if constexpr (detail::has_get_npu_word_embedding<Model>::value) {
      return model_->get_npu_word_embedding();
    } else {
      return CausalLM::get_npu_word_embedding();
    }
  }

  void set_npu_word_embedding(layer::NpuWordEmbedding& embedding) override {
    if constexpr (detail::has_set_npu_word_embedding<Model>::value) {
      model_->set_npu_word_embedding(embedding);
    } else {
      CausalLM::set_npu_word_embedding(embedding);
    }
  }
#endif
  layer::LmHead get_lm_head() override {
    if constexpr (detail::has_get_lm_head<Model>::value) {
      return model_->get_lm_head();
    } else {
      return CausalLM::get_lm_head();
    }
  }

  void set_lm_head(layer::LmHead& head) override {
    if constexpr (detail::has_set_lm_head<Model>::value) {
      model_->set_lm_head(head);
    } else {
      CausalLM::set_lm_head(head);
    }
  }

  layer::WordEmbedding get_word_embedding() override {
    if constexpr (detail::has_get_word_embedding<Model>::value) {
      return model_->get_word_embedding();
    } else {
      return CausalLM::get_word_embedding();
    }
  }

  void set_word_embedding(layer::WordEmbedding& embedding) override {
    if constexpr (detail::has_set_word_embedding<Model>::value) {
      model_->set_word_embedding(embedding);
    } else {
      CausalLM::set_word_embedding(embedding);
    }
  }

  torch::Device device() const override { return options_.device(); }

  const torch::TensorOptions& options() const override { return options_; }

 private:
  Model model_;
  torch::TensorOptions options_;
};

}  // namespace xllm

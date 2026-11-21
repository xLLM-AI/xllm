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
#include "core/framework/model_loader/model_loader.h"
#include "core/framework/model_loader/model_weight_loader.h"
#include "core/framework/quantization/quant_args.h"
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

 protected:
  using CausalLM::CausalLM;
};

template <typename ModelType>
class CausalVLMImpl : public CausalVLM {
 public:
  CausalVLMImpl(ModelType model, const ModelContext& context)
      : CausalVLM(model->language_model().make_lm_head(context)),
        model_(std::move(model)),
        options_(context.get_tensor_options()),
        encoder_embedding_mode_(
            context.get_model_args().encoder_embedding_mode()) {}

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
    if constexpr (requires {
                    model_->forward(tokens, positions, kv_caches, parameters);
                  }) {
      return model_->forward(tokens, positions, kv_caches, parameters);
    } else {
      return model_->language_model().forward(
          tokens, positions, kv_caches, parameters);
    }
  }

  bool is_hybrid_linear_attention() override {
    if constexpr (requires {
                    model_->language_model().is_hybrid_linear_attention();
                  }) {
      return model_->language_model().is_hybrid_linear_attention();
    }
    return CausalLM::is_hybrid_linear_attention();
  }

  bool supports_mla_graph_kv_bucketing() const override {
    if constexpr (requires { model_->supports_mla_graph_kv_bucketing(); }) {
      return model_->supports_mla_graph_kv_bucketing();
    }
    return CausalLM::supports_mla_graph_kv_bucketing();
  }

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override {
    if constexpr (requires { model_->pooler(hidden_states, seleted_idxes); }) {
      return model_->pooler(hidden_states, seleted_idxes);
    }
    return CausalLM::pooler(hidden_states, seleted_idxes);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) override {
    load_model_weights(*model_,
                       loader->get_state_dicts(),
                       *lm_head_.target<layer::LmHead>(),
                       ModelType::ContainedType::kLanguageWeightPrefix,
                       encoder_embedding_mode_);
  }

  void prepare_expert_weight(int32_t, const std::vector<int32_t>&) override {}
  void update_expert_weight(int32_t) override {}

  layer::WordEmbedding get_word_embedding() override {
    return model_->language_model().get_word_embedding();
  }

  torch::Device device() const override { return options_.device(); }

  const torch::TensorOptions& options() const override { return options_; }

 private:
  ModelType model_;
  torch::TensorOptions options_;
  bool encoder_embedding_mode_;
};

}  // namespace xllm

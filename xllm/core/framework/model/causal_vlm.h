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

#include <type_traits>
#include <utility>

#include "causal_lm.h"
#include "core/framework/model/model_weight_loader.h"
#include "core/framework/model_loader/model_loader.h"
#include "core/framework/quantization/quant_args.h"
#include "core/kv_cache/storage/kv_cache.h"
#include "model_args.h"
#include "model_params_view.h"

namespace xllm {

class CausalVLM : public torch::nn::Module, public Model {
 public:
  ~CausalVLM() override = default;

  virtual CausalLM& language_model() = 0;

  virtual torch::Tensor pooler(const torch::Tensor& hidden_states,
                               const torch::Tensor& pooling_token_indices);
  virtual MMDict encode(const ModelParamsView& parameters) = 0;
  virtual torch::Tensor get_input_embeddings(
      const torch::Tensor& input_ids,
      const ModelParamsView& input_params) = 0;
};

template <typename ModelType>
class CausalVLMImpl final : public CausalVLM {
  using LanguageModel = std::remove_reference_t<
      decltype(std::declval<ModelType>()->language_model())>;

 public:
  explicit CausalVLMImpl(const ModelContext& context)
      : model_(context), language_model_(model_->language_model(), context) {
    model_->eval();
  }

  CausalLM& language_model() override { return language_model_; }

  MMDict encode(const ModelParamsView& parameters) override {
    return model_->get_multimodal_embeddings(parameters);
  }

  torch::Tensor get_input_embeddings(
      const torch::Tensor& input_ids,
      const ModelParamsView& input_params) override {
    return model_->get_input_embeddings(input_ids, input_params);
  }

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override {
    if constexpr (requires { model_->pooler(hidden_states, seleted_idxes); }) {
      return model_->pooler(hidden_states, seleted_idxes);
    }
    return language_model_.pooler(hidden_states, seleted_idxes);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) override {
    auto lm_head = language_model_.get_lm_head();
    load_model_weights(*model_, *loader, lm_head);
  }

  torch::Device device() const override { return language_model_.device(); }

  const torch::TensorOptions& options() const override {
    return language_model_.options();
  }

 private:
  ModelType model_;
  CausalLMImpl<LanguageModel> language_model_;
};

}  // namespace xllm

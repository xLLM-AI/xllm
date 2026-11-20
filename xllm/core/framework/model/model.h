/* Copyright 2026 The xLLM Authors.

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
#include <torch/torch.h>

#include <memory>
#include <string>
#include <utility>

#include "common/macros.h"
#include "core/framework/model/model_traits.h"
#include "core/framework/model_loader/model_loader.h"

namespace xllm {

class Model {
 public:
  virtual ~Model() = default;

  virtual std::string executor_type() const { return {}; }

  virtual torch::Tensor pooler(const torch::Tensor& hidden_states,
                               const torch::Tensor& seleted_idxes) {
    if (seleted_idxes.defined()) {
      return hidden_states.index_select(/*dim=*/0, seleted_idxes);
    }
    return hidden_states;
  }

  virtual void load_model(std::unique_ptr<ModelLoader> loader) = 0;

  virtual torch::Device device() const = 0;

  virtual const torch::TensorOptions& options() const = 0;

  virtual void lazy_load_model(std::unique_ptr<ModelLoader> loader) {
    NOT_IMPLEMENTED();
  }

  virtual void free_model_weights() { NOT_IMPLEMENTED(); }

  virtual void reload_model_weights() { NOT_IMPLEMENTED(); }

  virtual void reload_model_weights_from_device() { NOT_IMPLEMENTED(); }
};

template <typename ModelType, typename Interface>
class ModelImpl : public Interface {
 public:
  ModelImpl(ModelType model, const torch::TensorOptions& options)
      : model_(std::move(model)), options_(options) {}

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override {
    if constexpr (detail::has_pooler<ModelType>::value) {
      return model_->pooler(hidden_states, seleted_idxes);
    }
    return Interface::pooler(hidden_states, seleted_idxes);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) override {
    model_->load_model(std::move(loader));
  }

  torch::Device device() const override { return options_.device(); }

  const torch::TensorOptions& options() const override { return options_; }

  void lazy_load_model(std::unique_ptr<ModelLoader> loader) override {
    if constexpr (detail::has_lazy_load_model<ModelType>::value) {
      model_->lazy_load_model(std::move(loader));
    } else {
      Interface::lazy_load_model(std::move(loader));
    }
  }

  void free_model_weights() override {
    if constexpr (detail::has_free_model_weights<ModelType>::value) {
      model_->free_model_weights();
    } else {
      Interface::free_model_weights();
    }
  }

  void reload_model_weights() override {
    if constexpr (detail::has_reload_model_weights<ModelType>::value) {
      model_->reload_model_weights();
    } else {
      Interface::reload_model_weights();
    }
  }

  void reload_model_weights_from_device() override {
    if constexpr (detail::has_reload_model_weights_from_device<
                      ModelType>::value) {
      model_->reload_model_weights_from_device();
    } else {
      Interface::reload_model_weights_from_device();
    }
  }

 protected:
  ModelType model_;

 private:
  torch::TensorOptions options_;
};

}  // namespace xllm

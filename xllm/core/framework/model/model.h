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

#include "common/macros.h"
#include "core/framework/model_loader/model_loader.h"
#include "core/layers/common/lm_head.h"
#include "core/layers/common/word_embedding.h"

namespace xllm {

// Input-independent model weights and runtime resources. Forward contracts stay
// on the language-model interfaces.
class Model {
 public:
  virtual ~Model() = default;

  virtual void load_model(std::unique_ptr<ModelLoader> loader) = 0;

  virtual torch::Device device() const = 0;

  virtual const torch::TensorOptions& options() const = 0;

  virtual layer::LmHead get_lm_head() {
    NOT_IMPLEMENTED();
    return nullptr;
  }
  virtual void set_lm_head(layer::LmHead& head) { NOT_IMPLEMENTED(); }
  virtual layer::WordEmbedding get_word_embedding() {
    NOT_IMPLEMENTED();
    return nullptr;
  }

  virtual void set_word_embedding(layer::WordEmbedding& embedding) {
    NOT_IMPLEMENTED();
  }

  virtual void lazy_load_model(std::unique_ptr<ModelLoader> loader) {
    NOT_IMPLEMENTED();
  }

  virtual void free_model_weights() { NOT_IMPLEMENTED(); }

  virtual void reload_model_weights() { NOT_IMPLEMENTED(); }

  virtual void reload_model_weights_from_device() { NOT_IMPLEMENTED(); }
};

}  // namespace xllm

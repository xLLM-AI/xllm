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

#include <glog/logging.h>
#include <torch/torch.h>

#include <cmath>
#include <memory>
#include <vector>

#include "core/framework/model/lm_head_owner.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model/rec_model_params.h"
#include "core/framework/model_context/model_context.h"
#include "core/framework/model_loader/model_loader.h"
#include "core/kv_cache/storage/kv_cache.h"
#include "core/layers/common/lm_head.h"
#include "core/layers/common/word_embedding.h"

namespace xllm {

// T5 convention: the tied vocabulary head multiplies the decoder output by
// 1/sqrt(hidden_size). Shared by the stacked and flat OneRec decoders.
inline float t5_head_scale(int64_t hidden_size) {
  return 1.0f /
         std::sqrt(static_cast<float>(std::max<int64_t>(1, hidden_size)));
}

template <typename ModelType>
class GenericRecImplBase : public torch::nn::Module,
                           public detail::LmHeadOwner<> {
 public:
  using HeadOwner = detail::LmHeadOwner<>;

  explicit GenericRecImplBase(const ModelContext& context) {
    model_ = register_module("model", ModelType(context));
    this->init_lm_head(*this, context);
  }

  virtual ModelOutput forward(const torch::Tensor& tokens,
                              const torch::Tensor& positions,
                              std::vector<KVCache>& kv_caches,
                              const RecModelParams& input_params) {
    return model_->forward(tokens, positions, kv_caches, input_params);
  }

  void load_lm_head_weights(const StateDict& state_dict,
                            layer::LmHead& head,
                            bool tie_word_embeddings) override {
    if (tie_word_embeddings) {
      head->load_state_dict(state_dict.get_dict_with_prefix("model.shared."));
    } else {
      head->load_state_dict(state_dict.get_dict_with_prefix("lm_head."));
    }
  }

  virtual void load_model(std::unique_ptr<ModelLoader> loader,
                          std::string prefix = "model.") {
    HeadOwner::load_lm_head_weights(loader);
    for (const auto& state_dict : loader->get_state_dicts()) {
      model_->load_state_dict(state_dict->get_dict_with_prefix(prefix));
    }
  }

  virtual layer::WordEmbedding get_word_embedding() {
    return model_->get_word_embedding();
  }

  virtual void set_word_embedding(layer::WordEmbedding& embedding) {
    model_->set_word_embedding(embedding);
  }

 protected:
  ModelType model_{nullptr};
};

}  // namespace xllm

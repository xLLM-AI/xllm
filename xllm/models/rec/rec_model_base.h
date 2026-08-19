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

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_context.h"
#include "core/framework/model_loader.h"
#include "core/layers/common/lm_head.h"
#include "core/layers/common/word_embedding.h"

namespace xllm {

template <typename ModelType>
class RecForCausalLMImplBase : public torch::nn::Module {
 public:
  explicit RecForCausalLMImplBase(const ModelContext& context) {
    const auto& args = context.get_model_args();
    tie_word_embeddings_ = args.tie_word_embeddings();
    const float denom =
        std::sqrt(static_cast<float>(std::max<int64_t>(1, args.hidden_size())));
    scale_factor_ = denom > 0.0f ? (1.0f / denom) : 1.0f;

    model_ = register_module("model", ModelType(context));
    lm_head_ = register_module("lm_head", layer::LmHead(context));
  }

  virtual ModelOutput forward(const torch::Tensor& tokens,
                              const torch::Tensor& positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelInputParams& input_params) {
    return model_->forward(tokens, positions, kv_caches, input_params);
  }

  virtual torch::Tensor logits(const torch::Tensor& hidden_states,
                               const torch::Tensor& selected_idxes) {
    auto h = hidden_states;
    if (tie_word_embeddings_) {
      h = hidden_states * scale_factor_;
    }
    if (selected_idxes.defined()) {
      h = h.index_select(/*dim=*/0, selected_idxes);
    }
    return lm_head_(h);
  }

  virtual torch::Tensor pooler(const torch::Tensor& hidden_states,
                               const torch::Tensor& selected_idxes) {
    if (selected_idxes.defined()) {
      return hidden_states.index_select(/*dim=*/0, selected_idxes);
    }
    return hidden_states;
  }

  virtual void load_model(std::unique_ptr<ModelLoader> loader,
                          std::string prefix = "model.") {
    for (const auto& state_dict : loader->get_state_dicts()) {
      model_->load_state_dict(state_dict->get_dict_with_prefix(prefix));
      if (tie_word_embeddings_) {
        lm_head_->load_state_dict(
            state_dict->get_dict_with_prefix(prefix + "shared."));
      } else {
        lm_head_->load_state_dict(state_dict->get_dict_with_prefix("lm_head."));
      }
    }
  }

  virtual void prepare_expert_weight(int32_t layer_id,
                                     const std::vector<int32_t>& expert_ids) {
    (void)layer_id;
    (void)expert_ids;
  }

  virtual void update_expert_weight(int32_t layer_id) { (void)layer_id; }

  virtual layer::LmHead get_lm_head() { return lm_head_; }

  virtual void set_lm_head(layer::LmHead& head) { lm_head_ = head; }

  virtual layer::WordEmbedding get_word_embedding() {
    return model_->get_word_embedding();
  }

  virtual void set_word_embedding(layer::WordEmbedding& embedding) {
    model_->set_word_embedding(embedding);
  }

 protected:
  float scale_factor_ = 1.0f;
  bool tie_word_embeddings_ = false;

  ModelType model_{nullptr};
  layer::LmHead lm_head_{nullptr};
};

}  // namespace xllm

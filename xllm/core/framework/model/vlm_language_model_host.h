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

#include <vector>

#include "core/framework/model/model_output.h"
#include "core/framework/model/model_params_view.h"
#include "core/framework/model/model_traits.h"
#include "core/kv_cache/storage/kv_cache.h"

namespace xllm {

namespace layer {
class LmHead;
class WordEmbedding;
#if defined(USE_NPU)
class NpuLmHead;
class NpuWordEmbedding;
#endif
}  // namespace layer

namespace detail {

// Mixin for VLM wrappers that host a causal-LM language model. Provides the
// delegation surface every language tower offers (forward / logits / pooler /
// embedding and head accessors) so each VLM family only hand-writes its
// vision-side logic. CausalLMImpl probes these through its has_* traits. The
// owner registers its language model into the inherited `language_model_`
// holder (register_module on the final host keeps it in the host's module
// tree). The NPU/CPU accessor variants are guarded by requires-clauses so
// each instantiation only exposes the flavor its language tower provides.
template <typename LanguageModel>
class VlmLanguageModelHost {
 public:
  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelParamsView& input_params) {
    return language_model_->forward(tokens, positions, kv_caches, input_params);
  }

  // hidden_states: [num_tokens, hidden_size]; selected_idxes: [num_tokens] or
  // undefined; out_hidden (optional export): [num_selected_tokens,
  // hidden_size] gathered rows; returns [num_selected_tokens, vocab_size].
  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes,
                       torch::Tensor& out_hidden) {
    return language_model_->logits(hidden_states, seleted_idxes, out_hidden);
  }

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    return language_model_->pooler(hidden_states, seleted_idxes);
  }

  layer::WordEmbedding get_word_embedding()
    requires has_get_word_embedding<LanguageModel>::value
  {
    return language_model_->get_word_embedding();
  }

  void set_word_embedding(layer::WordEmbedding& embedding)
    requires has_set_word_embedding<LanguageModel>::value
  {
    language_model_->set_word_embedding(embedding);
  }

#if defined(USE_NPU)
  layer::NpuWordEmbedding get_npu_word_embedding()
    requires has_get_npu_word_embedding<LanguageModel>::value
  {
    return language_model_->get_npu_word_embedding();
  }

  void set_npu_word_embedding(layer::NpuWordEmbedding& embedding)
    requires has_set_npu_word_embedding<LanguageModel>::value
  {
    language_model_->set_npu_word_embedding(embedding);
  }

  layer::NpuLmHead get_npu_lm_head()
    requires has_family_get_npu_lm_head<LanguageModel>::value
  {
    return language_model_->get_npu_lm_head();
  }
#endif

 protected:
  // The registered language model holder (TORCH_MODULE holders expose
  // operator->).
  LanguageModel language_model_{nullptr};
};

}  // namespace detail
}  // namespace xllm

/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <memory>
#include <utility>
#include <vector>

#include "models/vlm/npu/qwen3_omni_moe_thinker.h"
#include "processors/qwen3_omni_moe_processor.h"

namespace xllm::npu::model {

class Qwen3OmniMoeForConditionalGenerationImpl final
    : public torch::nn::Module {
 public:
  explicit Qwen3OmniMoeForConditionalGenerationImpl(
      const ModelContext& context) {
    thinker_ = register_module(
        "thinker", Qwen3OmniMoeThinkerForConditionalGeneration(context));
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    torch::NoGradGuard no_grad;
    return thinker_(tokens, positions, kv_caches, input_params);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_indices) {
    return thinker_->logits(hidden_states, selected_indices);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    thinker_->load_model(std::move(loader));
  }

  torch::Tensor get_input_embeddings(const torch::Tensor input_ids,
                                     const ModelInputParams& input_params) {
    return thinker_->get_input_embeddings(input_ids, input_params);
  }

  MMDict get_multimodal_embeddings(const ModelInputParams& input_params) {
    return thinker_->get_multimodal_embeddings(input_params);
  }
  layer::NpuLmHead get_npu_lm_head() { return thinker_->get_npu_lm_head(); }

  void set_npu_lm_head(layer::NpuLmHead& head) {
    thinker_->set_npu_lm_head(head);
  }

  layer::NpuWordEmbedding get_npu_word_embedding() {
    return thinker_->get_npu_word_embedding();
  }

  void set_npu_word_embedding(layer::NpuWordEmbedding& npu_word_embedding) {
    thinker_->set_npu_word_embedding(npu_word_embedding);
  }

 private:
  Qwen3OmniMoeThinkerForConditionalGeneration thinker_{nullptr};
};
TORCH_MODULE(Qwen3OmniMoeForConditionalGeneration);

REGISTER_MULTIMODAL_PROCESSOR(qwen3_omni_moe, Qwen3OmniMoeMultimodalProcessor);
REGISTER_CAUSAL_VLM_MODEL(qwen3_omni_moe, Qwen3OmniMoeForConditionalGeneration);
REGISTER_MPOSITION_GENERATOR(qwen3_omni_moe, xllm::Qwen3VLMPositionGenerator);

REGISTER_MODEL_ARGS(qwen3_omni_moe,
                    [&] { load_qwen3_omni_moe_model_args(json, args); });

}  // namespace xllm::npu::model

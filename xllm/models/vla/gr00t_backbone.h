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

#include "core/layers/common/dense_mlp.h"
#include "core/layers/common/rms_norm.h"
#include "models/llm/qwen3.h"
#include "models/vlm/qwen3_vl.h"

namespace xllm {

// The policy checkpoint already determines the truncated decoder depth. GR00T
// consumes every token before the final RMSNorm, including the residual stream.
class Gr00tLanguageModelImpl final : public QWen3ModelImpl {
 public:
  explicit Gr00tLanguageModelImpl(const ModelContext& context)
      : QWen3ModelImpl(context, /*precompute_rotary_cache=*/false),
        options_(context.get_tensor_options()) {
    const auto& args = context.get_model_args();
    const int64_t head_dim = args.head_dim();
    auto frequencies =
        1.0 / torch::pow(args.rope_theta(),
                         torch::arange(
                             0, head_dim, 2, options_.dtype(torch::kFloat32)) /
                             head_dim);
    // The reference casts its nonpersistent RoPE buffer with the model, then
    // promotes that rounded buffer to FP32 when computing token frequencies.
    inv_freq_ = frequencies.to(options_).to(torch::kFloat32);
    CHECK_EQ(mrope_section_.size(), 3);
    // QWen3 stores decoder holders in a vector rather than registering them
    // as module children, so configure each actual decoder subtree explicitly.
    for (const auto& decoder : layers_) {
      for (const auto& module : decoder->modules(/*include_self=*/false)) {
        auto norm = std::dynamic_pointer_cast<layer::RMSNormImpl>(module);
        if (norm) {
          norm->set_hf_precision_mode();
        }
        auto mlp = std::dynamic_pointer_cast<layer::DenseMLPImpl>(module);
        if (mlp) {
          mlp->set_hf_precision_mode();
        }
      }
    }
  }

  std::pair<torch::Tensor, torch::Tensor> apply_mrope(
      const torch::Tensor positions) override {
    auto frequencies = positions.to(torch::kFloat32).unsqueeze(-1) * inv_freq_;
    auto interleaved = frequencies[0].clone();
    for (int64_t axis = 1; axis < 3; ++axis) {
      auto indices = torch::arange(
          axis, mrope_section_[axis] * 3, 3, options_.dtype(torch::kInt64));
      interleaved.index_copy_(
          -1, indices, frequencies[axis].index_select(-1, indices));
    }
    auto embedding = torch::cat({interleaved, interleaved}, -1);
    return {embedding.cos().to(options_), embedding.sin().to(options_)};
  }

 protected:
  void add_deepstack(torch::Tensor& hidden_states,
                     std::optional<torch::Tensor>& residual,
                     const torch::Tensor& deep_stack) override {
    // HF completes the decoder residual addition before adding DeepStack.
    // Reordering these two BF16 additions changes the trained features.
    if (residual.has_value()) {
      hidden_states = hidden_states + residual.value();
      residual = std::nullopt;
    }
    hidden_states = hidden_states + deep_stack;
  }

  ModelOutput finalize_hidden_states(
      torch::Tensor& hidden_states,
      const std::optional<torch::Tensor>& residual) override {
    return ModelOutput(residual.has_value() ? hidden_states + residual.value()
                                            : hidden_states);
  }

 private:
  torch::TensorOptions options_;
  torch::Tensor inv_freq_;
};
TORCH_MODULE(Gr00tLanguageModel);

// A feature extractor, deliberately separate from the chat model registry: it
// has no vocabulary head and must never enter the logits/sampling path.
class Gr00tBackboneImpl final : public torch::nn::Module {
 public:
  explicit Gr00tBackboneImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    visual_ = register_module(
        "visual",
        Qwen3_VisionTransformer(context, /*hf_buffer_precision=*/true));
    language_model_ =
        register_module("language_model", Gr00tLanguageModel(context));
  }

  torch::Tensor encode(const torch::Tensor& pixel_values,
                       const torch::Tensor& image_grid_thw) {
    return visual_->forward(pixel_values.to(options_),
                            image_grid_thw.to(options_.device()));
  }

  torch::Tensor embed_tokens(const torch::Tensor& input_ids) {
    return language_model_->get_word_embedding()->forward(input_ids);
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelParamsView& params) {
    return language_model_->forward(tokens, positions, kv_caches, params);
  }

  void load_state_dict(const StateDict& state_dict) {
    visual_->load_state_dict(
        state_dict.get_dict_with_prefix("backbone.model.model.visual."));
    language_model_->load_state_dict(state_dict.get_dict_with_prefix(
        "backbone.model.model.language_model."));
  }

 private:
  torch::TensorOptions options_;
  Qwen3_VisionTransformer visual_{nullptr};
  Gr00tLanguageModel language_model_{nullptr};
};
TORCH_MODULE(Gr00tBackbone);

}  // namespace xllm

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

#include "core/framework/model/model_output.h"
#include "core/framework/multimodal/mm_data_item.h"
#include "core/layers/common/lm_head.h"
#include "models/model_registry.h"

namespace xllm {

struct Qwen3_VLImageInputs {
  torch::Tensor pixel_values;
  torch::Tensor image_grid_thw;
};

struct Qwen3_VLVideoInputs {
  torch::Tensor pixel_values_videos;
  torch::Tensor video_grid_thw;
  torch::Tensor second_per_grid_ts;
};

template <typename VisionTransformer, typename LanguageModel>
class Qwen3VLForConditionalGenerationBase : public torch::nn::Module {
 public:
  Qwen3VLForConditionalGenerationBase(const ModelContext& context)
      : model_args_(context.get_model_args()),
        options_(context.get_tensor_options()) {
    visual_ = register_module("visual", VisionTransformer(context));
    language_model_ = register_module("language_model", LanguageModel(context));
  }

  virtual ~Qwen3VLForConditionalGenerationBase() = default;

  void prepare_encoder_input(const ModelInputParams& input_params,
                             std::optional<Qwen3_VLImageInputs>& image_inputs,
                             std::optional<Qwen3_VLVideoInputs>& video_inputs) {
    const auto& mm_data = input_params.multimodal.mm_data;
    torch::Tensor pixel_values;
    if (const auto& res = mm_data.get<torch::Tensor>("pixel_values"))
      pixel_values = res.value();

    torch::Tensor image_grid_thw;
    if (const auto& res = mm_data.get<torch::Tensor>("image_grid_thw"))
      image_grid_thw = res.value();

    torch::Tensor pixel_values_videos;
    if (const auto& res = mm_data.get<torch::Tensor>("pixel_values_videos"))
      pixel_values_videos = res.value();

    torch::Tensor video_grid_thw;
    if (const auto& res = mm_data.get<torch::Tensor>("video_grid_thw"))
      video_grid_thw = res.value();

    if (pixel_values.defined() && image_grid_thw.defined())
      image_inputs = Qwen3_VLImageInputs{pixel_values, image_grid_thw};

    if (pixel_values_videos.defined() && video_grid_thw.defined())
      video_inputs = Qwen3_VLVideoInputs{pixel_values_videos, video_grid_thw};
  }

  MMDict get_multimodal_embeddings(const ModelInputParams& input_params) {
    std::optional<Qwen3_VLImageInputs> image_input;
    std::optional<Qwen3_VLVideoInputs> video_input;
    prepare_encoder_input(input_params, image_input, video_input);

    MMDict multimodal_embeds;
    auto merge_size = model_args_.mm_image_merge_size();
    if (image_input) {
      auto image_embeds =
          visual_(image_input->pixel_values.to(options_),
                  image_input->image_grid_thw.to(options_.device()));

      auto image_tokens =
          (image_input->image_grid_thw.prod(-1) / merge_size / merge_size)
              .cpu()
              .contiguous()
              .to(torch::kLong);

      std::vector<int64_t> image_tokens_vec(
          image_tokens.data_ptr<int64_t>(),
          image_tokens.data_ptr<int64_t>() + image_tokens.numel());
      multimodal_embeds["image|embedding"] =
          image_embeds.split(image_tokens_vec, /*dim=*/0);
    }
    if (video_input) {
      auto video_embeds =
          visual_(video_input->pixel_values_videos.to(options_),
                  video_input->video_grid_thw.to(options_.device()));

      auto video_tokens =
          (video_input->video_grid_thw.prod(-1) / merge_size / merge_size)
              .cpu()
              .contiguous()
              .to(torch::kLong);

      std::vector<int64_t> video_tokens_vec(
          video_tokens.data_ptr<int64_t>(),
          video_tokens.data_ptr<int64_t>() + video_tokens.numel());
      multimodal_embeds["video|embedding"] =
          video_embeds.split(video_tokens_vec, /*dim=*/0);
    }
    return multimodal_embeds;
  }

  std::pair<torch::Tensor, std::vector<torch::Tensor>>
  split_multimodal_embedding(const torch::Tensor& embedding) const {
    const size_t num_deepstacks =
        model_args_.mm_deepstack_visual_indexes().size();
    CHECK(embedding.defined()) << "Multimodal embedding is not defined.";
    const int64_t num_chunks = static_cast<int64_t>(num_deepstacks + 1);
    auto chunks = embedding.chunk(/*chunks=*/num_chunks, /*dim=*/1);
    std::vector<torch::Tensor> deepstacks;
    deepstacks.reserve(num_deepstacks);
    for (size_t idx = 1; idx < chunks.size(); ++idx) {
      deepstacks.push_back(chunks[idx]);
    }
    return {chunks[0], std::move(deepstacks)};
  }

  torch::Tensor merge_multimodal_embeddings(
      torch::Tensor inputs_embeds,
      const torch::Tensor& multimodal_embeds,
      const torch::Tensor& is_multimodal) {
    inputs_embeds.index_put_({is_multimodal}, multimodal_embeds);
    return inputs_embeds;
  }

  torch::Tensor get_input_embeddings(const torch::Tensor input_ids,
                                     const ModelInputParams& input_params) {
    const auto& mm_data = input_params.multimodal.mm_data;
    auto inputs_embeds = language_model_->get_input_embeddings(input_ids);
    const size_t num_deepstacks =
        model_args_.mm_deepstack_visual_indexes().size();
    std::vector<torch::Tensor> deepstack_input_embeds;
    deepstack_input_embeds.resize(num_deepstacks);
    for (auto& deepstack : deepstack_input_embeds) {
      deepstack = torch::zeros_like(inputs_embeds);
    }
    auto merge_modality = [&](const std::string& embed_key,
                              const std::string& mask_key) {
      auto emb = mm_data.get<torch::Tensor>(embed_key);
      if (!emb.has_value()) return;
      auto mask = mm_data.get<torch::Tensor>(mask_key);
      if (!mask.has_value()) return;
      auto [embedding, deepstacks] = split_multimodal_embedding(emb.value());
      inputs_embeds =
          merge_multimodal_embeddings(inputs_embeds, embedding, mask.value());
      for (size_t idx = 0; idx < num_deepstacks; ++idx) {
        deepstack_input_embeds[idx] = merge_multimodal_embeddings(
            deepstack_input_embeds[idx], deepstacks[idx], mask.value());
      }
    };

    merge_modality("image|embedding", "image|mask");
    merge_modality("video|embedding", "video|mask");
    input_params.multimodal.deep_stacks = std::move(deepstack_input_embeds);
    return inputs_embeds;
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    return language_model_(tokens, positions, kv_caches, input_params);
  }

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    return language_model_->pooler(hidden_states, seleted_idxes);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    return language_model_->logits(hidden_states, seleted_idxes);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      visual_->load_state_dict(
          state_dict->get_dict_with_prefix("model.visual."));
    }

#if defined(USE_NPU)
    visual_->verify_loaded_weights("model.visual.");
    visual_->merge_loaded_weights();
#endif

    if (!model_args_.encoder_embedding_mode()) {
      language_model_->load_model(std::move(loader), "model.language_model.");
    }
  }

  layer::LmHead get_lm_head() { return language_model_->get_lm_head(); }
  void set_lm_head(layer::LmHead& head) { language_model_->set_lm_head(head); }

  layer::WordEmbedding get_word_embedding() {
    return language_model_->get_word_embedding();
  }

  void set_word_embedding(layer::WordEmbedding& word_embedding) {
    language_model_->set_word_embedding(word_embedding);
  }

 protected:
  ModelArgs model_args_;
  torch::TensorOptions options_;
  VisionTransformer visual_{nullptr};
  LanguageModel language_model_{nullptr};
};

}  // namespace xllm

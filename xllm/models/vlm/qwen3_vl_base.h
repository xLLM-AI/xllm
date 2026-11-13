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

#include "core/framework/config/kernel_config.h"
#include "core/framework/model/model_output.h"
#include "core/framework/multimodal/mm_data_item.h"
#include "models/model_registry.h"

namespace xllm {

struct Qwen3_VLImageInputs {
  torch::Tensor pixel_values;
  torch::Tensor image_grid_thw;
};

struct Qwen3_VLVideoInputs {
  torch::Tensor pixel_values_videos;
  torch::Tensor video_grid_thw;
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

  void prepare_encoder_input(const ModelParamsView& input_params,
                             std::optional<Qwen3_VLImageInputs>& image_inputs,
                             std::optional<Qwen3_VLVideoInputs>& video_inputs) {
    const auto& mm_data = input_params.multimodal().mm_data;
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

  MMDict get_multimodal_embeddings(const ModelParamsView& input_params) {
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

  torch::Tensor get_input_embeddings(const torch::Tensor input_ids,
                                     const ModelParamsView& input_params) {
    const auto& mm_data = input_params.multimodal().mm_data;
    auto inputs_embeds = language_model_->get_input_embeddings(input_ids);
    if (!mm_data.valid()) {
      return inputs_embeds;
    }
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
      inputs_embeds.index_put_({mask.value()}, embedding);
      for (size_t idx = 0; idx < num_deepstacks; ++idx) {
        deepstack_input_embeds[idx].index_put_({mask.value()}, deepstacks[idx]);
      }
    };

    merge_modality("image|embedding", "image|mask");
    merge_modality("video|embedding", "video|mask");
    input_params.multimodal().deep_stacks = std::move(deepstack_input_embeds);
    return inputs_embeds;
  }

  const LanguageModel& language_model() const { return language_model_; }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      visual_->load_state_dict(
          state_dict->get_dict_with_prefix("model.visual."));
    }

#if defined(USE_NPU)
    if constexpr (requires {
                    visual_->verify_loaded_weights("model.visual.");
                    visual_->merge_loaded_weights();
                  }) {
      visual_->verify_loaded_weights("model.visual.");
      visual_->merge_loaded_weights();
    }
#endif

    if (!model_args_.encoder_embedding_mode()) {
      language_model_->load_model(std::move(loader), "model.language_model.");
    }
  }

 protected:
  ModelArgs model_args_;
  torch::TensorOptions options_;
  VisionTransformer visual_{nullptr};
  LanguageModel language_model_{nullptr};
};

inline const ModelArgsLoader kQwen3VLMoeArgsLoader = [](const JsonReader& json,
                                                        ModelArgs* args) {
  LOAD_ARG_OR(model_type, "model_type", "qwen3_vl_moe");
  LOAD_ARG_OR(attention_bias, "text_config.attention_bias", false);
  LOAD_ARG_OR(attention_dropout, "attention_dropout", 0.0f);
  LOAD_ARG_OR(bos_token_id, "text_config.bos_token_id", 151643);
  LOAD_ARG_OR(decoder_sparse_step, "text_config.decoder_sparse_step", 1);
  LOAD_ARG_OR(dtype, "text_config.dtype", "bfloat16");
  LOAD_ARG_OR(eos_token_id, "text_config.eos_token_id", 151645);
  LOAD_ARG_OR(hidden_act, "text_config.hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "text_config.hidden_size", 2048);
  LOAD_ARG_OR(initializer_range, "text_config.initializer_range", 0.02);
  LOAD_ARG_OR(intermediate_size, "text_config.intermediate_size", 5632);
  LOAD_ARG_OR(
      max_position_embeddings, "text_config.max_position_embeddings", 128000);
  LOAD_ARG_OR(moe_intermediate_size, "text_config.moe_intermediate_size", 1408);
  LOAD_ARG_OR(norm_topk_prob, "text_config.norm_topk_prob", true);
  LOAD_ARG_OR(n_heads, "text_config.num_attention_heads", 16);
  LOAD_ARG_OR(num_experts, "text_config.num_experts", 128);
  LOAD_ARG_OR(num_experts_per_tok, "text_config.num_experts_per_tok", 8);
  LOAD_ARG_OR(n_layers, "text_config.num_hidden_layers", 24);
  LOAD_ARG_OR(n_kv_heads, "text_config.num_key_value_heads", 16);
  LOAD_ARG_OR_FUNC(head_dim, /*json_name=*/"text_config.head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });
  LOAD_ARG_OR(rms_norm_eps, "text_config.rms_norm_eps", 1e-06);
  LOAD_ARG_OR(rope_scaling_rope_type, "text_config.rope_scaling.type", "mrope");
  LOAD_ARG(rope_scaling_mrope_section,
           /*json_name=*/"text_config.rope_scaling.mrope_section");
  LOAD_ARG_OR(rope_theta, "text_config.rope_theta", 5000000.0f);
  LOAD_ARG_OR(vocab_size, "text_config.vocab_size", 151936);

  LOAD_ARG(mm_deepstack_visual_indexes,
           /*json_name=*/"vision_config.deepstack_visual_indexes");
  LOAD_ARG_OR(mm_num_hidden_layers, "vision_config.depth", 27);
  LOAD_ARG_OR(mm_hidden_act, "vision_config.hidden_act", "gelu_pytorch_tanh");
  LOAD_ARG_OR(mm_hidden_size, "vision_config.hidden_size", 1152);
  LOAD_ARG_OR(mm_num_channels, "vision_config.in_channels", 3);
  LOAD_ARG_OR(mm_initializer_range, "vision_config.initializer_range", 0.02);
  LOAD_ARG_OR(mm_intermediate_size, "vision_config.intermediate_size", 4304);
  LOAD_ARG_OR(mm_num_attention_heads, "vision_config.num_heads", 16);
  LOAD_ARG_OR(mm_num_position_embeddings,
              "vision_config.num_position_embeddings",
              2304);
  LOAD_ARG_OR(mm_projection_dim, "vision_config.out_hidden_size", 3584);
  LOAD_ARG_OR(mm_patch_size, "vision_config.patch_size", 16);
  LOAD_ARG_OR(mm_spatial_merge_size, "vision_config.spatial_merge_size", 2);
  LOAD_ARG_OR(mm_temporal_patch_size, "vision_config.temporal_patch_size", 2);
  LOAD_ARG_OR_FUNC(mm_head_dim, /*json_name=*/"vision_config.head_dim", [&] {
    return args->mm_hidden_size() / args->mm_num_attention_heads();
  });

  LOAD_ARG_OR(image_token_id, "image_token_id", 151655);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_ARG_OR(video_token_id, "video_token_id", 151656);
  LOAD_ARG_OR(vision_end_token_id, "vision_end_token_id", 151653);
  LOAD_ARG_OR(vision_start_token_id, "vision_start_token_id", 151652);
  LOAD_ARG_OR(n_routed_experts, "num_experts", 128);
  args->n_shared_experts() = 0;
  args->scoring_func() = "softmax";
  args->topk_method() = "";
  args->n_group() = -1;
  args->topk_group() = 0;
  args->routed_scaling_factor() = 1.0;
  return true;
};

}  // namespace xllm

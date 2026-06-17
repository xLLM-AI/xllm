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

#include <atb/atb_infer.h>
#include <c10/core/ScalarType.h>
#include <glog/logging.h>
#include <torch/nn/options/vision.h>
#include <torch/torch.h>

#include <unordered_map>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/layers/npu/npu_lm_head_impl.h"
#include "glm4v.h"
#include "models/llm/npu/oxygen.h"
#include "models/model_registry.h"
#include "models/vlm/mposition/mposition.h"
#include "models/vlm/utils/multimodal_utils.h"
#include "processors/qwen2_vl_image_processor.h"
#include "processors/qwen2_vl_video_processor.h"
#include "qwen2_5_vl.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"

namespace xllm::npu::model {
class OxygenvlmForConditionalGenerationImpl : public torch::nn::Module {
 public:
  OxygenvlmForConditionalGenerationImpl(const ModelContext& context)
      : model_args_(context.get_model_args()),
        options_(context.get_tensor_options()) {
    encoder_dp_group_ = context.get_parallel_args().encoder_dp_group_;
    use_encoder_dp_ =
        encoder_dp_group_ != nullptr && encoder_dp_group_->world_size() > 1;
    if (use_encoder_dp_) {
      ModelContext visual_context = context.with_parallel_args(ParallelArgs(
          /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr));
      visual_ =
          register_module("visual", Glm4VisionTransformer(visual_context));
    } else {
      visual_ = register_module("visual", Glm4VisionTransformer(context));
    }

    language_model_ =
        register_module("language_model", OxygenForCausalLM(context));
  }

  void prepare_encoder_input(const ModelInputParams& input_params,
                             std::optional<Glm4VImageInputs>& image_inputs,
                             std::optional<Glm4VVideoInputs>& video_inputs) {
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
      image_inputs = Glm4VImageInputs{pixel_values, image_grid_thw};

    if (pixel_values_videos.defined() && video_grid_thw.defined())
      video_inputs = Glm4VVideoInputs{pixel_values_videos, video_grid_thw};
  }

  MMDict get_multimodal_embeddings(const ModelInputParams& input_params) {
    std::optional<Glm4VImageInputs> image_input;
    std::optional<Glm4VVideoInputs> video_input;
    prepare_encoder_input(input_params, image_input, video_input);

    MMDict multimodal_embeds;
    if (image_input) {
      torch::Tensor image_pixels = image_input->pixel_values.to(options_);
      torch::Tensor image_grid =
          image_input->image_grid_thw.to(options_.device());
      std::vector<int32_t> image_token_nums =
          get_mm_token_nums(input_params.multimodal.mm_data, MMType::IMAGE);
      if (!use_encoder_dp_) {
        auto image_embeds = visual_(image_pixels, image_grid);
        multimodal_embeds["image|embedding"] =
            split_by_token_nums(image_embeds, image_token_nums);
      } else {
        multimodal_embeds["image|embedding"] =
            run_dp_encoder(visual_,
                           image_pixels,
                           image_grid,
                           image_token_nums,
                           model_args_.mm_projection_dim(),
                           encoder_dp_group_);
      }
    }
    if (video_input) {
      torch::Tensor video_pixels =
          video_input->pixel_values_videos.to(options_);
      torch::Tensor video_grid =
          video_input->video_grid_thw.to(options_.device());
      std::vector<int32_t> video_token_nums =
          get_mm_token_nums(input_params.multimodal.mm_data, MMType::VIDEO);
      auto video_embeds = visual_(video_pixels, video_grid);
      multimodal_embeds["video|embedding"] =
          split_by_token_nums(video_embeds, video_token_nums);
    }
    return multimodal_embeds;
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
    auto merge_modality = [&](const std::string& embed_key,
                              const std::string& mask_key) {
      auto emb = mm_data.get<torch::Tensor>(embed_key);
      if (!emb.has_value()) return;
      auto mask = mm_data.get<torch::Tensor>(mask_key);
      if (!mask.has_value()) return;
      inputs_embeds =
          merge_multimodal_embeddings(inputs_embeds, emb.value(), mask.value());
    };

    merge_modality("image|embedding", "image|mask");
    merge_modality("video|embedding", "video|mask");

    return inputs_embeds;
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    auto emb = language_model_(tokens, positions, kv_caches, input_params);
    return emb;
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
    visual_->verify_loaded_weights("model.visual.");
    visual_->merge_loaded_weights();
    if (!model_args_.encoder_embedding_mode()) {
      language_model_->load_model(std::move(loader), "model.language_model.");
    }
  }

  layer::NpuLmHead get_npu_lm_head() {
    return language_model_->get_npu_lm_head();
  }
  void set_npu_lm_head(layer::NpuLmHead& head) {
    language_model_->set_npu_lm_head(head);
  }

  layer::NpuWordEmbedding get_npu_word_embedding() {
    return language_model_->get_npu_word_embedding();
  }

  void set_npu_word_embedding(layer::NpuWordEmbedding& npu_word_embedding) {
    language_model_->set_npu_word_embedding(npu_word_embedding);
  }

 private:
  ModelArgs model_args_;
  torch::TensorOptions options_;
  ProcessGroup* encoder_dp_group_ = nullptr;
  bool use_encoder_dp_ = false;
  Glm4VisionTransformer visual_{nullptr};
  OxygenForCausalLM language_model_{nullptr};
};
TORCH_MODULE(OxygenvlmForConditionalGeneration);

using OxygenVLMultimodalProcessor = MultimodalProcessor<Qwen2VLPromptProcessor,
                                                        Qwen2VLImageProcessor,
                                                        Qwen2VLVideoProcessor>;
REGISTER_MULTIMODAL_PROCESSOR(oxygenvlm, OxygenVLMultimodalProcessor);
REGISTER_CAUSAL_VLM_MODEL(oxygenvlm, OxygenvlmForConditionalGeneration);
REGISTER_MPOSITION_GENERATOR(oxygenvlm, xllm::QwenVLMPositionGenerator);
// register the model args
REGISTER_MODEL_ARGS(oxygenvlm, [&] {
  LOAD_ARG_OR(model_type, "model_type", "oxygenvlm");
  LOAD_ARG_OR(vision_start_token_id, "vision_start_token_id", 151652);
  LOAD_ARG_OR(vision_end_token_id, "vision_end_token_id", 151653);
  LOAD_ARG_OR(vision_token_id, "vision_token_id", 151654);
  LOAD_ARG_OR(video_token_id, "video_token_id", 151656);
  LOAD_ARG_OR(image_token_id, "image_token_id", 151655);

  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);

  // text config
  LOAD_ARG_OR(vocab_size, "text_config.vocab_size", 151936);
  LOAD_ARG_OR(eos_token_id, "text_config.eos_token_id", 151645);
  LOAD_ARG_OR(attention_bias, "text_config.attention_bias", false);
  LOAD_ARG_OR(attention_dropout, "text_config.attention_dropout", 0.0f);
  LOAD_ARG_OR(hidden_act, "text_config.hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "text_config.hidden_size", 5120);
  LOAD_ARG_OR(initializer_range, "text_config.initializer_range", 0.02);
  LOAD_ARG_OR(intermediate_size, "text_config.intermediate_size", 25600);
  LOAD_ARG_OR(
      max_position_embeddings, "text_config.max_position_embeddings", 40960);
  LOAD_ARG_OR(n_heads, "text_config.num_attention_heads", 64);
  LOAD_ARG_OR(head_dim, "text_config.head_dim", 128);

  LOAD_ARG_OR(n_layers, "text_config.num_hidden_layers", 64);
  LOAD_ARG_OR(n_kv_heads, "text_config.num_key_value_heads", 8);
  LOAD_ARG_OR(rms_norm_eps, "text_config.rms_norm_eps", 1e-05);
  LOAD_ARG_OR(dtype, "text_config.dtype", "bfloat16");
  LOAD_ARG_OR(rope_scaling_rope_type, "text_config.rope_scaling.type", "mrope");
  LOAD_ARG(rope_scaling_mrope_section,
           "text_config.rope_scaling.mrope_section");
  LOAD_ARG_OR(rope_theta, "text_config.rope_theta", 1000000);

  // vision config
  LOAD_ARG_OR(mm_num_hidden_layers, "vision_config.depth", 24);
  LOAD_ARG_OR(mm_hidden_act, "vision_config.hidden_act", "silu");
  LOAD_ARG_OR(mm_hidden_size, "vision_config.hidden_size", 1536);
  LOAD_ARG_OR(mm_image_size, "vision_config.image_size", 336);
  LOAD_ARG_OR(mm_num_channels, "vision_config.in_channels", 3);
  LOAD_ARG_OR(
      mm_intermediate_size, "vision_config.projector_hidden_size", 4096);
  LOAD_ARG_OR(mm_num_attention_heads, "vision_config.num_heads", 12);
  LOAD_ARG_OR(mm_projection_dim, "vision_config.out_hidden_size", 5120);
  LOAD_ARG_OR(mm_patch_size, "vision_config.patch_size", 14);
  LOAD_ARG_OR(mm_spatial_merge_size, "vision_config.spatial_merge_size", 2);
  LOAD_ARG_OR(mm_temporal_patch_size, "vision_config.temporal_patch_size", 2);

  LOAD_ARG_OR_FUNC(mm_head_dim, "head_dim", [&] {
    return args->mm_hidden_size() / args->mm_num_attention_heads();
  });
  if (args->rope_scaling_rope_type() == "default")
    args->rope_scaling_rope_type() = "mrope";
});

}  // namespace xllm::npu::model

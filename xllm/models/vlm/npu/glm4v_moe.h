/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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
#include <torch/torch.h>

#include <boost/algorithm/string.hpp>
#include <unordered_map>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_context.h"
#include "core/layers/npu/npu_lm_head_impl.h"
#include "core/layers/npu/npu_rms_norm_impl.h"
#include "glm4v.h"
#include "models/llm/npu/glm4_moe.h"
#include "models/model_registry.h"
#include "models/vlm/mposition/mposition.h"
#include "models/vlm/utils/multimodal_utils.h"
#include "processors/glm4v_image_processor.h"
#include "processors/glm4v_video_processor.h"

namespace xllm::npu::model {

class Glm4vMoeForConditionalGenerationImpl : public torch::nn::Module {
 public:
  Glm4vMoeForConditionalGenerationImpl(const ModelContext& context)
      : model_args_(context.get_model_args()),
        options_(context.get_tensor_options()) {
    encoder_dp_group_ = context.get_parallel_args().encoder_dp_group_;
    use_encoder_dp_ =
        encoder_dp_group_ != nullptr && encoder_dp_group_->world_size() > 1;
    ModelContext vision_context = context.with_parallel_args(ParallelArgs(
        /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr));
    visual_ = register_module("visual", Glm4VisionTransformer(vision_context));

    language_model_ =
        register_module("language_model", Glm4MoeForCausalLM(context));
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
      std::vector<torch::Tensor> temp_frames_hw;
      torch::Tensor video_grid =
          video_input->video_grid_thw.to(options_.device());
      for (int i = 0; i < video_input->video_grid_thw.size(0); ++i) {
        auto t = video_grid[i][0].item<int32_t>();
        auto h = video_grid[i][1].item<int32_t>();
        auto w = video_grid[i][2].item<int32_t>();
        auto repeated_row =
            torch::tensor({1, h, w}).unsqueeze(0).repeat({t, 1});
        temp_frames_hw.push_back(repeated_row);
      }
      auto flatten_video_grid_thw = torch::cat(temp_frames_hw, 0);
      torch::Tensor video_pixels =
          video_input->pixel_values_videos.to(options_);
      auto video_embeds = visual_(video_pixels, flatten_video_grid_thw);
      std::vector<int32_t> video_token_nums =
          get_mm_token_nums(input_params.multimodal.mm_data, MMType::VIDEO);
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
    return language_model_(tokens, positions, kv_caches, input_params);
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
    // verify
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
  Glm4MoeForCausalLM language_model_{nullptr};
};
TORCH_MODULE(Glm4vMoeForConditionalGeneration);

using Glm4vMoeMultimodalProcessor = MultimodalProcessor<GLM4VPromptProcessor,
                                                        Glm4VImageProcessor,
                                                        Glm4VVideoProcessor>;
REGISTER_MULTIMODAL_PROCESSOR(glm4v_moe, Glm4vMoeMultimodalProcessor);
REGISTER_CAUSAL_VLM_MODEL(glm4v_moe, Glm4vMoeForConditionalGeneration);
REGISTER_MPOSITION_GENERATOR(glm4v_moe, xllm::Glm4VMPositionGenerator);
// register the model args
REGISTER_MODEL_ARGS(glm4v_moe, [&] {
  LOAD_ARG_OR(model_type, "model_type", "glm4v_moe");
  LOAD_ARG_OR(image_start_token_id, "image_start_token_id", 151339);
  LOAD_ARG_OR(image_end_token_id, "image_end_token_id", 151340);
  LOAD_ARG_OR(video_start_token_id, "video_start_token_id", 151341);
  LOAD_ARG_OR(video_end_token_id, "video_end_token_id", 151342);
  LOAD_ARG_OR(image_token_id, "image_token_id", 151363);
  LOAD_ARG_OR(video_token_id, "video_token_id", 151364);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);

  // text config
  LOAD_ARG_OR(vocab_size, "text_config.vocab_size", 151552);
  // LOAD_ARG_OR(pad_token_id, "text_config.pad_token_id", 151329);
  LOAD_ARG_OR(
      eos_token_id_vec, "text_config.eos_token_id", std::vector<int>{151329});
  LOAD_ARG_OR_FUNC(head_dim, "text_config.head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });
  LOAD_ARG_OR(attention_bias, "text_config.attention_bias", true);
  LOAD_ARG_OR(attention_dropout, "text_config.attention_dropout", 0.0f);
  LOAD_ARG_OR(first_k_dense_replace, "text_config.first_k_dense_replace", 1);
  LOAD_ARG_OR(hidden_act, "text_config.hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "text_config.hidden_size", 4096);
  LOAD_ARG_OR(initializer_range, "text_config.initializer_range", 0.02);
  LOAD_ARG_OR(intermediate_size, "text_config.intermediate_size", 10944);
  LOAD_ARG_OR(
      max_position_embeddings, "text_config.max_position_embeddings", 131072);
  LOAD_ARG_OR(moe_intermediate_size, "text_config.moe_intermediate_size", 1408);
  LOAD_ARG_OR(n_group, "text_config.n_group", 1);
  LOAD_ARG_OR(num_experts, "text_config.n_routed_experts", 128);
  LOAD_ARG_OR(n_shared_experts, "text_config.n_shared_experts", 1);
  LOAD_ARG_OR(norm_topk_prob, "text_config.norm_topk_prob", true);
  LOAD_ARG_OR(n_heads, "text_config.num_attention_heads", 96);
  LOAD_ARG_OR(num_experts_per_tok, "text_config.num_experts_per_tok", 8);
  LOAD_ARG_OR(n_layers, "text_config.num_hidden_layers", 46);
  LOAD_ARG_OR(n_kv_heads, "text_config.num_key_value_heads", 8);
  // LOAD_ARG_OR(partial_rotary_factor, "text_config.partial_rotary_factor",
  // 0.5);
  LOAD_ARG_OR(rms_norm_eps, "text_config.rms_norm_eps", 1e-05);
  LOAD_ARG_OR(dtype, "text_config.dtype", "bfloat16");
  LOAD_ARG_OR(
      rope_scaling_rope_type, "text_config.rope_parameters.type", "mrope");
  LOAD_ARG(rope_scaling_mrope_section,
           "text_config.rope_parameters.mrope_section");
  LOAD_ARG_OR(rope_theta, "text_config.rope_parameters.rope_theta", 500000.0f);
  LOAD_ARG_OR(routed_scaling_factor, "text_config.routed_scaling_factor", 1.0);
  LOAD_ARG_OR(topk_group, "text_config.topk_group", 1);
  // LOAD_ARG_OR(use_cache, "text_config.use_cache", true);
  LOAD_ARG_OR(use_qk_norm, "text_config.use_qk_norm", false);

  // vision config
  // LOAD_ARG_OR(mm_attention_bias, "vision_config.attention_bias", false);
  // LOAD_ARG_OR(mm_attention_dropout, "vision_config.attention_dropout", 0.0f);
  LOAD_ARG_OR(mm_num_hidden_layers, "vision_config.depth", 24);
  LOAD_ARG_OR(mm_hidden_act, "vision_config.hidden_act", "silu");
  LOAD_ARG_OR(mm_hidden_size, "vision_config.hidden_size", 1536);
  LOAD_ARG_OR(mm_image_size, "vision_config.image_size", 336);
  LOAD_ARG_OR(mm_num_channels, "vision_config.in_channels", 3);
  LOAD_ARG_OR(mm_initializer_range, "vision_config.initializer_range", 0.02);
  LOAD_ARG_OR(mm_intermediate_size, "vision_config.intermediate_size", 10944);
  LOAD_ARG_OR(mm_num_attention_heads, "vision_config.num_heads", 12);
  LOAD_ARG_OR(mm_projection_dim, "vision_config.out_hidden_size", 4096);
  LOAD_ARG_OR(mm_patch_size, "vision_config.patch_size", 14);
  // LOAD_ARG_OR(mm_rms_norm_eps, "vision_config.rms_norm_eps", 1e-05);
  LOAD_ARG_OR(mm_spatial_merge_size, "vision_config.spatial_merge_size", 2);
  LOAD_ARG_OR(mm_temporal_patch_size, "vision_config.temporal_patch_size", 2);
  LOAD_ARG_OR_FUNC(mm_head_dim, "head_dim", [&] {
    return args->mm_hidden_size() / args->mm_num_attention_heads();
  });

  SET_ARG(stop_token_ids,
          std::unordered_set<int32_t>(args->eos_token_id_vec().begin(),
                                      args->eos_token_id_vec().end()));
  SET_ARG(rope_scaling_rope_type, "mrope");
});
}  // namespace xllm::npu::model

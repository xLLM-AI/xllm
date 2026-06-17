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

#include "core/common/global_flags.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_context.h"
#include "core/layers/npu/npu_lm_head_impl.h"
#include "core/layers/npu/npu_qwen3_vision_encoder_layer_impl.h"
#include "core/layers/npu/npu_rms_norm_impl.h"
#include "models/llm/npu/qwen3_moe.h"
#include "models/model_registry.h"
#include "models/vlm/mposition/mposition.h"
#include "models/vlm/utils/multimodal_utils.h"
#include "processors/qwen2_vl_image_processor.h"
#include "processors/qwen3_vl_video_processor.h"
#include "qwen2_5_vl.h"
#include "qwen3_vl.h"

namespace xllm::npu::model {

using torch::indexing::None;
using ISlice = torch::indexing::Slice;

class Qwen3_VLMoeForConditionalGenerationImpl : public torch::nn::Module {
 public:
  Qwen3_VLMoeForConditionalGenerationImpl(const ModelContext& context)
      : model_args_(context.get_model_args()),
        options_(context.get_tensor_options()) {
    encoder_dp_group_ = context.get_parallel_args().encoder_dp_group_;
    use_encoder_dp_ =
        encoder_dp_group_ != nullptr && encoder_dp_group_->world_size() > 1;
    if (use_encoder_dp_) {
      ModelContext visual_context = context.with_parallel_args(ParallelArgs(
          /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr));
      visual_ =
          register_module("visual", Qwen3_VisionTransformer(visual_context));
    } else {
      visual_ = register_module("visual", Qwen3_VisionTransformer(context));
    }

    language_model_ =
        register_module("language_model", Qwen3MoeForCausalLM(context));
  }

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
        const int32_t image_feature_dim =
            model_args_.mm_projection_dim() *
            static_cast<int32_t>(
                model_args_.mm_deepstack_visual_indexes().size() + 1);
        multimodal_embeds["image|embedding"] =
            run_dp_encoder(visual_,
                           image_pixels,
                           image_grid,
                           image_token_nums,
                           image_feature_dim,
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

  std::pair<torch::Tensor, std::vector<torch::Tensor>>
  split_multimodal_embedding(const torch::Tensor& embedding) const {
    const size_t num_deepstacks =
        model_args_.mm_deepstack_visual_indexes().size();
    CHECK(embedding.defined()) << "Multimodal embedding is not defined.";
    CHECK_GT(num_deepstacks, 0)
        << "There should be at least one deepstack when splitting multimodal "
           "embedding.";
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
  Qwen3_VisionTransformer visual_{nullptr};
  Qwen3MoeForCausalLM language_model_{nullptr};
};
TORCH_MODULE(Qwen3_VLMoeForConditionalGeneration);

using Qwen3VLMoeMultimodalProcessor =
    MultimodalProcessor<Qwen3VLPromptProcessor,
                        Qwen2VLImageProcessor,
                        Qwen3VLVideoProcessor>;
REGISTER_MULTIMODAL_PROCESSOR(qwen3_vl_moe, Qwen3VLMoeMultimodalProcessor);
REGISTER_CAUSAL_VLM_MODEL(qwen3_vl_moe, Qwen3_VLMoeForConditionalGeneration);
REGISTER_MPOSITION_GENERATOR(qwen3_vl_moe, xllm::Qwen3VLMPositionGenerator);
// register the model args
REGISTER_MODEL_ARGS(qwen3_vl_moe, [&] {
  // text config
  LOAD_ARG_OR(model_type, "model_type", "qwen3_vl_moe");
  LOAD_ARG_OR(attention_bias, "text_config.attention_bias", false);
  LOAD_ARG_OR(attention_dropout, "attention_dropout", 0.0f);
  LOAD_ARG_OR(bos_token_id, "text_config.bos_token_id", 151643);
  LOAD_ARG_OR(decoder_sparse_step, "text_config.decoder_sparse_step", 1);
  LOAD_ARG_OR(dtype, "text_config.dtype", "bfloat16");
  LOAD_ARG_OR(eos_token_id, "text_config.eos_token_id", 151645);
  LOAD_ARG_OR_FUNC(head_dim, "text_config.head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });
  LOAD_ARG_OR(hidden_act, "text_config.hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "text_config.hidden_size", 2048);
  LOAD_ARG_OR(initializer_range, "text_config.initializer_range", 0.02);
  LOAD_ARG_OR(intermediate_size, "text_config.intermediate_size", 5632);
  LOAD_ARG_OR(
      max_position_embeddings, "text_config.max_position_embeddings", 128000);
  // LOAD_ARG(mlp_only_layers, "text_config.mlp_only_layers");
  LOAD_ARG_OR(moe_intermediate_size, "text_config.moe_intermediate_size", 1408);
  LOAD_ARG_OR(norm_topk_prob, "text_config.norm_topk_prob", true);
  LOAD_ARG_OR(n_heads, "text_config.num_attention_heads", 16);
  LOAD_ARG_OR(num_experts, "text_config.num_experts", 128);
  LOAD_ARG_OR(num_experts_per_tok, "text_config.num_experts_per_tok", 8);
  LOAD_ARG_OR(n_layers, "text_config.num_hidden_layers", 24);
  LOAD_ARG_OR(n_kv_heads, "text_config.num_key_value_heads", 16);
  LOAD_ARG_OR(rms_norm_eps, "text_config.rms_norm_eps", 1e-06);
  LOAD_ARG_OR(rope_scaling_rope_type, "text_config.rope_scaling.type", "mrope");
  LOAD_ARG(rope_scaling_mrope_section,
           "text_config.rope_scaling.mrope_section");
  // LOAD_ARG_OR(rope_scaling_mrope_interleaved,"text_config.rope_scaling.mrope_interleaved",true);
  LOAD_ARG_OR(rope_theta, "text_config.rope_theta", 5000000.0f);
  LOAD_ARG_OR(vocab_size, "text_config.vocab_size", 151936);

  // vision config
  LOAD_ARG(mm_deepstack_visual_indexes,
           "vision_config.deepstack_visual_indexes");
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
  LOAD_ARG_OR_FUNC(mm_head_dim, "head_dim", [&] {
    return args->mm_hidden_size() / args->mm_num_attention_heads();
  });

  LOAD_ARG_OR(image_token_id, "image_token_id", 151655);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_ARG_OR(video_token_id, "video_token_id", 151656);
  LOAD_ARG_OR(vision_end_token_id, "vision_end_token_id", 151653);
  LOAD_ARG_OR(vision_start_token_id, "vision_start_token_id", 151652);
});
}  // namespace xllm::npu::model

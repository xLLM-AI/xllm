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

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "models/dit/encoders/clip_text_model.h"

namespace xllm {

// Wan consumes the penultimate CLIP hidden states, before pooling/projection.
class CLIPVisionEncoderImpl final : public torch::nn::Module {
 public:
  explicit CLIPVisionEncoderImpl(const ModelContext& context) {
    const auto& args = context.get_model_args();
    const auto& options = context.get_tensor_options();
    const int64_t hidden_size = args.mm_hidden_size();
    const int64_t patch_size = args.mm_image_patch_size();
    const int64_t grid_size = args.mm_image_size() / patch_size;
    patch_embedding_ = register_module(
        "patch_embedding",
        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(/*in_channels=*/3, hidden_size, patch_size)
                .stride(patch_size)
                .bias(false)));
    patch_embedding_->to(options.device(), options.dtype().toScalarType());
    class_embedding_ = register_parameter("class_embedding",
                                          torch::empty({hidden_size}, options),
                                          /*requires_grad=*/false);
    position_embedding_ = register_parameter(
        "position_embedding",
        torch::empty({grid_size * grid_size + 1, hidden_size}, options),
        /*requires_grad=*/false);
    pre_layernorm_ = register_module(
        "pre_layernorm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size})
                                 .eps(args.mm_layer_norm_eps())));
    pre_layernorm_->to(options.device(), options.dtype().toScalarType());
    blocks_ = register_module("layers", torch::nn::ModuleList());
    // hidden_states[-2] is the output of layer N-2 (N-1 layers executed).
    const int64_t layer_count = args.mm_num_hidden_layers() - 1;
    layers_.reserve(layer_count);
    for (int64_t index = 0; index < layer_count; ++index) {
      auto layer = CLIPEncoderLayer(context);
      blocks_->push_back(layer);
      layers_.emplace_back(std::move(layer));
    }
  }

  torch::Tensor forward(const torch::Tensor& pixels) {
    torch::Tensor hidden =
        patch_embedding_->forward(pixels).flatten(2).transpose(1, 2);
    torch::Tensor class_token =
        class_embedding_.view({1, 1, -1}).expand({pixels.size(0), 1, -1});
    hidden = pre_layernorm_->forward(torch::cat({class_token, hidden}, 1) +
                                     position_embedding_);
    for (auto& layer : layers_) {
      hidden = layer->forward(hidden, torch::Tensor());
    }
    return hidden;
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    for (const auto& shard : loader->get_state_dicts()) {
      auto state = shard->get_dict_with_prefix("vision_model.");
      weight::load_weight(state,
                          "embeddings.patch_embedding.weight",
                          patch_embedding_->weight,
                          loaded_[0]);
      weight::load_weight(
          state, "embeddings.class_embedding", class_embedding_, loaded_[1]);
      weight::load_weight(state,
                          "embeddings.position_embedding.weight",
                          position_embedding_,
                          loaded_[2]);
      weight::load_weight(
          state, "pre_layrnorm.weight", pre_layernorm_->weight, loaded_[3]);
      weight::load_weight(
          state, "pre_layrnorm.bias", pre_layernorm_->bias, loaded_[4]);
      for (size_t index = 0; index < layers_.size(); ++index) {
        layers_[index]->load_state_dict(state.get_dict_with_prefix(
            "encoder.layers." + std::to_string(index) + "."));
      }
    }
    for (bool loaded : loaded_) {
      CHECK(loaded) << "Missing CLIP vision embedding or normalization weight";
    }
    for (size_t index = 0; index < layers_.size(); ++index) {
      layers_[index]->verify_loaded_weights("vision_model.encoder.layers." +
                                            std::to_string(index) + ".");
    }
    eval();
  }

 private:
  torch::nn::Conv2d patch_embedding_{nullptr};
  torch::Tensor class_embedding_;
  torch::Tensor position_embedding_;
  torch::nn::LayerNorm pre_layernorm_{nullptr};
  torch::nn::ModuleList blocks_{nullptr};
  std::vector<CLIPEncoderLayer> layers_;
  std::array<bool, 5> loaded_{};
};
TORCH_MODULE(CLIPVisionEncoder);

REGISTER_MODEL_ARGS(CLIPVisionModelWithProjection, [&] {
  LOAD_ARG_OR(dtype, "torch_dtype", "float32");
  LOAD_ARG_OR(mm_hidden_size, "hidden_size", 1280);
  LOAD_ARG_OR(mm_intermediate_size, "intermediate_size", 5120);
  LOAD_ARG_OR(mm_num_hidden_layers, "num_hidden_layers", 32);
  LOAD_ARG_OR(mm_num_attention_heads, "num_attention_heads", 16);
  LOAD_ARG_OR(mm_hidden_act, "hidden_act", "gelu");
  LOAD_ARG_OR(mm_layer_norm_eps, "layer_norm_eps", 1e-5f);
  LOAD_ARG_OR(mm_image_size, "image_size", 224);
  LOAD_ARG_OR(mm_image_patch_size, "patch_size", 14);
  args->mm_head_dim() = args->mm_hidden_size() / args->mm_num_attention_heads();
});

REGISTER_MODEL_ARGS(CLIPImageProcessor, [&] {
  LOAD_ARG_OR(mm_image_do_resize, "do_resize", true);
  LOAD_ARG_OR(mm_image_do_center_crop, "do_center_crop", true);
  LOAD_ARG_OR(mm_image_do_rescale, "do_rescale", true);
  LOAD_ARG_OR(mm_image_do_normalize, "do_normalize", true);
  LOAD_ARG_OR(mm_image_resize_shortest_edge, "size.shortest_edge", 224);
  LOAD_ARG_OR(mm_image_crop_height_size, "crop_size.height", 224);
  LOAD_ARG_OR(mm_image_crop_width_size, "crop_size.width", 224);
  LOAD_ARG_OR(mm_image_resample, "resample", 3);
  LOAD_ARG_OR(mm_image_rescale_factor, "rescale_factor", 1.0 / 255.0);
  LOAD_ARG(mm_image_normalize_mean, "image_mean");
  LOAD_ARG(mm_image_normalize_std, "image_std");
});

}  // namespace xllm

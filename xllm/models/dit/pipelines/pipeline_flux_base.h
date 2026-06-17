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

#if defined(USE_NPU)
#include <acl/acl.h>
#endif
#include <torch/torch.h>

#include <algorithm>
#include <memory>
#include <string>

#include "core/framework/dit_model_loader.h"
#include "core/framework/model_context.h"
#include "core/framework/request/dit_request_state.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/state_dict/utils.h"
#include "models/dit/autoencoders/autoencoder_kl.h"
#include "models/dit/encoders/clip_text_model.h"
#include "models/dit/encoders/t5_encoder.h"
#include "models/dit/schedulers/flowmatch_euler_discrete_scheduler.h"
#include "models/dit/transformers/transformer_flux.h"
#include "models/model_registry.h"

namespace xllm {

constexpr int64_t ROPE_SCALE_BASE = 10000;

float calculate_shift(int64_t image_seq_len,
                      int64_t base_seq_len = 256,
                      int64_t max_seq_len = 4096,
                      float base_shift = 0.5f,
                      float max_shift = 1.15f) {
  float m =
      (max_shift - base_shift) / static_cast<float>(max_seq_len - base_seq_len);
  float b = base_shift - m * static_cast<float>(base_seq_len);
  float mu = static_cast<float>(image_seq_len) * m + b;
  return mu;
}

std::pair<torch::Tensor, int64_t> retrieve_timesteps(
    FlowMatchEulerDiscreteScheduler scheduler,
    int64_t num_inference_steps = 0,
    torch::Device device = torch::kCPU,
    std::optional<std::vector<float>> sigmas = std::nullopt,
    std::optional<float> mu = std::nullopt) {
  torch::Tensor scheduler_timesteps;
  int64_t steps;
  if (sigmas.has_value()) {
    steps = sigmas->size();
    scheduler->set_timesteps(
        static_cast<int>(steps), device, *sigmas, mu, std::nullopt);

    scheduler_timesteps = scheduler->timesteps();
  } else {
    steps = num_inference_steps;
    scheduler->set_timesteps(
        static_cast<int>(steps), device, std::nullopt, mu, std::nullopt);
    scheduler_timesteps = scheduler->timesteps();
  }
  if (scheduler_timesteps.device() != device) {
    scheduler_timesteps = scheduler_timesteps.to(device);
  }
  return {scheduler_timesteps, steps};
}

class FluxPosEmbedImpl : public torch::nn::Module {
 public:
  FluxPosEmbedImpl(int64_t theta, std::vector<int64_t> axes_dim) {
    theta_ = theta;
    axes_dim_ = axes_dim;
  }

  std::pair<torch::Tensor, torch::Tensor> forward_cache(
      const torch::Tensor& txt_ids,
      const torch::Tensor& img_ids,
      int64_t height = -1,
      int64_t width = -1) {
    auto seq_len = txt_ids.size(0);

    // recompute the cache if height or width changes
    if (height != cached_image_height_ || width != cached_image_width_ ||
        seq_len != max_seq_len_) {
      torch::Tensor ids = torch::cat({txt_ids, img_ids}, 0);
      cached_image_height_ = height;
      cached_image_width_ = width;
      max_seq_len_ = seq_len;
      auto [cos, sin] = forward(ids);
      freqs_cos_cache_ = std::move(cos);
      freqs_sin_cache_ = std::move(sin);
    }
    return {freqs_cos_cache_, freqs_sin_cache_};
  }

  std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& ids) {
    int64_t n_axes = ids.size(-1);
    std::vector<torch::Tensor> cos_out, sin_out;
    auto pos = ids.to(torch::kFloat32);
    torch::Dtype freqs_dtype = torch::kFloat64;
    for (int64_t i = 0; i < n_axes; ++i) {
      auto pos_slice = pos.select(-1, i);
      auto result = get_1d_rotary_pos_embed(axes_dim_[i],
                                            pos_slice,
                                            theta_,
                                            true,  // repeat_interleave_real
                                            1,
                                            1,
                                            true,  // use_real
                                            freqs_dtype);
      auto cos = result[0];
      auto sin = result[1];
      cos_out.push_back(cos);
      sin_out.push_back(sin);
    }

    auto freqs_cos = torch::cat(cos_out, -1);
    auto freqs_sin = torch::cat(sin_out, -1);
    return {freqs_cos, freqs_sin};
  }

 private:
  int64_t theta_;
  std::vector<int64_t> axes_dim_;
  torch::Tensor freqs_cos_cache_;
  torch::Tensor freqs_sin_cache_;
  int64_t max_seq_len_ = -1;
  int64_t cached_image_height_ = -1;
  int64_t cached_image_width_ = -1;
};
TORCH_MODULE(FluxPosEmbed);

class FluxPipelineBaseImpl : public torch::nn::Module {
 protected:
  torch::Tensor get_t5_prompt_embeds(std::vector<std::string>& prompt,
                                     int64_t num_images_per_prompt = 1,
                                     int64_t max_sequence_length = 512) {
    int64_t batch_size = prompt.size();
    std::vector<std::vector<int32_t>> text_input_ids;
    text_input_ids.reserve(batch_size);
    CHECK(tokenizer_2_->batch_encode(prompt, &text_input_ids));
    for (auto& ids : text_input_ids) {
      ids.resize(max_sequence_length, 0);
    }

    std::vector<int32_t> text_input_ids_flat;
    text_input_ids_flat.reserve(batch_size * max_sequence_length);
    for (const auto& ids : text_input_ids) {
      text_input_ids_flat.insert(
          text_input_ids_flat.end(), ids.begin(), ids.end());
    }
    auto input_ids =
        torch::tensor(text_input_ids_flat, torch::dtype(torch::kLong))
            .view({batch_size, max_sequence_length})
            .to(options_.device());
    torch::Tensor prompt_embeds = t5_->forward(input_ids);
    prompt_embeds = prompt_embeds.to(options_);
    int64_t seq_len = prompt_embeds.size(1);
    prompt_embeds = prompt_embeds.repeat({1, num_images_per_prompt, 1});
    prompt_embeds =
        prompt_embeds.view({batch_size * num_images_per_prompt, seq_len, -1});
    return prompt_embeds;
  }

  torch::Tensor get_clip_prompt_embeds(std::vector<std::string>& prompt,
                                       int64_t num_images_per_prompt = 1) {
    int64_t batch_size = prompt.size();
    std::vector<std::vector<int32_t>> text_input_ids;
    text_input_ids.reserve(batch_size);
    CHECK(tokenizer_->batch_encode(prompt, &text_input_ids));
    for (auto& ids : text_input_ids) {
      ids.resize(tokenizer_max_length_, 49407);
      ids.back() = 49407;
    }

    std::vector<int32_t> text_input_ids_flat;
    text_input_ids_flat.reserve(batch_size * tokenizer_max_length_);
    for (const auto& ids : text_input_ids) {
      text_input_ids_flat.insert(
          text_input_ids_flat.end(), ids.begin(), ids.end());
    }
    auto input_ids =
        torch::tensor(text_input_ids_flat, torch::dtype(torch::kLong))
            .view({batch_size, tokenizer_max_length_})
            .to(options_.device());
    auto encoder_output = clip_text_model_->forward(input_ids);
    torch::Tensor prompt_embeds = encoder_output;
    prompt_embeds = prompt_embeds.to(options_);
    prompt_embeds = prompt_embeds.repeat({1, num_images_per_prompt});
    prompt_embeds =
        prompt_embeds.view({batch_size * num_images_per_prompt, -1});
    return prompt_embeds;
  }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> encode_prompt(
      std::optional<std::vector<std::string>> prompt,
      std::optional<std::vector<std::string>> prompt_2,
      std::optional<torch::Tensor> prompt_embeds,
      std::optional<torch::Tensor> pooled_prompt_embeds,
      int64_t num_images_per_prompt = 1,
      int64_t max_sequence_length = 512) {
    std::vector<std::string> prompt_list;
    if (prompt.has_value()) {
      prompt_list = prompt.value();
    }
    if (prompt_list.empty()) {
      prompt_list = {""};
    }
    if (!prompt_embeds.has_value()) {
      std::vector<std::string> prompt_2_list;
      if (prompt_2.has_value()) {
        prompt_2_list = prompt_2.value();
      }
      if (prompt_2_list.empty()) {
        prompt_2_list = prompt_list;
      }
      pooled_prompt_embeds =
          get_clip_prompt_embeds(prompt_list, num_images_per_prompt);
      prompt_embeds = get_t5_prompt_embeds(
          prompt_2_list, num_images_per_prompt, max_sequence_length);
    }
    torch::Tensor text_ids =
        torch::zeros({prompt_embeds.value().size(1), 3}, options_);

    return std::make_tuple(prompt_embeds.value(),
                           pooled_prompt_embeds.has_value()
                               ? pooled_prompt_embeds.value()
                               : torch::Tensor(),
                           text_ids);
  }

  torch::Tensor prepare_latent_image_ids(int64_t batch_size,
                                         int64_t height,
                                         int64_t width) {
    torch::Tensor latent_image_ids = torch::zeros({height, width, 3}, options_);
    torch::Tensor height_range = torch::arange(height, options_).unsqueeze(1);
    latent_image_ids.select(2, 1) += height_range;
    torch::Tensor width_range = torch::arange(width, options_).unsqueeze(0);
    latent_image_ids.select(2, 2) += width_range;
    latent_image_ids = latent_image_ids.view({height * width, 3});
    return latent_image_ids;
  }

  torch::Tensor pack_latents(const torch::Tensor& latents,
                             int64_t batch_size,
                             int64_t num_channels_latents,
                             int64_t height,
                             int64_t width) {
    torch::Tensor latents_packed = latents.view(
        {batch_size, num_channels_latents, height / 2, 2, width / 2, 2});
    latents_packed = latents_packed.permute({0, 2, 4, 1, 3, 5});
    latents_packed = latents_packed.reshape(
        {batch_size, (height / 2) * (width / 2), num_channels_latents * 4});

    return latents_packed;
  }

  torch::Tensor unpack_latents(const torch::Tensor& latents,
                               int64_t height,
                               int64_t width,
                               int64_t vae_scale_factor) {
    int64_t batch_size = latents.size(0);
    int64_t num_patches = latents.size(1);
    int64_t channels = latents.size(2);
    height = 2 * (height / (vae_scale_factor_ * 2));
    width = 2 * (width / (vae_scale_factor_ * 2));

    torch::Tensor latents_unpacked =
        latents.view({batch_size, height / 2, width / 2, channels / 4, 2, 2});
    latents_unpacked = latents_unpacked.permute({0, 3, 1, 4, 2, 5});
    latents_unpacked = latents_unpacked.reshape(
        {batch_size, channels / (2 * 2), height, width});

    return latents_unpacked;
  }

 protected:
  T5EncoderModel t5_{nullptr};
  CLIPTextModel clip_text_model_{nullptr};
  torch::Device device_ = torch::kCPU;
  torch::ScalarType dtype_;
  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<Tokenizer> tokenizer_2_;
  torch::TensorOptions options_;
  int tokenizer_max_length_;
  int vae_scale_factor_;
};

}  // namespace xllm

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
#include <acl/acl.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/framework/dit_model_loader.h"
#include "core/framework/model_context.h"
#include "core/framework/request/dit_request_state.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/state_dict/utils.h"
#include "models/dit/autoencoders/autoencoder_kl_flux2.h"
#include "models/dit/schedulers/flowmatch_euler_discrete_scheduler.h"
#include "models/dit/transformers/transformer_flux2.h"
#include "models/model_registry.h"

namespace xllm {

inline std::string SYSTEM_MESSAGE =
    "You are an AI that reasons about image descriptions. You give structured "
    "responses focusing on object relationships, object attribution and "
    "actions "
    "without speculation.";

inline std::string SYSTEM_MESSAGE_UPSAMPLING_T2I =
    "You are an expert prompt engineer for FLUX.2 by Black Forest Labs. "
    "Rewrite user prompts to be more descriptive while strictly preserving "
    "their "
    "core subject and intent.\n"
    "Guidelines:\n"
    "1. Structure: Keep structured inputs structured (enhance within fields). "
    "Convert natural language to detailed paragraphs.\n"
    "2. Details: Add concrete visual specifics - form, scale, textures, "
    "materials, lighting (quality, direction, color), shadows, spatial "
    "relationships, and environmental context.\n"
    "3. Text in Images: Put ALL text in quotation marks, matching the "
    "prompt's language. Always provide explicit quoted text for objects that "
    "would "
    "contain text in reality (signs, labels, screens, etc.) - without it, "
    "the model generates gibberish.\n"
    "Output only the revised prompt and nothing else.";

inline std::string SYSTEM_MESSAGE_UPSAMPLING_I2I =
    "You are FLUX.2 by Black Forest Labs, an image-editing expert. You "
    "convert editing requests into one concise instruction (50-80 words, ~30 "
    "for "
    "brief requests).\n"
    "Rules:\n"
    "- Single instruction only, no commentary\n"
    "- Use clear, analytical language (avoid \"whimsical,\" \"cascading,\" "
    "etc.)\n"
    "- Specify what changes AND what stays the same (face, lighting, "
    "composition)\n"
    "- Reference actual image elements\n"
    "- Turn negatives into positives (\"don't change X\" → \"keep X\")\n"
    "- Make abstractions concrete (\"futuristic\" → \"glowing cyan neon, "
    "metallic "
    "panels\")\n"
    "- Keep content PG-13\n"
    "Output only the final instruction in plain text and nothing else.";

inline std::vector<std::vector<std::unordered_map<std::string, std::string>>>
format_input(const std::vector<std::string>& prompts,
             const std::string& system_message = SYSTEM_MESSAGE,
             const std::vector<std::vector<torch::Tensor>>& images =
                 std::vector<std::vector<torch::Tensor>>()) {
  std::vector<std::vector<std::unordered_map<std::string, std::string>>>
      messages_batch;
  messages_batch.reserve(prompts.size());

  for (const auto& prompt : prompts) {
    std::vector<std::unordered_map<std::string, std::string>> messages;

    messages.push_back(
        {{"role", "system"},
         {"content",
          "[{\"type\": \"text\", \"text\": \"" + system_message + "\"}]"}});

    messages.push_back(
        {{"role", "user"},
         {"content", "[{\"type\": \"text\", \"text\": \"" + prompt + "\"}]"}});

    messages_batch.push_back(messages);
  }

  return messages_batch;
}

inline float compute_empirical_mu(int64_t image_seq_len, int64_t num_steps) {
  double a1 = 8.73809524e-05, b1 = 1.89833333;
  double a2 = 0.00016927, b2 = 0.45666666;

  double mu;
  if (image_seq_len > 4300) {
    mu = a2 * image_seq_len + b2;
    return static_cast<float>(mu);
  }

  double m_200 = a2 * image_seq_len + b2;
  double m_10 = a1 * image_seq_len + b1;

  double a = (m_200 - m_10) / 190.0;
  double b = m_200 - 200.0 * a;
  mu = a * num_steps + b;

  return static_cast<float>(mu);
}

inline std::pair<torch::Tensor, int64_t> flux2_retrieve_timesteps(
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

class Flux2PosEmbedImpl final : public torch::nn::Module {
 public:
  Flux2PosEmbedImpl(int64_t theta, std::vector<int64_t> axes_dim) {
    theta_ = theta;
    axes_dim_ = axes_dim;
  }

  std::pair<torch::Tensor, torch::Tensor> forward_cache(
      const torch::Tensor& txt_ids,
      const torch::Tensor& img_ids,
      int64_t height = -1,
      int64_t width = -1) {
    int64_t seq_len = txt_ids.size(0);

    if (height != cached_image_height_ || width != cached_image_width_ ||
        seq_len != max_seq_len_) {
      torch::Tensor ids = torch::cat({txt_ids, img_ids}, 1);
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
    int64_t n_axes = axes_dim_.size();
    std::vector<torch::Tensor> cos_out, sin_out;
    auto pos = ids.to(torch::kFloat32);
    torch::Dtype freqs_dtype = torch::kFloat64;
    for (int64_t i = 0; i < n_axes; ++i) {
      auto pos_slice = pos.select(-1, i).squeeze(0);
      auto result = get_1d_rotary_pos_embed(
          axes_dim_[i], pos_slice, theta_, true, 1, 1, true, freqs_dtype);
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
TORCH_MODULE(Flux2PosEmbed);

class Flux2PipelineBaseImpl : public torch::nn::Module {
 protected:
  torch::Tensor get_mistral3_prompt_embeds(
      std::vector<std::string>& prompt,
      int64_t num_images_per_prompt = 1,
      int64_t max_sequence_length = 512,
      const std::vector<int64_t>& hidden_states_layers = {10, 20, 30},
      const std::string& system_message = SYSTEM_MESSAGE) {
    int64_t batch_size = prompt.size();

    auto messages_batch = format_input(prompt, system_message);

    std::vector<std::string> formatted_prompts;
    formatted_prompts.reserve(batch_size);

    std::vector<std::vector<int32_t>> text_input_ids;
    text_input_ids.reserve(batch_size);
    CHECK(tokenizer_->batch_encode(formatted_prompts, &text_input_ids));
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

    // TODO: implement actual text encoder forward call, e.g.:
    //   auto prompt_embeds = hidden_states_output...
    LOG(FATAL) << "Text encoder is not yet implemented. "
                  "Please provide pre-computed prompt_embeds directly.";

    return {};
  }

  torch::Tensor prepare_text_ids(const torch::Tensor& prompt_embeds) {
    int64_t batch_size = prompt_embeds.size(0);
    int64_t seq_len = prompt_embeds.size(1);

    std::vector<torch::Tensor> out_ids;
    out_ids.reserve(batch_size);
    auto int_opts = torch::TensorOptions().dtype(torch::kInt64);
    for (int64_t i = 0; i < batch_size; ++i) {
      auto t = torch::arange(1, int_opts);
      auto h = torch::arange(1, int_opts);
      auto w = torch::arange(1, int_opts);
      auto l = torch::arange(seq_len, int_opts);
      auto grid = torch::meshgrid({t, h, w, l}, "ij");
      auto coords = torch::stack({grid[0].flatten(),
                                  grid[1].flatten(),
                                  grid[2].flatten(),
                                  grid[3].flatten()},
                                 -1);
      out_ids.push_back(coords);
    }

    auto text_ids = torch::stack(out_ids, 0);
    return text_ids;
  }

  std::tuple<torch::Tensor, torch::Tensor> encode_prompt(
      std::optional<std::vector<std::string>> prompt,
      std::optional<torch::Tensor> prompt_embeds,
      int64_t num_images_per_prompt = 1,
      int64_t max_sequence_length = 512,
      const std::vector<int64_t>& hidden_states_layers = {10, 20, 30},
      const std::string& system_message = SYSTEM_MESSAGE) {
    std::vector<std::string> prompt_list;
    if (prompt.has_value()) {
      prompt_list = prompt.value();
    }
    if (prompt_list.empty()) {
      prompt_list = {""};
    }
    if (!prompt_embeds.has_value()) {
      prompt_embeds = get_mistral3_prompt_embeds(prompt_list,
                                                 num_images_per_prompt,
                                                 max_sequence_length,
                                                 hidden_states_layers);
    }
    torch::Tensor text_ids = prepare_text_ids(prompt_embeds.value());

    return std::make_tuple(prompt_embeds.value(), text_ids);
  }

  torch::Tensor prepare_latent_image_ids(const torch::Tensor& latents) {
    int64_t batch_size = latents.size(0);
    int64_t num_channels = latents.size(1);
    int64_t height = latents.size(2);
    int64_t width = latents.size(3);

    auto t = torch::arange(1, options_);
    auto h = torch::arange(height, options_);
    auto w = torch::arange(width, options_);
    auto l = torch::arange(1, options_);

    auto grid = torch::meshgrid({t, h, w, l}, "ij");
    auto coords = torch::stack({grid[0].flatten(),
                                grid[1].flatten(),
                                grid[2].flatten(),
                                grid[3].flatten()},
                               -1);

    auto latent_image_ids = coords.unsqueeze(0).expand({batch_size, -1, -1});
    return latent_image_ids;
  }

  torch::Tensor pack_latents(const torch::Tensor& latents) {
    int64_t batch_size = latents.size(0);
    int64_t num_channels = latents.size(1);
    int64_t height = latents.size(2);
    int64_t width = latents.size(3);

    torch::Tensor latents_packed =
        latents.reshape({batch_size, num_channels, height * width});
    latents_packed = latents_packed.permute({0, 2, 1});

    return latents_packed;
  }

  torch::Tensor patchify_latents(const torch::Tensor& latents) {
    int64_t batch_size = latents.size(0);
    int64_t num_channels_latents = latents.size(1);
    int64_t height = latents.size(2);
    int64_t width = latents.size(3);

    torch::Tensor latents_patched = latents.view(
        {batch_size, num_channels_latents, height / 2, 2, width / 2, 2});
    latents_patched = latents_patched.permute({0, 1, 3, 5, 2, 4});
    latents_patched = latents_patched.reshape(
        {batch_size, num_channels_latents * 4, height / 2, width / 2});

    return latents_patched;
  }

  torch::Tensor unpatchify_latents(const torch::Tensor& latents) {
    int64_t batch_size = latents.size(0);
    int64_t num_channels_latents = latents.size(1);
    int64_t height = latents.size(2);
    int64_t width = latents.size(3);

    torch::Tensor latents_unpatched = latents.reshape(
        {batch_size, num_channels_latents / (2 * 2), 2, 2, height, width});
    latents_unpatched = latents_unpatched.permute({0, 1, 4, 2, 5, 3});
    latents_unpatched = latents_unpatched.reshape(
        {batch_size, num_channels_latents / (2 * 2), height * 2, width * 2});

    return latents_unpatched;
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

  torch::Tensor unpack_latents_with_ids(const torch::Tensor& latents,
                                        const torch::Tensor& latent_ids) {
    int64_t batch_size = latents.size(0);
    int64_t seq_len = latents.size(1);
    int64_t channels = latents.size(2);

    std::vector<torch::Tensor> x_list;
    for (int64_t i = 0; i < batch_size; ++i) {
      torch::Tensor data = latents[i];
      torch::Tensor pos = latent_ids[i];

      torch::Tensor h_ids = pos.select(1, 1).to(torch::kInt64);
      torch::Tensor w_ids = pos.select(1, 2).to(torch::kInt64);

      int64_t h = h_ids.max().item<int64_t>() + 1;
      int64_t w = w_ids.max().item<int64_t>() + 1;

      torch::Tensor flat_ids = h_ids * w + w_ids;

      torch::Tensor out = torch::zeros({h * w, channels}, data.options());
      out.scatter_(0, flat_ids.unsqueeze(1).expand({-1, channels}), data);

      out = out.view({h, w, channels}).permute({2, 0, 1});
      x_list.push_back(out);
    }

    return torch::stack(x_list, 0);
  }

  torch::Tensor _prepare_image_ids(
      const std::vector<torch::Tensor>& image_latents,
      int64_t scale = 10) {
    if (image_latents.empty()) {
      LOG(FATAL) << "image_latents list cannot be empty!";
    }

    int64_t num_latents = image_latents.size();
    torch::Tensor t_indices = torch::arange(0, num_latents, torch::kInt64);
    torch::Tensor t_coords = scale + scale * t_indices;
    t_coords = t_coords.unsqueeze(1);

    std::vector<torch::Tensor> image_latent_ids;
    for (int64_t i = 0; i < num_latents; ++i) {
      torch::Tensor x = image_latents[i];
      x = x.squeeze(0);
      if (x.dim() != 3) {
        throw std::invalid_argument(
            "Each image latent must be 3D (C, H, W) or 4D (1, C, H, W), got " +
            std::to_string(x.dim()) + "D tensor!");
      }

      int64_t height = x.size(1);
      int64_t width = x.size(2);

      torch::Tensor h_coords = torch::arange(0, height, torch::kInt64);
      torch::Tensor w_coords = torch::arange(0, width, torch::kInt64);
      torch::Tensor l_coords = torch::arange(0, 1, torch::kInt64);

      torch::Tensor t = t_coords[i].expand({1});
      auto t_exp = t.repeat({height * width * 1});
      auto h_exp = h_coords.repeat_interleave(width * 1).repeat({1});
      auto w_exp = w_coords.repeat({height}).repeat_interleave(1);
      auto l_exp = l_coords.repeat({height * width}).repeat({1});

      torch::Tensor coords = torch::stack({t_exp, h_exp, w_exp, l_exp}, 1);
      image_latent_ids.push_back(coords);
    }

    torch::Tensor combined_coords = torch::cat(image_latent_ids, 0);
    combined_coords = combined_coords.unsqueeze(0);
    return combined_coords;
  }

 protected:
  torch::Device device_ = torch::kCPU;
  torch::ScalarType dtype_;
  std::unique_ptr<Tokenizer> tokenizer_;
  torch::TensorOptions options_;
  int32_t tokenizer_max_length_;
  int32_t vae_scale_factor_;
};
}  // namespace xllm

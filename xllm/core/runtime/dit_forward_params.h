/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <torch/torch.h>

#include <nlohmann/json.hpp>
#include <optional>

#include "framework/request/dit_request_state.h"

namespace xllm {

// dit related forward input params
struct DiTForwardInput {
  bool valid() const {
    return batch_size > 0 || prompts.size() > 0 || prompt_embeds.defined() ||
           pooled_prompt_embeds.defined() || images.defined() ||
           !images_list.empty();
  }

  void save_with_prefix(std::string prefix) const {
    torch::save(images, prefix + "images_cpp.pt");
    torch::save(prompt_embeds, prefix + "prompt_embeds_cpp.pt");
    torch::save(negative_prompt_embeds, prefix + "neg_prompt_embeds_cpp.pt");
  }
  void debug_print(std::ostream& os = std::cout) const {
    os << "=== DiTForwardInput Debug Info ===" << std::endl;

    // Print basic data types
    os << "batch_size: " << batch_size << std::endl;

    // Print prompts vectors
    os << "prompts: [";
    for (size_t i = 0; i < prompts.size(); ++i) {
      os << "\"" << prompts[i] << "\"";
      if (i < prompts.size() - 1) os << ", ";
    }
    os << "]" << std::endl;

    os << "prompts_2: [";
    for (size_t i = 0; i < prompts_2.size(); ++i) {
      os << "\"" << prompts_2[i] << "\"";
      if (i < prompts_2.size() - 1) os << ", ";
    }
    os << "]" << std::endl;

    os << "negative_prompts: [";
    for (size_t i = 0; i < negative_prompts.size(); ++i) {
      os << "\"" << negative_prompts[i] << "\"";
      if (i < negative_prompts.size() - 1) os << ", ";
    }
    os << "]" << std::endl;

    os << "negative_prompts_2: [";
    for (size_t i = 0; i < negative_prompts_2.size(); ++i) {
      os << "\"" << negative_prompts_2[i] << "\"";
      if (i < negative_prompts_2.size() - 1) os << ", ";
    }
    os << "]" << std::endl;

    // Print tensor shapes
    os << "\n--- Tensor Shapes ---" << std::endl;

    os << "images: ";
    if (images.defined()) {
      os << images.sizes() << std::endl;
    } else {
      os << "undefined" << std::endl;
    }

    os << "images_list: [";
    for (size_t i = 0; i < images_list.size(); ++i) {
      if (images_list[i].defined()) {
        os << images_list[i].sizes();
      } else {
        os << "undefined";
      }
      if (i < images_list.size() - 1) os << ", ";
    }
    os << "]" << std::endl;

    os << "mask_images: ";
    if (mask_images.defined()) {
      os << mask_images.sizes() << std::endl;
    } else {
      os << "undefined" << std::endl;
    }

    os << "control_image: ";
    if (control_image.defined()) {
      os << control_image.sizes() << std::endl;
    } else {
      os << "undefined" << std::endl;
    }

    os << "masked_image_latents: ";
    if (masked_image_latents.defined()) {
      os << masked_image_latents.sizes() << std::endl;
    } else {
      os << "undefined" << std::endl;
    }

    os << "prompt_embeds: ";
    if (prompt_embeds.defined()) {
      os << prompt_embeds.sizes() << std::endl;
    } else {
      os << "undefined" << std::endl;
    }

    os << "pooled_prompt_embeds: ";
    if (pooled_prompt_embeds.defined()) {
      os << pooled_prompt_embeds.sizes() << std::endl;
    } else {
      os << "undefined" << std::endl;
    }

    os << "negative_prompt_embeds: ";
    if (negative_prompt_embeds.defined()) {
      os << negative_prompt_embeds.sizes() << std::endl;
    } else {
      os << "undefined" << std::endl;
    }

    os << "negative_pooled_prompt_embeds: ";
    if (negative_pooled_prompt_embeds.defined()) {
      os << negative_pooled_prompt_embeds.sizes() << std::endl;
    } else {
      os << "undefined" << std::endl;
    }

    os << "latents: ";
    if (latents.defined()) {
      os << latents.sizes() << std::endl;
    } else {
      os << "undefined" << std::endl;
    }

    os << "last_images: ";
    if (last_images.defined()) {
      os << last_images.sizes() << std::endl;
    } else {
      os << "undefined" << std::endl;
    }

    // Print generation_params
    os << "\n--- Generation Parameters ---" << std::endl;
    os << "width: " << generation_params.width << std::endl;
    os << "height: " << generation_params.height << std::endl;
    os << "num_inference_steps: " << generation_params.num_inference_steps
       << std::endl;
    os << "true_cfg_scale: " << generation_params.true_cfg_scale << std::endl;
    os << "guidance_scale: " << generation_params.guidance_scale << std::endl;
    os << "num_images_per_prompt: " << generation_params.num_images_per_prompt
       << std::endl;
    os << "seed: " << generation_params.seed << std::endl;
    os << "max_sequence_length: " << generation_params.max_sequence_length
       << std::endl;
    os << "strength: " << generation_params.strength << std::endl;

    os << "===============================" << std::endl;
  }

  DiTForwardInput to(const torch::Device& device,
                     torch::ScalarType dtype = torch::kBFloat16) const {
    DiTForwardInput input = *this;

    if (prompt_embeds.defined()) {
      input.prompt_embeds = prompt_embeds.to(device, dtype);
    }

    if (pooled_prompt_embeds.defined()) {
      input.pooled_prompt_embeds = pooled_prompt_embeds.to(device, dtype);
    }

    if (negative_prompt_embeds.defined()) {
      input.negative_prompt_embeds = negative_prompt_embeds.to(device, dtype);
    }

    if (negative_pooled_prompt_embeds.defined()) {
      input.negative_pooled_prompt_embeds =
          negative_pooled_prompt_embeds.to(device, dtype);
    }

    if (latents.defined()) {
      input.latents = latents.to(device, dtype);
    }

    if (masked_image_latents.defined()) {
      input.masked_image_latents = masked_image_latents.to(device, dtype);
    }

    if (images.defined()) {
      input.images = images.to(device, /*dtype=*/torch::kUInt8);
    }

    if (mask_images.defined()) {
      input.mask_images = mask_images.to(device, /*dtype=*/torch::kUInt8);
    }

    for (auto& img : input.images_list) {
      img = img.to(device, /*dtype=*/torch::kUInt8);
    }

    if (control_image.defined()) {
      input.control_image = control_image.to(device, /*dtype=*/torch::kUInt8);
    }

    if (last_images.defined()) {
      input.last_images = last_images.to(device, /*dtype=*/torch::kUInt8);
    }

    if (prompt_audio.defined()) {
      input.prompt_audio = prompt_audio.to(device, torch::kFloat32);
    }
    return input;
  }

  int batch_size = 0;

  // Primary input text description for image generation
  std::vector<std::string> prompts;

  // Secondary prompt for additional details (e.g., color, lighting)
  std::vector<std::string> prompts_2;

  // Negative prompt to exclude low-quality features
  std::vector<std::string> negative_prompts;

  // Secondary negative prompt to exclude additional unwanted features
  std::vector<std::string> negative_prompts_2;

  torch::Tensor images;

  std::vector<torch::Tensor> images_list;

  torch::Tensor mask_images;

  torch::Tensor control_image;

  torch::Tensor masked_image_latents;

  torch::Tensor prompt_embeds;

  torch::Tensor pooled_prompt_embeds;

  torch::Tensor negative_prompt_embeds;

  torch::Tensor negative_pooled_prompt_embeds;

  torch::Tensor latents;

  // Last images for video generation
  torch::Tensor last_images;

  // Optional prompt audio for voice cloning (LongCat-AudioDiT)
  // Shape: (batch, 1, num_samples) at 24kHz
  torch::Tensor prompt_audio;

  // Transcript of the prompt audio — used for duration estimation only.
  std::string audio_prompt_text;

  // generation params
  DiTGenerationParams generation_params;
};

// dit related forward output params
struct DiTForwardOutput {
  void save_with_prefix(std::string prefix) const {
    if (!tensors.empty()) {
      torch::save(tensors[0], prefix + "dit_images_cpp.pt");
    }
  }
  // generated tensor (for image/audio models)
  std::vector<torch::Tensor> tensors;
  // generated text (for text diffusion models like Cola-DLM)
  std::vector<std::string> text_output;
};

}  // namespace xllm

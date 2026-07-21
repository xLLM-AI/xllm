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

#include "dit_batch.h"

#include <c10/core/DeviceType.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "core/framework/config/dit_config.h"

namespace {

bool check_tensors_valid(const std::vector<torch::Tensor>& vec) {
  CHECK(!vec.empty());

  torch::Tensor ref_tensor = vec[0];
  if (!ref_tensor.defined()) return false;

  if (vec.size() == 1) return true;

  const auto ref_shape = ref_tensor.sizes();
  for (size_t i = 1; i < vec.size(); ++i) {
    if (!vec[i].defined()) return false;

    if (vec[i].sizes() != ref_shape) {
      return false;
    }
  }

  return true;
}

}  // namespace

namespace xllm {

DiTForwardInput DiTBatch::prepare_forward_input() {
  CHECK(!request_vec_.empty());
  if (::xllm::DiTConfig::get_instance().dit_debug_print()) {
    LOG(INFO) << "DiT batch_size=" << request_vec_.size();
  }
  if (request_vec_[0]->state().request_kind() == DiTRequestKind::kText) {
    CHECK_EQ(request_vec_.size(), 1U)
        << "Cola-DLM text generation supports batch_size=1 only.";
  }

  DiTForwardInput input;
  input.batch_size = request_vec_.size();
  input.generation_params = request_vec_[0]->state().generation_params();

  std::vector<torch::Tensor> prompt_embeds;
  std::vector<torch::Tensor> pooled_prompt_embeds;

  std::vector<torch::Tensor> negative_prompt_embeds;
  std::vector<torch::Tensor> negative_pooled_prompt_embeds;

  std::vector<torch::Tensor> images;
  std::vector<torch::Tensor> mask_images;
  std::vector<torch::Tensor> control_images;
  std::vector<torch::Tensor> latents;
  std::vector<torch::Tensor> masked_image_latents;
  std::vector<torch::Tensor> last_images;
  std::vector<std::vector<torch::Tensor>> per_request_images;
  const auto batch_size = request_vec_.size();
  prompt_embeds.reserve(batch_size);
  pooled_prompt_embeds.reserve(batch_size);
  negative_prompt_embeds.reserve(batch_size);
  negative_pooled_prompt_embeds.reserve(batch_size);
  images.reserve(batch_size);
  mask_images.reserve(batch_size);
  control_images.reserve(batch_size);
  latents.reserve(batch_size);
  masked_image_latents.reserve(batch_size);
  last_images.reserve(batch_size);
  per_request_images.reserve(batch_size);

  std::vector<torch::Tensor> images_list;
  size_t images_size = 0;
  bool images_size_valid = true;
  bool images_size_initialized = false;
  for (const auto& request : request_vec_) {
    const auto& generation_params = request->state().generation_params();
    CHECK(input.generation_params == generation_params)
        << "DiT generation params must be equal in the same batch";

    const auto& input_params = request->state().input_params();
    if (!input_params.prompt.empty())
      input.prompts.emplace_back(input_params.prompt);

    if (!input_params.prompt_2.empty())
      input.prompts_2.emplace_back(input_params.prompt_2);

    if (!input_params.negative_prompt.empty())
      input.negative_prompts.emplace_back(input_params.negative_prompt);

    if (!input_params.negative_prompt_2.empty())
      input.negative_prompts_2.emplace_back(input_params.negative_prompt_2);

    prompt_embeds.emplace_back(input_params.prompt_embed);
    pooled_prompt_embeds.emplace_back(input_params.pooled_prompt_embed);

    negative_prompt_embeds.emplace_back(input_params.negative_prompt_embed);
    negative_pooled_prompt_embeds.emplace_back(
        input_params.negative_pooled_prompt_embed);

    latents.emplace_back(input_params.latent);
    masked_image_latents.emplace_back(input_params.masked_image_latent);

    images.emplace_back(input_params.image);
    mask_images.emplace_back(input_params.mask_image);
    control_images.emplace_back(input_params.control_image);
    last_images.emplace_back(input_params.last_image);

    std::vector<torch::Tensor> request_images = input_params.images;
    if (request_images.empty() && input_params.image.defined()) {
      request_images.emplace_back(input_params.image);
    }
    if (!images_size_initialized) {
      images_size = request_images.size();
      images_size_valid = images_size > 0;
      images_size_initialized = true;
    } else if (request_images.size() != images_size) {
      images_size_valid = false;
    }
    per_request_images.emplace_back(std::move(request_images));

    // Voice cloning: prompt_audio is per-request (batch_size==1 in practice).
    // Forward the first defined tensor; multi-batch voice cloning is not
    // supported (different prompt lengths can't be stacked).
    if (input_params.prompt_audio.defined() && !input.prompt_audio.defined()) {
      input.prompt_audio = input_params.prompt_audio;
    }
    if (!input_params.audio_prompt_text.empty() &&
        input.audio_prompt_text.empty()) {
      input.audio_prompt_text = input_params.audio_prompt_text;
    }
  }

  if (input.prompts.size() != request_vec_.size()) {
    input.prompts.clear();
  }

  if (input.prompts_2.size() != request_vec_.size()) {
    input.prompts_2.clear();
  }

  const bool has_full_negative_prompts =
      input.negative_prompts.size() == request_vec_.size();
  if (!has_full_negative_prompts) {
    input.negative_prompts.clear();
  }

  if (input.negative_prompts_2.size() != request_vec_.size()) {
    input.negative_prompts_2.clear();
  }

  if (check_tensors_valid(images)) {
    input.images = torch::stack(images);
  }

  if (images_size_valid) {
    images_list.reserve(images_size);
    std::vector<torch::Tensor> vec;
    vec.reserve(request_vec_.size());

    bool all_valid = true;
    for (size_t idx = 0; idx < images_size; ++idx) {
      vec.clear();
      for (const auto& request_images : per_request_images) {
        vec.emplace_back(request_images[idx]);
      }
      if (!check_tensors_valid(vec)) {
        all_valid = false;
        break;
      }
      images_list.emplace_back(torch::stack(vec));
    }
    if (all_valid) {
      input.images_list = std::move(images_list);
    }
  }

  if (check_tensors_valid(mask_images)) {
    input.mask_images = torch::stack(mask_images);
  }

  if (check_tensors_valid(control_images)) {
    input.control_image = torch::stack(control_images);
  }

  if (check_tensors_valid(prompt_embeds)) {
    input.prompt_embeds = torch::stack(prompt_embeds);
  }

  if (check_tensors_valid(pooled_prompt_embeds)) {
    input.pooled_prompt_embeds = torch::stack(pooled_prompt_embeds);
  }

  if (check_tensors_valid(negative_prompt_embeds)) {
    input.negative_prompt_embeds = torch::stack(negative_prompt_embeds);
  }

  if (check_tensors_valid(negative_pooled_prompt_embeds)) {
    input.negative_pooled_prompt_embeds =
        torch::stack(negative_pooled_prompt_embeds);
  }

  if (check_tensors_valid(latents)) {
    input.latents = torch::stack(latents);
  }

  if (check_tensors_valid(masked_image_latents)) {
    input.masked_image_latents = torch::stack(masked_image_latents);
  }

  if (check_tensors_valid(last_images)) {
    input.last_images = torch::stack(last_images);
  }

  return input;
}

void DiTBatch::process_forward_output(const DiTForwardOutput& output) {
  // Text diffusion models produce text output directly.
  if (!output.text_output.empty()) {
    CHECK(request_vec_.size() == output.text_output.size());
    for (int32_t idx = 0; idx < static_cast<int32_t>(request_vec_.size());
         ++idx) {
      auto& request = request_vec_[idx];
      request->handle_forward_text_output(output.text_output[idx]);
    }
    return;
  }
  CHECK(request_vec_.size() == output.tensors.size());
  for (int32_t idx = 0; idx < static_cast<int32_t>(request_vec_.size());
       ++idx) {
    auto& request = request_vec_[idx];
    request->handle_forward_output(output.tensors[idx]);
  }
}

}  // namespace xllm

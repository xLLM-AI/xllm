/* Copyright 2025-2026 The xLLM Authors.

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

#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <vector>

#include "dit_request_output.h"

namespace xllm {

using DiTOutputFunc = std::function<bool(const DiTRequestOutput& output)>;
using DiTOutputsFunc = std::function<std::vector<bool>(
    const std::vector<DiTRequestOutput>& outputs)>;

class Call;

enum class DiTRequestKind : int8_t {
  kImage = 0,
  kAudio = 1,
  kVideo = 2,
  kText = 3,
};

struct DiTGenerationParams {
  bool operator==(const DiTGenerationParams& other) const {
    return width == other.width && height == other.height &&
           num_inference_steps == other.num_inference_steps &&
           true_cfg_scale == other.true_cfg_scale &&
           guidance_scale == other.guidance_scale &&
           num_images_per_prompt == other.num_images_per_prompt &&
           seed == other.seed && seed_is_set == other.seed_is_set &&
           max_sequence_length == other.max_sequence_length &&
           strength == other.strength &&
           enable_cfg_renorm == other.enable_cfg_renorm &&
           cfg_renorm_min == other.cfg_renorm_min &&
           audio_duration_frames == other.audio_duration_frames &&
           audio_steps == other.audio_steps &&
           audio_guidance_method == other.audio_guidance_method &&
           audio_sampling_rate == other.audio_sampling_rate &&
           num_videos_per_prompt == other.num_videos_per_prompt &&
           num_frames == other.num_frames &&
           force_video_output == other.force_video_output &&
           video_fps == other.video_fps &&
           guidance_scale_2 == other.guidance_scale_2 &&
           seconds == other.seconds && boundary_ratio == other.boundary_ratio &&
           flow_shift == other.flow_shift &&
           max_new_tokens == other.max_new_tokens &&
           diffusion_steps == other.diffusion_steps &&
           temperature == other.temperature && top_k == other.top_k &&
           top_p == other.top_p &&
           repetition_penalty == other.repetition_penalty;
  }

  bool operator!=(const DiTGenerationParams& other) const {
    return !(*this == other);
  }

  int32_t width = 512;

  int32_t height = 512;

  int32_t num_inference_steps = 28;

  float true_cfg_scale = 1.0;

  float guidance_scale = 3.5;

  uint32_t num_images_per_prompt = 1;

  uint32_t num_videos_per_prompt = 1;

  // Default seed for image/audio DiT models when the client omits seed (legacy
  // behavior). Cola-DLM uses seed_is_set to distinguish explicit seed=0 from
  // unset (stochastic) requests.
  int64_t seed = 0;

  // True when the client explicitly set seed in the request (proto has_seed).
  bool seed_is_set = false;

  int32_t max_sequence_length = 512;

  float strength = 1.0;

  bool enable_cfg_renorm = true;

  float cfg_renorm_min = 0.0f;

  // Audio generation params (for LongCat-AudioDiT)
  // Target duration in latent frames (prompt + gen). 0 means use max_duration.
  int32_t audio_duration_frames = 0;

  // Number of ODE Euler steps for audio generation
  int32_t audio_steps = 16;

  // Guidance method: "cfg" or "apg"
  std::string audio_guidance_method = "cfg";

  // Audio sample rate in Hz, read from model config.json (sampling_rate).
  int32_t audio_sampling_rate = 24000;

  int32_t num_frames = 81;

  bool force_video_output = false;

  double video_fps = 8.0;

  float guidance_scale_2 = 1.0;

  int32_t seconds = 5;

  float boundary_ratio = 0.9f;

  float flow_shift = 1.0f;

  // Text diffusion generation params (for Cola-DLM)
  int32_t max_new_tokens = 256;
  int32_t diffusion_steps = 16;
  float temperature = 0.0f;
  int32_t top_k = 0;
  float top_p = 1.0f;
  float repetition_penalty = 1.1f;
};

struct DiTInputParams {
  // Primary input text description for image generation
  std::string prompt;

  // Secondary prompt for additional details (e.g., color, lighting)
  std::string prompt_2;

  // Negative prompt to exclude low-quality features
  std::string negative_prompt;

  // Secondary negative prompt to exclude additional unwanted features
  std::string negative_prompt_2;

  torch::Tensor prompt_embed;

  torch::Tensor pooled_prompt_embed;

  torch::Tensor negative_prompt_embed;

  torch::Tensor negative_pooled_prompt_embed;

  torch::Tensor latent;

  torch::Tensor image;

  std::vector<torch::Tensor> images;

  torch::Tensor control_image;

  torch::Tensor mask_image;

  torch::Tensor masked_image_latent;

  // Video-specific input fields
  torch::Tensor last_image;

  // Prompt audio for voice cloning (LongCat-AudioDiT).
  // Float32 PCM, shape (1, num_samples), mono 24 kHz.
  torch::Tensor prompt_audio;

  // Transcript of the prompt audio (for duration estimation).
  std::string audio_prompt_text;
};

struct DiTRequestState {
 public:
  DiTRequestState(DiTInputParams& input_params,
                  DiTGenerationParams& generation_params,
                  const DiTOutputFunc& output_func,
                  const DiTOutputsFunc& outputs_func,
                  DiTRequestKind request_kind,
                  std::optional<Call*> call = std::nullopt)
      : input_params_(std::move(input_params)),
        generation_params_(std::move(generation_params)),
        output_func_(std::move(output_func)),
        outputs_func_(std::move(outputs_func)),
        request_kind_(request_kind),
        call_(call) {}
  DiTRequestState() {}
  DiTInputParams& input_params() { return input_params_; }
  const DiTInputParams& input_params() const { return input_params_; }
  DiTGenerationParams& generation_params() { return generation_params_; }
  const DiTGenerationParams& generation_params() const {
    return generation_params_;
  }
  DiTOutputFunc& output_func() { return output_func_; }
  DiTOutputsFunc& outputs_func() { return outputs_func_; }
  DiTRequestKind request_kind() const { return request_kind_; }
  std::optional<Call*>& call() { return call_; }

 private:
  DiTInputParams input_params_;
  DiTGenerationParams generation_params_;
  DiTOutputFunc output_func_;
  DiTOutputsFunc outputs_func_;
  DiTRequestKind request_kind_ = DiTRequestKind::kImage;
  std::optional<Call*> call_;
};

}  // namespace xllm

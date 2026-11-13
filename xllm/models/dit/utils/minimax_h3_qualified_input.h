/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "models/dit/utils/minimax_h3_packing.h"

namespace xllm {

struct H3QualifiedRequest {
  H3PackedLayout layout;
  torch::Tensor video_rows;
  torch::Tensor audio_rows;
  torch::Tensor video_sigmas;
  torch::Tensor audio_sigmas;
  int64_t condition_video_rows = 0;
  int64_t condition_audio_rows = 0;
  int64_t audio_output_samples = 0;
  H3TargetLatents target;
};

inline constexpr int64_t kH3QualifiedLatentChannels = 24;
inline constexpr int64_t kH3QualifiedAudioLatentChannels = 32;
inline constexpr int64_t kH3QualifiedAudioSamplesPerLatentFrame = 800;

struct H3QualifiedPreparedMedia {
  torch::Tensor reference_latent;
  torch::Tensor reference_pixels;
};

struct MiniMaxH3RowTimestepPlan {
  torch::Tensor row_timesteps;
  torch::Tensor unique_timesteps;
  torch::Tensor inverse_indices;
};

inline torch::Tensor h3_qualified_sigmas(float shift) {
  const torch::Tensor base = torch::linspace(
      1.0F, 0.0F, 50, torch::TensorOptions().dtype(torch::kFloat32));
  return (shift * base / (1.0F + (shift - 1.0F) * base)).contiguous();
}

inline std::array<torch::Tensor, 3> h3_qualified_generation_noise(
    int64_t seed,
    const H3TargetLatents& target,
    int64_t reference_latent_t,
    int64_t reference_latent_h,
    int64_t reference_latent_w) {
  if (seed < 0) {
    throw std::invalid_argument(
        "MiniMax-H3 qualified seed must be nonnegative");
  }
  if (target.latent_t <= 0 || target.latent_h <= 0 || target.latent_w <= 0 ||
      target.audio_t <= 0 || target.audio_channels <= 0 ||
      reference_latent_t <= 0 || reference_latent_h <= 0 ||
      reference_latent_w <= 0) {
    throw std::invalid_argument(
        "MiniMax-H3 qualified noise requires a positive derived shape");
  }
  torch::Generator generator = torch::make_generator<torch::CPUGeneratorImpl>();
  generator.set_current_seed(static_cast<uint64_t>(seed));
  const auto options =
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32);
  // Preserve the RNG draw order for a given request shape.
  std::array<torch::Tensor, 3> noise;
  noise[0] = torch::randn({1,
                           kH3QualifiedLatentChannels,
                           reference_latent_t,
                           reference_latent_h,
                           reference_latent_w},
                          generator,
                          options);
  noise[1] = torch::randn({1,
                           kH3QualifiedLatentChannels,
                           target.latent_t,
                           target.latent_h,
                           target.latent_w},
                          generator,
                          options);
  noise[2] = torch::randn(
      {target.audio_t * target.audio_channels, kH3QualifiedAudioLatentChannels},
      generator,
      options);
  return noise;
}

inline H3QualifiedRequest build_qualified_request(
    const torch::Tensor& condition_hidden,
    const torch::Tensor& condition_tags,
    const H3QualifiedPreparedMedia& media,
    int64_t seed,
    const H3TargetLatents& target,
    int64_t reference_latent_t,
    int64_t reference_latent_h,
    int64_t reference_latent_w) {
  const std::vector<int64_t> reference_shape{1,
                                             kH3QualifiedLatentChannels,
                                             reference_latent_t,
                                             reference_latent_h,
                                             reference_latent_w};
  if (!media.reference_latent.defined() ||
      media.reference_latent.sizes().vec() != reference_shape ||
      media.reference_latent.scalar_type() != torch::kFloat32) {
    throw std::invalid_argument(
        "MiniMax-H3 reference latent must be FP32 with the derived shape");
  }
  const std::vector<H3ReferenceBlock> references{
      {.kind = H3ReferenceBlockKind::IMAGE,
       .latent_t = reference_latent_t,
       .latent_h = reference_latent_h,
       .latent_w = reference_latent_w}};
  H3QualifiedRequest request;
  // Use the same storage alignment for both qualified attention paths.
  request.layout = minimax_h3_build_ref2va_packed_layout(
      condition_hidden,
      condition_tags,
      target,
      references,
      /*sequence_length=*/std::nullopt,
      /*alignment=*/minimax_h3_packing_detail::kH3RainfusionStorageAlignment);
  auto noise = h3_qualified_generation_noise(
      seed, target, reference_latent_t, reference_latent_h, reference_latent_w);
  const torch::Device device = media.reference_latent.device();
  const torch::Tensor timestep =
      torch::tensor(
          0.999F, torch::TensorOptions().device(device).dtype(torch::kFloat32))
          .reshape({1, 1, 1, 1, 1});
  const torch::Tensor anchor = timestep * media.reference_latent +
                               (1.0 - timestep) * noise[0].to(device);
  request.video_rows =
      torch::cat({minimax_h3_patchify_video_latent(anchor),
                  minimax_h3_patchify_video_latent(noise[1].to(device))},
                 0)
          .unsqueeze(0)
          .contiguous();
  request.audio_rows = noise[2].to(device).unsqueeze(0).contiguous();
  const H3Slice reference_rows =
      request.layout.reference_blocks.at(0).row_slice;
  request.condition_video_rows = reference_rows.stop - reference_rows.start;
  request.target = target;
  request.audio_output_samples =
      target.audio_t * kH3QualifiedAudioSamplesPerLatentFrame;
  request.video_sigmas = h3_qualified_sigmas(12.0F);
  request.audio_sigmas = h3_qualified_sigmas(3.0F);
  return request;
}

inline MiniMaxH3RowTimestepPlan minimax_h3_build_row_timestep_plan(
    const H3PackedLayout& layout,
    float video_timestep,
    float audio_timestep) {
  if (!std::isfinite(video_timestep) || !std::isfinite(audio_timestep) ||
      video_timestep < 0.0F || video_timestep > 1.0F || audio_timestep < 0.0F ||
      audio_timestep > 1.0F) {
    throw std::invalid_argument(
        "MiniMax-H3 row timesteps must be finite and in [0,1]");
  }
  if (layout.aligned_length <= 0 || layout.used_length <= 0 ||
      layout.aligned_length < layout.used_length ||
      layout.img_pos.scalar_type() != torch::kInt64 ||
      layout.audio_pos.scalar_type() != torch::kInt64 ||
      layout.update_mask.scalar_type() != torch::kBool ||
      layout.audio_update_mask.scalar_type() != torch::kBool ||
      layout.img_pos.numel() != layout.update_mask.numel() ||
      layout.audio_pos.numel() != layout.audio_update_mask.numel()) {
    throw std::invalid_argument(
        "MiniMax-H3 row timestep layout metadata is invalid");
  }

  std::vector<float> rows(static_cast<size_t>(layout.aligned_length),
                          video_timestep);
  const torch::Tensor image_positions =
      layout.img_pos.to(torch::kCPU).contiguous();
  const torch::Tensor audio_positions =
      layout.audio_pos.to(torch::kCPU).contiguous();
  const torch::Tensor image_update =
      layout.update_mask.to(torch::kCPU).contiguous();
  const torch::Tensor audio_update =
      layout.audio_update_mask.to(torch::kCPU).contiguous();
  const int64_t* image_position_values =
      image_positions.const_data_ptr<int64_t>();
  const int64_t* audio_position_values =
      audio_positions.const_data_ptr<int64_t>();
  const bool* image_update_values = image_update.const_data_ptr<bool>();
  const bool* audio_update_values = audio_update.const_data_ptr<bool>();
  for (int64_t index = 0; index < image_positions.numel(); ++index) {
    const int64_t position = image_position_values[index];
    if (position < 0 || position >= layout.used_length) {
      throw std::invalid_argument(
          "MiniMax-H3 image position is outside used rows");
    }
    rows[static_cast<size_t>(position)] =
        image_update_values[index] ? video_timestep
                                   : std::max(video_timestep, 0.999F);
  }
  for (int64_t index = 0; index < audio_positions.numel(); ++index) {
    const int64_t position = audio_position_values[index];
    if (position < 0 || position >= layout.used_length) {
      throw std::invalid_argument(
          "MiniMax-H3 audio position is outside used rows");
    }
    rows[static_cast<size_t>(position)] = audio_update_values[index]
                                              ? audio_timestep
                                              : std::max(audio_timestep, 1.0F);
  }

  std::vector<float> unique = rows;
  std::sort(unique.begin(), unique.end());
  unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
  std::vector<int64_t> inverse(rows.size());
  for (size_t index = 0; index < rows.size(); ++index) {
    inverse[index] = static_cast<int64_t>(
        std::lower_bound(unique.begin(), unique.end(), rows[index]) -
        unique.begin());
  }
  return {
      .row_timesteps =
          torch::from_blob(
              rows.data(), {static_cast<int64_t>(rows.size())}, torch::kFloat32)
              .clone(),
      .unique_timesteps =
          torch::from_blob(unique.data(),
                           {static_cast<int64_t>(unique.size())},
                           torch::kFloat32)
              .clone(),
      .inverse_indices =
          torch::from_blob(inverse.data(),
                           {static_cast<int64_t>(inverse.size())},
                           torch::kInt64)
              .clone(),
  };
}

}  // namespace xllm

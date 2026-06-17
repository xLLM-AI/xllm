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

#include <torch/nn/functional.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

#include "framework/model_context.h"
#include "models/dit/autoencoder_kl.h"

namespace xllm {

class VideoProcessorImpl : public VAEImageProcessorImpl {
 public:
  using VAEImageProcessorImpl::VAEImageProcessorImpl;

  torch::Tensor preprocess_video(const torch::Tensor& video,
                                 std::optional<int64_t> height = std::nullopt,
                                 std::optional<int64_t> width = std::nullopt) {
    torch::Tensor processed = video.clone();

    if (processed.dtype() != torch::kFloat32) {
      processed = processed.to(torch::kFloat32);
    }

    if (processed.max().item<float>() > 1.1f) {
      processed = processed / 255.0f;
    }

    int64_t input_dim = processed.dim();
    int64_t batch_size, num_frames, h, w;

    if (input_dim == 5) {
      batch_size = processed.size(0);
      num_frames = processed.size(2);
      h = processed.size(3);
      w = processed.size(4);
    } else if (input_dim == 4) {
      batch_size = 1;
      num_frames = processed.size(1);
      h = processed.size(2);
      w = processed.size(3);
      processed = processed.unsqueeze(0);
    } else {
      LOG(FATAL) << "Unsupported video tensor dimension: " << input_dim;
    }

    int64_t target_h = height.value_or(h);
    int64_t target_w = width.value_or(w);

    std::vector<torch::Tensor> processed_videos;
    processed_videos.reserve(batch_size);

    for (int64_t b = 0; b < batch_size; ++b) {
      std::vector<torch::Tensor> frames;
      frames.reserve(num_frames);

      for (int64_t f = 0; f < num_frames; ++f) {
        torch::Tensor frame = processed.index({b, torch::indexing::Slice(), f});
        frame = preprocess(frame, target_h, target_w);
        frames.push_back(frame);
      }

      torch::Tensor video_frames = torch::stack(frames, 0);
      processed_videos.push_back(video_frames);
    }

    torch::Tensor result = torch::stack(processed_videos, 0);
    result = result.permute({0, 2, 1, 3, 4});

    return result;
  }

  torch::Tensor postprocess_video(const torch::Tensor& video,
                                  const std::string& output_type = "pt") {
    int64_t batch_size = video.size(0);
    std::vector<torch::Tensor> outputs;
    outputs.reserve(batch_size);

    for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      torch::Tensor batch_vid = video[batch_idx].permute({1, 0, 2, 3});
      torch::Tensor batch_output = postprocess(batch_vid);
      outputs.push_back(batch_output);
    }

    return torch::stack(outputs, 0);
  }

  static std::pair<int64_t, int64_t> classify_height_width_bin(
      int64_t height,
      int64_t width,
      const std::map<float, std::pair<int64_t, int64_t>>& ratios) {
    float ar = static_cast<float>(height) / static_cast<float>(width);

    if (ratios.empty()) {
      LOG(FATAL) << "ratios map mush not be empty.";
    }
    auto it = ratios.begin();
    float closest_ratio = it->first;
    float min_diff = std::abs(it->first - ar);

    for (const auto& [ratio, hw] : ratios) {
      float diff = std::abs(ratio - ar);
      if (diff < min_diff) {
        min_diff = diff;
        closest_ratio = ratio;
      }
    }

    return ratios.at(closest_ratio);
  }

  torch::Tensor resize_and_crop_tensor(const torch::Tensor& samples,
                                       int64_t new_width,
                                       int64_t new_height) {
    int64_t orig_height = samples.size(3);
    int64_t orig_width = samples.size(4);

    if (orig_height == new_height && orig_width == new_width) {
      return samples.clone();
    }

    float ratio = std::max(
        static_cast<float>(new_height) / static_cast<float>(orig_height),
        static_cast<float>(new_width) / static_cast<float>(orig_width));

    int64_t resized_width = static_cast<int64_t>(orig_width * ratio);
    int64_t resized_height = static_cast<int64_t>(orig_height * ratio);

    int64_t n = samples.size(0);
    int64_t c = samples.size(1);
    int64_t t = samples.size(2);
    int64_t h = samples.size(3);
    int64_t w = samples.size(4);

    torch::Tensor reshaped =
        samples.permute({0, 2, 1, 3, 4}).reshape({n * t, c, h, w});

    reshaped = torch::nn::functional::interpolate(
        reshaped,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{resized_height, resized_width})
            .mode(torch::kBicubic)
            .align_corners(false));

    int64_t start_x = (resized_width - new_width) / 2;
    int64_t end_x = start_x + new_width;
    int64_t start_y = (resized_height - new_height) / 2;
    int64_t end_y = start_y + new_height;

    reshaped = reshaped.index({torch::indexing::Slice(),
                               torch::indexing::Slice(),
                               torch::indexing::Slice(start_y, end_y),
                               torch::indexing::Slice(start_x, end_x)});

    reshaped = reshaped.reshape({n, t, c, new_height, new_width})
                   .permute({0, 2, 1, 3, 4});

    return reshaped;
  }
};

TORCH_MODULE(VideoProcessor);

}  // namespace xllm

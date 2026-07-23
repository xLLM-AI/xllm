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
#include <cmath>
#include <memory>
#include <vector>

#include "models/dit/autoencoders/autoencoder_kl.h"
#include "torch/torch.h"

namespace xllm {

class Flux2ImageProcessorImpl final : public VAEImageProcessorImpl {
 public:
  explicit Flux2ImageProcessorImpl(const ModelContext& context,
                                   int64_t vae_scale_factor = 16)
      : VAEImageProcessorImpl(context, true, true, false, true, false, 32),
        vae_scale_factor_(vae_scale_factor),
        vae_latent_channels_(32) {}

  torch::Tensor check_image_input(const torch::Tensor& image,
                                  int64_t max_aspect_ratio = 8,
                                  int64_t min_side_length = 64,
                                  int64_t max_area = 1024 * 1024) {
    if (image.dim() != 3 && image.dim() != 4) {
      LOG(FATAL) << "Image must be 3D (C, H, W) or 4D (B, C, H, W), got dim: "
                 << image.dim();
    }

    int64_t height = image.size(-2);
    int64_t width = image.size(-1);

    if (width < min_side_length || height < min_side_length) {
      LOG(FATAL) << "Image too small: " << width << "x" << height
                 << ". Both dimensions must be at least " << min_side_length
                 << "px";
    }

    float aspect_ratio = std::max(static_cast<float>(width) / height,
                                  static_cast<float>(height) / width);
    if (aspect_ratio > max_aspect_ratio) {
      LOG(FATAL) << "Aspect ratio too extreme: " << width << "x" << height
                 << " (ratio: " << aspect_ratio
                 << ":1). Maximum allowed ratio is " << max_aspect_ratio
                 << ":1";
    }

    return image;
  }

  torch::Tensor resize_to_target_area(const torch::Tensor& image,
                                      int64_t target_area = 1024 * 1024) {
    int64_t image_width = image.size(-1);
    int64_t image_height = image.size(-2);

    float scale = std::sqrt(static_cast<float>(target_area) /
                            static_cast<float>(image_width * image_height));
    int64_t width = static_cast<int64_t>(image_width * scale);
    int64_t height = static_cast<int64_t>(image_height * scale);
    return torch::nn::functional::interpolate(
               image.unsqueeze(0),  // [1, 3, H, W]
               torch::nn::functional::InterpolateFuncOptions()
                   .mode(torch::kBilinear)
                   .align_corners(false)
                   .size(std::vector<int64_t>{height, width}))
        .squeeze(0);
  }

  torch::Tensor resize_if_exceeds_area(const torch::Tensor& image,
                                       int64_t target_area = 1024 * 1024) {
    int64_t image_width = image.size(-1);
    int64_t image_height = image.size(-2);
    int64_t pixel_count = image_width * image_height;

    if (pixel_count <= target_area) {
      return image;
    }

    return resize_to_target_area(image, target_area);
  }

  torch::Tensor resize_and_crop(const torch::Tensor& image,
                                int64_t width,
                                int64_t height) {
    int64_t image_width = image.size(-1);
    int64_t image_height = image.size(-2);

    int64_t left = (image_width - width) / 2;
    int64_t top = (image_height - height) / 2;
    int64_t right = left + width;
    int64_t bottom = top + height;

    return image.slice(-2, top, bottom).slice(-1, left, right);
  }

  torch::Tensor concatenate_images(const std::vector<torch::Tensor>& images) {
    if (images.empty()) {
      LOG(FATAL) << "Cannot concatenate empty image list";
    }

    if (images.size() == 1) {
      return images[0].clone();
    }

    int64_t total_width = 0;
    int64_t max_height = 0;

    for (const auto& img : images) {
      total_width += img.size(-1);
      max_height = std::max(max_height, img.size(-2));
    }

    torch::TensorOptions options = images[0].options();
    auto background_color = torch::full({1, 3, max_height, 1}, 1.0f, options);

    auto new_img = torch::full({1, 3, max_height, total_width}, 1.0f, options);

    int64_t x_offset = 0;
    for (const auto& img : images) {
      int64_t img_height = img.size(-2);
      int64_t img_width = img.size(-1);

      int64_t y_offset = (max_height - img_height) / 2;

      new_img.slice(-1, x_offset, x_offset + img_width)
          .slice(-2, y_offset, y_offset + img_height)
          .copy_(img);

      x_offset += img_width;
    }

    return new_img.squeeze(0);
  }

 private:
  int64_t vae_scale_factor_;
  int64_t vae_latent_channels_;
};
TORCH_MODULE(Flux2ImageProcessor);
}  // namespace xllm

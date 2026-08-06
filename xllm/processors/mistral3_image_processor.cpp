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

#include "mistral3_image_processor.h"

#include <algorithm>
#include <cmath>

#include "processors/transforms.h"

namespace xllm {

Mistral3ImageProcessor::Mistral3ImageProcessor(const ModelArgs& args) {
  image_mean_ =
      torch::tensor(std::vector<double>{0.48145466, 0.4578275, 0.40821073},
                    torch::dtype(torch::kFloat32));
  image_std_ =
      torch::tensor(std::vector<double>{0.26862954, 0.26130258, 0.27577711},
                    torch::dtype(torch::kFloat32));
  rescale_factor_ = 0.00392156862745098;

  patch_size_ = args.mm_patch_size();
  merge_size_ = args.mm_spatial_merge_size();
  max_longest_edge_ = args.mm_image_size();

  // Fuse rescale into normalize: (pixel/255 - mean) / std = pixel * f -
  // mean/std
  if (do_rescale_ && do_normalize_) {
    image_mean_.mul_(1.0 / rescale_factor_);
    image_std_.mul_(1.0 / rescale_factor_);
    do_rescale_ = false;
  }
}

bool Mistral3ImageProcessor::process_image(
    const std::vector<torch::Tensor>& images,
    std::vector<torch::Tensor>& pixel_values,
    std::vector<torch::Tensor>& thw) const {
  pixel_values.clear();
  thw.clear();
  pixel_values.reserve(images.size());
  thw.reserve(images.size());

  for (const auto& img : images) {
    int64_t height = img.size(1);
    int64_t width = img.size(2);
    torch::Tensor processed = img;

    // Resize: ensure H, W divisible by patch_size, longest edge <= max
    if (do_resize_) {
      auto [new_h, new_w] =
          smart_resize(height, width, patch_size_, max_longest_edge_);
      processed = transforms::resize(processed,
                                     {new_h, new_w},
                                     /*resample=*/3,
                                     /*antialias=*/true);
      height = new_h;
      width = new_w;
    }

    // Normalize (rescale already fused in)
    if (do_normalize_) {
      processed = transforms::normalize(processed, image_mean_, image_std_);
    }

    // Rescale (only if not fused)
    if (do_rescale_) {
      processed = transforms::rescale(processed, rescale_factor_);
    }

    // pixel_values: [1, C, H, W] for Conv2d patch embedding inside model
    pixel_values.push_back(processed.unsqueeze(0).to(torch::kFloat32));

    // image_grid_thw: [[1, grid_h, grid_w]]
    int64_t grid_h = height / patch_size_;
    int64_t grid_w = width / patch_size_;
    thw.push_back(torch::tensor({int64_t(1), grid_h, grid_w},
                                torch::TensorOptions().dtype(torch::kInt64))
                      .reshape({1, 3}));
  }

  return true;
}

bool Mistral3ImageProcessor::process(
    const std::vector<torch::Tensor>& images,
    std::vector<MMDataItem>& output_items) const {
  std::vector<torch::Tensor> pixel_values;
  std::vector<torch::Tensor> thw;
  if (!process_image(images, pixel_values, thw)) {
    return false;
  }

  output_items.clear();
  output_items.reserve(images.size());
  const size_t image_size = images.size();
  for (size_t index = 0; index < image_size; ++index) {
    output_items.emplace_back(MMType::IMAGE,
                              MMDict{{"pixel_values", pixel_values[index]},
                                     {"image_grid_thw", thw[index]}});
  }
  return true;
}

std::pair<int64_t, int64_t> Mistral3ImageProcessor::smart_resize(
    int64_t height,
    int64_t width,
    int64_t patch_size,
    int64_t max_longest_edge) const {
  // Align to patch_size * merge_size so that grid_h and grid_w are
  // divisible by merge_size (required by spatial patch merger).
  int64_t align = patch_size * merge_size_;
  int64_t new_h =
      static_cast<int64_t>(std::round(static_cast<double>(height) / align)) *
      align;
  int64_t new_w =
      static_cast<int64_t>(std::round(static_cast<double>(width) / align)) *
      align;

  int64_t longest = std::max(new_h, new_w);
  if (longest > max_longest_edge) {
    double scale = static_cast<double>(max_longest_edge) / longest;
    new_h = static_cast<int64_t>(std::floor(new_h * scale / align)) * align;
    new_w = static_cast<int64_t>(std::floor(new_w * scale / align)) * align;
  }

  new_h = std::max(new_h, align);
  new_w = std::max(new_w, align);

  return {new_h, new_w};
}

}  // namespace xllm

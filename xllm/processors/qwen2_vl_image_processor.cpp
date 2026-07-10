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

#include "processors/qwen2_vl_image_processor.h"

#include "processors/transforms.h"

namespace xllm {

namespace {

using Size = std::pair<int32_t, int32_t>;

std::optional<Size> smart_resize(int32_t height,
                                 int32_t width,
                                 int32_t factor = 28,
                                 int32_t min_pixels = 56 * 56,
                                 int32_t max_pixels = 14 * 14 * 4 * 1280) {
  if (static_cast<double>(std::max(height, width)) / std::min(height, width) >
      200) {
    LOG(ERROR) << "Absolute aspect ratio must be smaller than 200, height: "
               << height << ", width: " << width;
    return std::nullopt;
  }

  int32_t h_bar =
      static_cast<int32_t>(std::rint(height / static_cast<double>(factor))) *
      factor;
  int32_t w_bar =
      static_cast<int32_t>(std::rint(width / static_cast<double>(factor))) *
      factor;

  int64_t resized_pixels = static_cast<int64_t>(h_bar) * w_bar;
  if (resized_pixels > max_pixels) {
    double beta = std::sqrt((static_cast<int64_t>(height) * width) /
                            static_cast<double>(max_pixels));
    h_bar = static_cast<int32_t>(
                std::floor(height / beta / static_cast<double>(factor))) *
            factor;
    w_bar = static_cast<int32_t>(
                std::floor(width / beta / static_cast<double>(factor))) *
            factor;
  } else if (resized_pixels < min_pixels) {
    double beta = std::sqrt(
        min_pixels / static_cast<double>(static_cast<int64_t>(height) * width));
    h_bar = static_cast<int32_t>(
                std::ceil(height * beta / static_cast<double>(factor))) *
            factor;
    w_bar = static_cast<int32_t>(
                std::ceil(width * beta / static_cast<double>(factor))) *
            factor;
  }

  return std::make_pair(h_bar, w_bar);
}

}  // namespace

Qwen2VLImageProcessor::Qwen2VLImageProcessor(const ModelArgs& args) {
  image_mean_ = torch::tensor(args.mm_image_normalize_mean(),
                              torch::dtype(torch::kFloat32));
  image_std_ = torch::tensor(args.mm_image_normalize_std(),
                             torch::dtype(torch::kFloat32));
  if (args.mm_image_max_pixels() && args.mm_image_min_pixels()) {
    min_pixels_ = args.mm_image_min_pixels();
    max_pixels_ = args.mm_image_max_pixels();
  } else if (args.mm_image_shortest_edge() && args.mm_image_longest_edge()) {
    min_pixels_ = args.mm_image_shortest_edge();
    max_pixels_ = args.mm_image_longest_edge();
  }
  if (args.mm_image_patch_size() > 0) {
    patch_size_ = args.mm_image_patch_size();
  }
  if (args.mm_image_temporal_patch_size() > 0) {
    temporal_patch_size_ = args.mm_image_temporal_patch_size();
  }
  if (args.mm_image_merge_size() > 0) {
    merge_size_ = args.mm_image_merge_size();
  }

  if (do_rescale_ && do_normalize_) {
    image_mean_.mul_(1.0 / rescale_factor_);
    image_std_.mul_(1.0 / rescale_factor_);
    do_rescale_ = false;
  }
}

bool Qwen2VLImageProcessor::process_image(
    const std::vector<torch::Tensor>& images,
    std::vector<torch::Tensor>& pixel_values,
    std::vector<torch::Tensor>& thw) const {
  torch::Tensor batch_images = torch::stack(images);
  const auto shape = batch_images.sizes();
  const int64_t batch_size = shape[0];
  int64_t resized_height = shape[2];
  int64_t resized_width = shape[3];

  if (do_resize_) {
    auto size = smart_resize(static_cast<int32_t>(resized_height),
                             static_cast<int32_t>(resized_width),
                             patch_size_ * merge_size_,
                             min_pixels_,
                             max_pixels_);
    if (!size) {
      return false;
    }

    std::tie(resized_height, resized_width) = *size;
    batch_images = transforms::resize(
        batch_images, {resized_height, resized_width}, resample_, true);
  }

  if (do_normalize_) {
    batch_images = transforms::normalize(batch_images, image_mean_, image_std_);
  }

  if (do_rescale_) {
    batch_images = transforms::rescale(batch_images, rescale_factor_);
  }

  torch::Tensor patches = batch_images.unsqueeze(1);
  if (temporal_patch_size_ > 1) {
    torch::Tensor repeats =
        patches.repeat({1, temporal_patch_size_ - 1, 1, 1, 1});
    patches = torch::cat({patches, repeats}, 1);
  }

  const auto patch_shape = patches.sizes();
  const int64_t channel = patch_shape[2];
  const int64_t grid_t = patch_shape[1] / temporal_patch_size_;
  const int64_t grid_h = resized_height / patch_size_;
  const int64_t grid_w = resized_width / patch_size_;

  patches = patches.view({batch_size,
                          grid_t,
                          temporal_patch_size_,
                          channel,
                          grid_h / merge_size_,
                          merge_size_,
                          patch_size_,
                          grid_w / merge_size_,
                          merge_size_,
                          patch_size_});

  patches = patches.permute({0, 1, 4, 7, 5, 8, 3, 2, 6, 9});
  torch::Tensor batch_pixel_values = patches.reshape(
      {batch_size,
       grid_t * grid_h * grid_w,
       channel * temporal_patch_size_ * patch_size_ * patch_size_});
  torch::Tensor batch_thw = torch::tensor({grid_t, grid_h, grid_w})
                                .repeat({batch_size, 1})
                                .reshape({batch_size, 1, 3});

  pixel_values = batch_pixel_values.unbind(0);
  thw = batch_thw.unbind(0);
  return true;
}

bool Qwen2VLImageProcessor::process(
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

}  // namespace xllm

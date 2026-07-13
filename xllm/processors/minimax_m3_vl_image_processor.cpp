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

#include "processors/minimax_m3_vl_image_processor.h"

#include <cmath>
#include <optional>
#include <utility>

#include "processors/transforms.h"

namespace xllm {
namespace {

using Size = std::pair<int32_t, int32_t>;

int32_t round_by_factor(int32_t number, int32_t factor) {
  const int32_t quotient = number / factor;
  const int32_t remainder = number % factor;
  const int32_t doubled_remainder = remainder * 2;
  if (doubled_remainder < factor) {
    return quotient * factor;
  }
  if (doubled_remainder > factor) {
    return (quotient + 1) * factor;
  }
  return (quotient % 2 == 0) ? quotient * factor : (quotient + 1) * factor;
}

int32_t ceil_by_factor(double number, int32_t factor) {
  return static_cast<int32_t>(std::ceil(number / factor)) * factor;
}

int32_t floor_by_factor(double number, int32_t factor) {
  return static_cast<int32_t>(std::floor(number / factor)) * factor;
}

std::optional<Size> smart_resize(int32_t height,
                                 int32_t width,
                                 int32_t factor,
                                 int32_t min_pixels,
                                 int32_t max_pixels) {
  if (static_cast<double>(std::max(height, width)) / std::min(height, width) >
      200) {
    LOG(ERROR) << "Absolute aspect ratio must be smaller than 200, height: "
               << height << ", width: " << width;
    return std::nullopt;
  }

  int32_t h_bar = std::max(factor, round_by_factor(height, factor));
  int32_t w_bar = std::max(factor, round_by_factor(width, factor));
  int64_t resized_pixels = static_cast<int64_t>(h_bar) * w_bar;
  if (resized_pixels > max_pixels) {
    const double beta = std::sqrt((static_cast<int64_t>(height) * width) /
                                  static_cast<double>(max_pixels));
    h_bar = floor_by_factor(height / beta, factor);
    w_bar = floor_by_factor(width / beta, factor);
  } else if (resized_pixels < min_pixels) {
    const double beta = std::sqrt(
        min_pixels / static_cast<double>(static_cast<int64_t>(height) * width));
    h_bar = ceil_by_factor(height * beta, factor);
    w_bar = ceil_by_factor(width * beta, factor);
  }

  return std::make_pair(h_bar, w_bar);
}

}  // namespace

MiniMaxM3VLImageProcessor::MiniMaxM3VLImageProcessor(const ModelArgs& args) {
  image_mean_ = torch::tensor(args.mm_image_normalize_mean(),
                              torch::dtype(torch::kFloat32));
  image_std_ = torch::tensor(args.mm_image_normalize_std(),
                             torch::dtype(torch::kFloat32));
  max_pixels_ =
      args.mm_image_max_pixels() > 0 ? args.mm_image_max_pixels() : max_pixels_;
  min_pixels_ =
      args.mm_image_min_pixels() > 0 ? args.mm_image_min_pixels() : min_pixels_;
  patch_size_ =
      args.mm_image_patch_size() > 0 ? args.mm_image_patch_size() : patch_size_;
  temporal_patch_size_ = args.mm_image_temporal_patch_size() > 0
                             ? args.mm_image_temporal_patch_size()
                             : args.mm_temporal_patch_size();
  merge_size_ = args.mm_image_merge_size() > 0 ? args.mm_image_merge_size()
                                               : args.mm_spatial_merge_size();

  image_mean_.mul_(1.0 / rescale_factor_);
  image_std_.mul_(1.0 / rescale_factor_);
}

bool MiniMaxM3VLImageProcessor::process_image(
    const std::vector<torch::Tensor>& images,
    std::vector<torch::Tensor>& pixel_values,
    std::vector<torch::Tensor>& thw) const {
  pixel_values.clear();
  thw.clear();
  pixel_values.reserve(images.size());
  thw.reserve(images.size());
  for (const torch::Tensor& image : images) {
    std::vector<torch::Tensor> image_pixel_values;
    std::vector<torch::Tensor> image_thw;
    if (!process_image_batch({image}, image_pixel_values, image_thw)) {
      return false;
    }
    pixel_values.push_back(std::move(image_pixel_values[0]));
    thw.push_back(std::move(image_thw[0]));
  }
  return true;
}

bool MiniMaxM3VLImageProcessor::process_image_batch(
    const std::vector<torch::Tensor>& images,
    std::vector<torch::Tensor>& pixel_values,
    std::vector<torch::Tensor>& thw) const {
  torch::Tensor batch_images = torch::stack(images);
  const torch::IntArrayRef shape = batch_images.sizes();
  const int64_t batch_size = shape[0];
  int64_t resized_height = shape[2];
  int64_t resized_width = shape[3];

  std::optional<Size> size = smart_resize(static_cast<int32_t>(resized_height),
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
  batch_images = transforms::normalize(batch_images, image_mean_, image_std_);

  torch::Tensor patches = batch_images.unsqueeze(1);
  if (temporal_patch_size_ > 1) {
    torch::Tensor repeats =
        patches
            .index({torch::indexing::Slice(),
                    torch::indexing::Slice(-1, torch::indexing::None)})
            .repeat({1, temporal_patch_size_ - 1, 1, 1, 1});
    patches = torch::cat({patches, repeats}, 1);
  }

  const torch::IntArrayRef patch_shape = patches.sizes();
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
  torch::Tensor batch_thw =
      torch::tensor(
          {grid_t, grid_h, grid_w},
          torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU))
          .repeat({batch_size, 1})
          .reshape({batch_size, 1, 3});

  pixel_values = batch_pixel_values.unbind(0);
  thw = batch_thw.unbind(0);
  return true;
}

bool MiniMaxM3VLImageProcessor::process(
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

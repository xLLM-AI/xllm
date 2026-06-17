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

#include "processors/transforms.h"

#include <glog/logging.h>

#include <algorithm>

namespace xllm::transforms {

torch::Tensor resize(const torch::Tensor& image,
                     const std::vector<int64_t>& size,
                     int32_t resample,
                     bool antialias) {
  if (image.dim() != 3 && image.dim() != 4) {
    LOG(FATAL) << "Input image must be a 3D or 4D tensor "
               << "((C x H x W) or (N x C x H x W)).";
  }
  auto options = torch::nn::functional::InterpolateFuncOptions()
                     .size(size)
                     .align_corners(false)
                     .antialias(antialias);
  switch (resample) {
    case 1:
      options.mode(torch::kNearest);
      break;
    case 2:
      options.mode(torch::kBilinear);
      break;
    case 3:
      options.mode(torch::kBicubic);
      break;
    default:
      LOG(FATAL) << "Invalid resample value. Must be one of 1, 2, or 3.";
  }
  auto input = image.dim() == 3 ? image.unsqueeze(0) : image;
  auto resized =
      torch::nn::functional::interpolate(input, options).clamp(0, 255);
  return image.dim() == 3 ? resized.squeeze(0) : resized;
}

torch::Tensor center_crop(const torch::Tensor& image,
                          const std::pair<int32_t, int32_t>& crop_size) {
  if (image.dim() != 3) {
    LOG(FATAL)
        << "Input image must be a 3-dimensional tensor in (C, H, W) format.";
  }

  int32_t crop_height = crop_size.first;
  int32_t crop_width = crop_size.second;
  int32_t orig_height = image.size(1);
  int32_t orig_width = image.size(2);

  int32_t top = (orig_height - crop_height) / 2;
  int32_t bottom = top + crop_height;
  int32_t left = (orig_width - crop_width) / 2;
  int32_t right = left + crop_width;

  if (top >= 0 && bottom <= orig_height && left >= 0 && right <= orig_width) {
    return image.index({torch::indexing::Slice(),
                        torch::indexing::Slice(top, bottom),
                        torch::indexing::Slice(left, right)});
  }

  int32_t new_height = std::max(crop_height, orig_height);
  int32_t new_width = std::max(crop_width, orig_width);
  auto padded_image =
      torch::zeros({image.size(0), new_height, new_width}, image.options());

  int32_t top_pad = (new_height - orig_height + 1) / 2;
  int32_t left_pad = (new_width - orig_width + 1) / 2;

  padded_image.index_put_(
      {torch::indexing::Slice(),
       torch::indexing::Slice(top_pad, top_pad + orig_height),
       torch::indexing::Slice(left_pad, left_pad + orig_width)},
      image);

  top = (new_height - crop_height) / 2;
  bottom = top + crop_height;
  left = (new_width - crop_width) / 2;
  right = left + crop_width;

  return padded_image.index({torch::indexing::Slice(),
                             torch::indexing::Slice(top, bottom),
                             torch::indexing::Slice(left, right)});
}

torch::Tensor rescale(const torch::Tensor& image, double scale) {
  return image * scale;
}

torch::Tensor normalize(const torch::Tensor& image,
                        const torch::Tensor& mean,
                        const torch::Tensor& std) {
  if (image.dim() != 3 && image.dim() != 4) {
    LOG(FATAL) << "Input image must be a 3D or 4D tensor "
               << "((C, H, W) or (N, C, H, W)).";
  }

  int32_t num_channels = image.dim() == 3 ? image.size(0) : image.size(1);
  if (mean.dim() != 1 || std.dim() != 1 || mean.size(0) != num_channels ||
      std.size(0) != num_channels) {
    LOG(FATAL) << "Mean and std tensors must have the same number "
               << "of elements as the number of channels in the "
               << "image.";
  }

  auto result = image;
  if (!image.is_floating_point()) {
    result = image.to(torch::kFloat32);
  }

  auto device = image.device();
  auto options = torch::dtype(torch::kFloat32).device(device);

  auto mean_shape = image.dim() == 3 ? std::vector<int64_t>{-1, 1, 1}
                                     : std::vector<int64_t>{1, -1, 1, 1};
  auto mean_tensor = mean.to(options).reshape(mean_shape);
  auto std_tensor = std.to(options).reshape(mean_shape);

  result = result.sub(mean_tensor);
  return result.div_(std_tensor);
}

}  // namespace xllm::transforms

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

#include <glog/logging.h>
#include <torch/torch.h>

#include <optional>
#include <string>
#include <vector>

#include "framework/model_context.h"
#include "models/dit/utils/util.h"

namespace xllm {

class VAEImageProcessorImpl : public torch::nn::Module {
 public:
  explicit VAEImageProcessorImpl(
      ModelContext context,
      bool do_resize = true,
      bool do_normalize = true,
      bool do_binarize = false,
      bool do_convert_rgb = false,
      bool do_convert_grayscale = false,
      int64_t latent_channels = 4,
      std::optional<int64_t> scale_factor = std::nullopt) {
    const auto& model_args = context.get_model_args();
    options_ = context.get_tensor_options();
    scale_factor_ = scale_factor.has_value()
                        ? scale_factor.value()
                        : 1 << model_args.block_out_channels().size();
    latent_channels_ = latent_channels;
    do_resize_ = do_resize;
    do_normalize_ = do_normalize;
    do_binarize_ = do_binarize;
    do_convert_rgb_ = do_convert_rgb;
    do_convert_grayscale_ = do_convert_grayscale;
  }

  std::pair<int64_t, int64_t> adjust_dimensions(int64_t height,
                                                int64_t width) const {
    height = height - (height % scale_factor_);
    width = width - (width % scale_factor_);
    return {height, width};
  }

  torch::Tensor preprocess(
      const torch::Tensor& image,
      std::optional<int64_t> height = std::nullopt,
      std::optional<int64_t> width = std::nullopt,
      const std::string& resize_mode = "lanczos",
      std::optional<std::tuple<int64_t, int64_t, int64_t, int64_t>>
          crop_coords = std::nullopt) {
    torch::Tensor processed = image;
    if (crop_coords.has_value()) {
      auto [x1, y1, x2, y2] = crop_coords.value();
      x1 = std::max(int64_t(0), x1);
      y1 = std::max(int64_t(0), y1);
      x2 = std::min(processed.size(-1), x2);
      y2 = std::min(processed.size(-2), y2);

      if (processed.dim() == 3) {
        processed = processed.index({torch::indexing::Slice(),
                                     torch::indexing::Slice(y1, y2),
                                     torch::indexing::Slice(x1, x2)});
      } else if (processed.dim() == 4) {
        processed = processed.index({torch::indexing::Slice(),
                                     torch::indexing::Slice(),
                                     torch::indexing::Slice(y1, y2),
                                     torch::indexing::Slice(x1, x2)});
      }
    }
    int64_t channel = processed.size(1);
    if (channel == latent_channels_) {
      return image;
    }
    auto [target_h, target_w] =
        get_default_height_width(processed, height, width);
    if (do_resize_) {
      processed = resize(processed, target_h, target_w, resize_mode);
    }
    if (processed.max().item<float>() > 1.1f) {
      processed = processed / 255.0f;
    }
    if (do_normalize_) {
      processed = normalize(processed);
    }
    if (do_binarize_) {
      processed = (processed >= 0.5f).to(torch::kFloat32);
    }
    processed = processed.to(options_);
    return processed;
  }

  torch::Tensor postprocess(
      const torch::Tensor& tensor,
      std::optional<std::vector<bool>> do_denormalize = std::nullopt) {
    torch::Tensor processed = tensor.clone();
    if (do_normalize_) {
      if (!do_denormalize.has_value()) {
        processed = denormalize(processed);
      } else {
        for (int64_t i = 0; i < processed.size(0); ++i) {
          if (i < do_denormalize.value().size() && do_denormalize.value()[i]) {
            processed[i] = denormalize(processed[i]);
          }
        }
      }
    }
    return processed;
  }

 private:
  std::pair<int64_t, int64_t> get_default_height_width(
      const torch::Tensor& image,
      std::optional<int64_t> height = std::nullopt,
      std::optional<int64_t> width = std::nullopt) const {
    int64_t h, w;
    if (image.dim() == 3) {
      h = image.size(1);
      w = image.size(2);
    } else if (image.dim() == 4) {
      h = image.size(2);
      w = image.size(3);
    } else {
      LOG(FATAL) << "Unsupported image dimension: " << image.dim();
    }

    int64_t target_h = height.value_or(h);
    int64_t target_w = width.value_or(w);
    return adjust_dimensions(target_h, target_w);
  }

  torch::Tensor normalize(const torch::Tensor& tensor) const {
    return 2.0 * tensor - 1.0;
  }

  torch::Tensor denormalize(const torch::Tensor& tensor) const {
    return (tensor * 0.5 + 0.5).clamp(0.0, 1.0);
  }

 public:
  torch::Tensor resize(torch::Tensor image,
                       int64_t height,
                       int64_t width,
                       const std::string& resize_mode = "lanczos") {
    auto options = image.options();
    image = image.cpu();

    bool squeeze_batch = false;
    if (image.dim() == 3) {
      image = image.unsqueeze(0);
      squeeze_batch = true;
    }
    CHECK_EQ(image.dim(), 4) << "resize expects CHW or BCHW image";

    torch::Tensor resized;
    if (resize_mode == "lanczos") {
      std::vector<torch::Tensor> resized_images;
      resized_images.reserve(image.size(0));
      for (int64_t i = 0; i < image.size(0); ++i) {
        auto chw_image = image[i];
        auto hwc_image = chw_image.permute({1, 2, 0}).contiguous();

        int64_t h = hwc_image.size(0);
        int64_t w = hwc_image.size(1);
        int64_t c = hwc_image.size(2);

        torch::Tensor out = torch::empty({height, width, c}, torch::kUInt8);
        lanczos::resize_8bpc(hwc_image.data_ptr<uint8_t>(),
                             static_cast<int32_t>(w),
                             static_cast<int32_t>(h),
                             static_cast<int32_t>(c),
                             static_cast<int32_t>(width),
                             static_cast<int32_t>(height),
                             out.data_ptr<uint8_t>());
        resized_images.emplace_back(out.permute({2, 0, 1}));
      }
      resized = torch::stack(resized_images);
    } else if (resize_mode == "bicubic") {
      auto options = torch::nn::functional::InterpolateFuncOptions()
                         .size(std::vector<int64_t>{height, width})
                         .align_corners(false)
                         .antialias(true)
                         .mode(torch::kBicubic);
      resized = torch::nn::functional::interpolate(image, options);
    } else {
      LOG(FATAL) << "Currently only support 'lanczos' and 'bicubic'"
                 << ", but got: " << resize_mode;
    }

    if (squeeze_batch) {
      resized = resized.squeeze(0);
    }
    return resized.to(options);
  }

 private:
  int64_t scale_factor_ = 8;
  int64_t latent_channels_ = 4;
  bool do_resize_ = true;
  bool do_normalize_ = true;
  bool do_binarize_ = false;
  bool do_convert_rgb_ = false;
  bool do_convert_grayscale_ = false;
  torch::TensorOptions options_;
};
TORCH_MODULE(VAEImageProcessor);

}  // namespace xllm

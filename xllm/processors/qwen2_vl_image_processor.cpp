/* Copyright 2025-2026 The xLLM Authors.

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

#include "processors/qwen2_vl_image_processor.h"

#include <string>
#include <vector>

#include "core/framework/config/model_config.h"
#include "processors/python_image_processor.h"
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

// token-budget smart_resize, mirroring HF Glm5NextImageProcessor. The
// pixel budget is ``tokens * pixels_per_token`` where
// ``pixels_per_token = temporal_factor * factor^2`` and ``factor =
// patch_size * merge_size``. For a single image HF passes
// ``num_frames == temporal_factor`` (the image is temporally duplicated), so
// ``aligned_frames`` is at least ``temporal_factor``; we mirror that here.
// Used as a fallback for the model_impl=python path when the HF
// ``Glm5NextImageProcessor`` class is unavailable (stock transformers), so
// image input no longer requires the dev (patched) transformers.
std::optional<Size> smart_resize_tokens(int32_t num_frames,
                                        int32_t height,
                                        int32_t width,
                                        int32_t temporal_factor,
                                        int32_t factor,
                                        int32_t min_tokens,
                                        int32_t max_tokens) {
  if (static_cast<double>(std::max(height, width)) / std::min(height, width) >
      200) {
    LOG(ERROR) << "Absolute aspect ratio must be smaller than 200, height: "
               << height << ", width: " << width;
    return std::nullopt;
  }

  const int64_t pixels_per_token =
      static_cast<int64_t>(temporal_factor) * factor * factor;
  int64_t min_pixels = static_cast<int64_t>(min_tokens) * pixels_per_token;
  int64_t max_pixels = static_cast<int64_t>(max_tokens) * pixels_per_token;

  auto align = [](int32_t value, int32_t f) {
    return ((value + f - 1) / f) * f;
  };

  auto fit_within_budget = [&](int32_t aligned_frames) -> Size {
    int32_t low = 1;
    int32_t high = height;
    int32_t best_h = factor;
    int32_t best_w = factor;
    while (low <= high) {
      int32_t content_h = (low + high) / 2;
      int32_t content_w =
          std::max(1,
                   static_cast<int32_t>(std::floor(static_cast<double>(width) *
                                                   content_h / height)));
      int32_t cand_h = align(content_h, factor);
      int32_t cand_w = align(content_w, factor);
      int64_t budget = static_cast<int64_t>(aligned_frames) * cand_h * cand_w;
      if (budget <= max_pixels) {
        best_h = cand_h;
        best_w = cand_w;
        low = content_h + 1;
      } else {
        high = content_h - 1;
      }
    }
    return std::make_pair(best_h, best_w);
  };

  int32_t aligned_frames = std::max(
      temporal_factor,
      static_cast<int32_t>(
          std::rint(num_frames / static_cast<double>(temporal_factor))) *
          temporal_factor);
  int32_t aligned_h = align(height, factor);
  int32_t aligned_w = align(width, factor);
  int64_t aligned_budget =
      static_cast<int64_t>(aligned_frames) * aligned_h * aligned_w;

  if (aligned_budget < min_pixels) {
    double scale =
        std::sqrt(static_cast<double>(min_pixels) /
                  (static_cast<double>(num_frames) * height * width));
    aligned_h = align(
        std::max(1, static_cast<int32_t>(std::ceil(height * scale))), factor);
    aligned_w = align(
        std::max(1, static_cast<int32_t>(std::ceil(width * scale))), factor);
    aligned_budget =
        static_cast<int64_t>(aligned_frames) * aligned_h * aligned_w;
  }

  if (aligned_budget > max_pixels) {
    auto fitted = fit_within_budget(aligned_frames);
    aligned_h = fitted.first;
    aligned_w = fitted.second;
  }

  return std::make_pair(aligned_h, aligned_w);
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
  min_tokens_ = args.mm_image_min_tokens();
  max_tokens_ = args.mm_image_max_tokens();

  if (do_rescale_ && do_normalize_) {
    image_mean_.mul_(1.0 / rescale_factor_);
    image_std_.mul_(1.0 / rescale_factor_);
    do_rescale_ = false;
  }
  if (::xllm::ModelConfig::is_python_model_impl(
          ::xllm::ModelConfig::get_instance().model_impl())) {
    initialize_python_image_processor();
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
    std::optional<Size> size;
    if (min_tokens_ > 0 && max_tokens_ > 0) {
      // token-budget resize (matches HF Glm5NextImageProcessor). For a
      // single image HF passes num_frames == temporal_factor.
      size = smart_resize_tokens(temporal_patch_size_,
                                 static_cast<int32_t>(resized_height),
                                 static_cast<int32_t>(resized_width),
                                 temporal_patch_size_,
                                 patch_size_ * merge_size_,
                                 min_tokens_,
                                 max_tokens_);
    } else {
      size = smart_resize(static_cast<int32_t>(resized_height),
                          static_cast<int32_t>(resized_width),
                          patch_size_ * merge_size_,
                          min_pixels_,
                          max_pixels_);
    }
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
  // Python model executor: delegate image preprocessing to Python (HF
  // AutoImageProcessor). model_impl=python => no per-model C++ image
  // processor; the Python model's encode() consumes the HF output directly.
  if (::xllm::ModelConfig::is_python_model_impl(
          ::xllm::ModelConfig::get_instance().model_impl())) {
    if (process_python_images(images, output_items)) {
      return true;
    }
    // The HF ``Glm5NextImageProcessor`` class only exists in the dev (patched)
    // transformers. On a stock transformers build ``AutoImageProcessor`` fails
    // to resolve it; fall back to the C++ token-budget path, whose patchify is
    // byte-identical to HF's (verified) so the Python model consumes it
    // unchanged.
    LOG(WARNING)
        << "HF AutoImageProcessor unavailable for this model (likely stock "
           "transformers); falling back to the C++ image processor.";
  }

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

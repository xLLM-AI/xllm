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

#include <torch/torch.h>

#include <cstdint>
#include <tuple>
#include <vector>

#include "core/framework/model/model_args.h"
#include "processors/image_processor.h"
namespace xllm {

class MiniCPMVImageProcessor final : public ImageProcessor {
 public:
  explicit MiniCPMVImageProcessor(const ModelArgs& args);

  static std::pair<int32_t, int32_t> get_sliced_grid(
      const std::pair<int32_t, int32_t>& original_size,
      int32_t max_slice_nums,
      int32_t scale_resolution,
      bool never_split = false);

  bool process(torch::Tensor image,
               std::vector<torch::Tensor>& new_images,
               std::vector<torch::Tensor>& tgt_sizes) const;
  bool process(const std::vector<torch::Tensor>& images,
               std::vector<MMDataItem>& output_items) const override;

 private:
  int32_t ensure_divide(int32_t length, int32_t patch_size) const {
    return std::max(
        static_cast<int32_t>(
            std::lround(static_cast<float>(length) / patch_size) * patch_size),
        patch_size);
  }

  std::pair<int32_t, int32_t> find_best_resize(
      const std::pair<int32_t, int32_t>& original_size,
      int32_t scale_resolution,
      int32_t patch_size,
      bool allow_upscale = false) const;

  std::pair<int32_t, int32_t> get_refine_size(
      const std::pair<int32_t, int32_t>& original_size,
      const std::pair<int32_t, int32_t>& grid,
      int32_t scale_resolution,
      int32_t patch_size,
      bool allow_upscale = false) const;

  std::tuple<torch::Tensor,
             std::vector<std::vector<torch::Tensor>>,
             std::pair<int32_t, int32_t>>
  slice_image(const torch::Tensor& image,
              int32_t max_slice_nums = 9,
              int32_t scale_resolution = 448,
              int32_t patch_size = 14,
              bool never_split = false) const;

  std::vector<std::vector<torch::Tensor>> split_to_patches(
      const torch::Tensor& image,
      const std::pair<int32_t, int32_t>& grid) const;

  torch::Tensor reshape_by_patch(const torch::Tensor& image) const;

  std::vector<torch::Tensor> get_sliced_images(
      const torch::Tensor& image,
      int32_t max_slice_nums = -1) const;

 private:
  bool slice_mode_;
  int32_t max_slice_nums_;
  int32_t scale_resolution_;
  int32_t patch_size_;
  int32_t image_feature_size_;
  torch::Tensor norm_mean_;
  torch::Tensor norm_std_;
};

}  // namespace xllm

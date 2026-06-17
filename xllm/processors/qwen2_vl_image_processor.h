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
#include <vector>

#include "core/framework/model/model_args.h"
#include "processors/image_processor.h"

namespace xllm {

class Qwen2VLImageProcessor final : public ImageProcessor {
 public:
  explicit Qwen2VLImageProcessor(const ModelArgs& args);

  bool process(const std::vector<torch::Tensor>& images,
               std::vector<MMDataItem>& output_items) const override;

  bool process_image(const std::vector<torch::Tensor>& images,
                     std::vector<torch::Tensor>& pixel_values,
                     std::vector<torch::Tensor>& thw) const;

 private:
  bool do_convert_rgb_ = true;
  bool do_normalize_ = true;
  bool do_rescale_ = true;
  bool do_resize_ = true;

  torch::Tensor image_mean_;
  torch::Tensor image_std_;

  int32_t max_pixels_ = 12845056;
  int32_t min_pixels_ = 3136;

  int32_t merge_size_ = 2;
  int32_t patch_size_ = 14;

  int32_t resample_ = 3;
  double rescale_factor_ = 0.00392156862745098;

  int32_t temporal_patch_size_ = 2;
};

}  // namespace xllm

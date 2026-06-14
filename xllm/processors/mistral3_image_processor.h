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

#include <utility>
#include <vector>

#include "image_processor.h"

namespace xllm {

class Mistral3ImageProcessor : public ImageProcessor {
 public:
  explicit Mistral3ImageProcessor(const ModelArgs& args);
  ~Mistral3ImageProcessor() override = default;

  bool process(const std::vector<torch::Tensor>& images,
               std::vector<MMDataItem>& output_items) const override;

 private:
  bool process_image(const std::vector<torch::Tensor>& images,
                     std::vector<torch::Tensor>& pixel_values,
                     std::vector<torch::Tensor>& thw) const;

  std::pair<int64_t, int64_t> smart_resize(int64_t height,
                                           int64_t width,
                                           int64_t patch_size,
                                           int64_t max_longest_edge) const;

  bool do_normalize_ = true;
  bool do_rescale_ = true;
  bool do_resize_ = true;

  torch::Tensor image_mean_;
  torch::Tensor image_std_;
  double rescale_factor_ = 0.00392156862745098;

  int64_t patch_size_ = 14;
  int64_t max_longest_edge_ = 1540;
  int64_t merge_size_ = 2;
};

}  // namespace xllm

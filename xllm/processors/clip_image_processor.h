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
namespace xllm {

class CLIPImageProcessor {
 public:
  explicit CLIPImageProcessor(const ModelArgs& args);

  torch::Tensor process_images(const torch::Tensor& images) const;

 private:
  std::vector<int64_t> get_resize_output_image_size(
      const torch::Tensor& image,
      int32_t shortest_edge) const;

 private:
  bool do_resize_;
  bool do_center_crop_;
  bool do_rescale_;
  bool do_normalize_;
  int32_t shortest_edge_;
  int32_t resample_;
  double rescale_factor_;
  std::pair<int32_t, int32_t> crop_size_;
  torch::Tensor image_mean_;
  torch::Tensor image_std_;
};

}  // namespace xllm

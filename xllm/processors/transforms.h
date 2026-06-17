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
#include <utility>
#include <vector>

namespace xllm::transforms {

torch::Tensor resize(const torch::Tensor& image,
                     const std::vector<int64_t>& size,
                     int32_t resample,
                     bool antialias = true);

torch::Tensor center_crop(const torch::Tensor& image,
                          const std::pair<int32_t, int32_t>& crop_size);

torch::Tensor rescale(const torch::Tensor& image, double scale);

torch::Tensor normalize(const torch::Tensor& image,
                        const torch::Tensor& mean,
                        const torch::Tensor& std);

}  // namespace xllm::transforms

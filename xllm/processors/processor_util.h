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

#include <array>
#include <cstdint>
#include <map>
#include <vector>

namespace xllm {

using ImageShape = std::array<int64_t, 3>;

struct ImageBatchBucket {
  std::vector<size_t> indices;
  std::vector<torch::Tensor> images;
};

inline std::map<ImageShape, ImageBatchBucket> group_images_by_shape(
    const std::vector<torch::Tensor>& images) {
  std::map<ImageShape, ImageBatchBucket> buckets;
  const size_t image_size = images.size();
  for (size_t index = 0; index < image_size; ++index) {
    const torch::Tensor& image = images[index];
    const auto sizes = image.sizes();
    ImageShape shape = {sizes[0], sizes[1], sizes[2]};
    ImageBatchBucket& bucket = buckets[shape];
    bucket.indices.push_back(index);
    bucket.images.push_back(image);
  }
  return buckets;
}

}  // namespace xllm

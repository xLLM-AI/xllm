/* Copyright 2026 The xLLM Authors.

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

#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "core/framework/multimodal/mm_batch_data.h"

namespace xllm::test {

inline MMBatchData make_vision_batch(int64_t frames, float pixel_value = 1.0F) {
  MMItemVec items;
  items.reserve(2);
  items.emplace_back(
      MMType::IMAGE,
      MMDict{{"image_grid_thw", torch::tensor({{1, 4, 4}}, torch::kInt32)},
             {"pixel_values", torch::full({16, 8}, pixel_value)}});
  items.emplace_back(
      MMType::VIDEO,
      MMDict{
          {"video_grid_thw",
           torch::tensor({{frames, int64_t{4}, int64_t{4}}}, torch::kInt64)},
          {"pixel_values_videos", torch::full({frames * 16, 8}, pixel_value)}});
  std::vector<MMData> requests;
  requests.reserve(1);
  requests.emplace_back(MMType::IMAGE | MMType::VIDEO, std::move(items));
  return MMBatchData(std::move(requests));
}

}  // namespace xllm::test

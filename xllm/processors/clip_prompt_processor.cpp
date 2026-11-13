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

#include "processors/clip_prompt_processor.h"

#include <cstdint>

#include "processors/vision_token_scanner.h"

namespace xllm {

CLIPVLPromptProcessor::CLIPVLPromptProcessor(const ModelArgs& args) {
  merge_size_ = args.mm_image_merge_size();
}

bool CLIPVLPromptProcessor::process(std::string& prompt,
                                    const MMData& mm_data) {
  torch::Tensor image_grid_thw;
  if (auto res = mm_data.get<torch::Tensor>("image_grid_thw")) {
    image_grid_thw = res.value();
  }

  torch::Tensor video_grid_thw;
  if (auto res = mm_data.get<torch::Tensor>("video_grid_thw")) {
    video_grid_thw = res.value();
  }

  if (!image_grid_thw.defined() && !video_grid_thw.defined()) {
    return true;
  }

  const int32_t merge_length = merge_size_ * merge_size_;
  int32_t total_image_token = 0;
  if (image_grid_thw.defined()) {
    const int64_t count = image_grid_thw.sizes()[0];
    for (int64_t idx = 0; idx < count; ++idx) {
      total_image_token +=
          image_grid_thw[idx].prod().item<int32_t>() / merge_length;
    }
  }

  int32_t total_video_token = 0;
  if (video_grid_thw.defined()) {
    const int64_t count = video_grid_thw.sizes()[0];
    for (int64_t idx = 0; idx < count; ++idx) {
      total_video_token +=
          video_grid_thw[idx].prod().item<int32_t>() / merge_length;
    }
  }

  size_t total_token_len = total_image_token * image_token_.size() +
                           total_video_token * video_token_.size();
  std::string data;
  data.reserve(prompt.size() + total_token_len);

  int32_t image_index = 0;
  int32_t video_index = 0;
  const torch::Tensor* grid_thw = nullptr;
  const std::string* token = nullptr;
  int32_t* index = nullptr;

  size_t begin = 0;
  VisionTokenScanner scanner(prompt, image_token_, video_token_);
  using TokenType = VisionTokenScanner::TokenType;
  auto pair = scanner.find_next(begin);
  while (pair.second != std::string::npos) {
    data.append(prompt, begin, pair.second - begin);
    if (pair.first == TokenType::IMAGE) {
      grid_thw = &image_grid_thw;
      token = &image_token_;
      index = &image_index;
    } else if (pair.first == TokenType::VIDEO) {
      grid_thw = &video_grid_thw;
      token = &video_token_;
      index = &video_index;
    } else {
      LOG(FATAL) << "Unexpected token type encountered.";
    }

    auto token_num =
        (*grid_thw)[(*index)].prod().item<int32_t>() / merge_length;
    while (token_num--) {
      data.append(*token);
    }

    ++(*index);
    begin = pair.second + token->size();
    pair = scanner.find_next(begin);
  }

  if (begin < prompt.size()) {
    data.append(prompt, begin, std::string::npos);
  }
  prompt = std::move(data);
  return true;
}

}  // namespace xllm

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

#include "processors/qwen2_vl_prompt_processor.h"

#include <algorithm>
#include <cstdint>

#include "processors/vision_token_scanner.h"

namespace xllm {

Qwen2VLPromptProcessor::Qwen2VLPromptProcessor(const ModelArgs& args) {
  merge_size_ = args.mm_image_merge_size();
  vision_start_token_id_ = args.vision_start_token_id();
  vision_end_token_id_ = args.vision_end_token_id();
  image_token_id_ = args.image_token_id();
  video_token_id_ = args.video_token_id();
}

bool Qwen2VLPromptProcessor::process(std::string& prompt,
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

    if (!grid_thw->defined() || *index >= grid_thw->size(0)) {
      return false;
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

bool Qwen2VLPromptProcessor::find_mm_spans(
    const std::vector<int32_t>& token_ids,
    MMData& mm_data) {
  if (mm_data.empty()) {
    return true;
  }
  auto& items = mm_data.items<MMItemVec>();
  auto start = token_ids.begin();
  size_t item_index = 0;
  int64_t image_tokens = 0;
  int64_t video_tokens = 0;
  const int32_t merge_length = merge_size_ * merge_size_;
  while (start != token_ids.end()) {
    auto begin = std::find(start, token_ids.end(), vision_start_token_id_);
    if (begin == token_ids.end()) {
      break;
    }
    auto end = std::find(begin + 1, token_ids.end(), vision_end_token_id_);
    if (end == token_ids.end() || end == begin + 1 ||
        item_index >= items.size()) {
      return false;
    }
    auto& item = items[item_index];
    const int32_t token = *(begin + 1);
    const bool image = token == image_token_id_;
    if ((!image && token != video_token_id_) ||
        item.type() != (image ? MMType::IMAGE : MMType::VIDEO) ||
        !std::all_of(
            begin + 1, end, [token](int32_t id) { return id == token; })) {
      return false;
    }
    const auto grid =
        item.get<torch::Tensor>(image ? "image_grid_thw" : "video_grid_thw");
    const int32_t length = static_cast<int32_t>(std::distance(begin + 1, end));
    if (!grid.has_value() ||
        length != grid->prod().item<int64_t>() / merge_length) {
      return false;
    }
    auto& state = item.mutable_state();
    state.mutable_token_pos() = {
        static_cast<int32_t>(std::distance(token_ids.begin(), begin)) + 1,
        length};
    state.mutable_mm_token_mask() = torch::ones(
        {length},
        torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    state.mutable_mm_token_num() = length;
    (image ? image_tokens : video_tokens) += length;
    ++item_index;
    start = end + 1;
  }
  return item_index == items.size() &&
         std::count(token_ids.begin(), token_ids.end(), image_token_id_) ==
             image_tokens &&
         std::count(token_ids.begin(), token_ids.end(), video_token_id_) ==
             video_tokens;
}

}  // namespace xllm

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

#include "processors/minimax_m3_vl_prompt_processor.h"

#include <algorithm>

namespace xllm {

MiniMaxM3VLPromptProcessor::MiniMaxM3VLPromptProcessor(const ModelArgs& args) {
  merge_size_ = args.mm_image_merge_size() > 0 ? args.mm_image_merge_size()
                                               : args.mm_spatial_merge_size();
  vision_start_token_id_ = args.vision_start_token_id();
  vision_end_token_id_ = args.vision_end_token_id();
  image_token_id_ = args.image_token_id();
}

void MiniMaxM3VLPromptProcessor::process(std::string& prompt,
                                         const MMData& mm_data) {
  torch::Tensor image_grid_thw;
  if (const auto& res = mm_data.get<torch::Tensor>("image_grid_thw")) {
    image_grid_thw = res.value();
  }
  if (!image_grid_thw.defined()) {
    return;
  }

  const int32_t merge_length = merge_size_ * merge_size_;
  int32_t total_image_token = 0;
  const int64_t image_count = image_grid_thw.sizes()[0];
  for (int64_t index = 0; index < image_count; ++index) {
    total_image_token +=
        image_grid_thw[index].prod().item<int32_t>() / merge_length;
  }

  const size_t total_token_len =
      total_image_token * image_token_.size() +
      static_cast<size_t>(image_count) *
          (vision_start_token_.size() + vision_end_token_.size());
  std::string data;
  data.reserve(prompt.size() + total_token_len);

  int64_t image_index = 0;
  size_t begin = 0;
  std::pair<bool, size_t> image_pos = find_image_token(prompt, begin);
  while (image_pos.first) {
    data.append(prompt, begin, image_pos.second - begin);
    CHECK_LT(image_index, image_count)
        << "More MiniMax-M3 image tokens than processed images.";
    int32_t token_num =
        image_grid_thw[image_index].prod().item<int32_t>() / merge_length;
    data.append(vision_start_token_);
    while (token_num-- > 0) {
      data.append(image_token_);
    }
    data.append(vision_end_token_);

    ++image_index;
    begin = image_pos.second + image_token_.size();
    image_pos = find_image_token(prompt, begin);
  }

  CHECK_EQ(image_index, image_count)
      << "Fewer MiniMax-M3 image tokens than processed images.";

  if (begin < prompt.size()) {
    data.append(prompt, begin, std::string::npos);
  }
  prompt = std::move(data);
}

void MiniMaxM3VLPromptProcessor::find_mm_spans(
    const std::vector<int32_t>& token_ids,
    MMData& mm_data) {
  std::vector<int32_t>::const_iterator start = token_ids.begin();
  int32_t global_mm_index = 0;
  MMItemVec& mm_items = mm_data.items<MMItemVec>();
  while (true) {
    std::vector<int32_t>::const_iterator vision_start_it =
        std::find(start, token_ids.end(), vision_start_token_id_);
    if (vision_start_it == token_ids.end()) {
      break;
    }
    std::vector<int32_t>::const_iterator vision_end_it =
        std::find(vision_start_it + 1, token_ids.end(), vision_end_token_id_);
    CHECK(vision_end_it != token_ids.end())
        << "MiniMax-M3 image span is missing vision end token.";

    const int32_t offset =
        static_cast<int32_t>(std::distance(token_ids.begin(), vision_start_it));
    const int32_t length =
        static_cast<int32_t>(std::distance(vision_start_it + 1, vision_end_it));
    CHECK_LT(global_mm_index, static_cast<int32_t>(mm_items.size()))
        << "More MiniMax-M3 image spans than multimodal items.";

    MMDataItem& item = mm_items[global_mm_index];
    if (length > 0 && *(vision_start_it + 1) == image_token_id_) {
      item.mutable_state().mutable_token_pos() = {offset + 1, length};
      torch::Tensor mask = torch::ones(
          {length},
          torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
      item.mutable_state().mutable_mm_token_mask() = mask;
      item.mutable_state().mutable_mm_token_num() = length;
    }

    ++global_mm_index;
    start = std::next(vision_end_it);
  }
}

std::pair<bool, size_t> MiniMaxM3VLPromptProcessor::find_image_token(
    const std::string& prompt,
    size_t begin) const {
  const size_t image_pos = prompt.find(image_token_, begin);
  return {image_pos != std::string::npos, image_pos};
}

}  // namespace xllm

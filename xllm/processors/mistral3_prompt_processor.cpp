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

#include "processors/mistral3_prompt_processor.h"

#include <torch/torch.h>

#include <algorithm>

namespace xllm {

Mistral3PromptProcessor::Mistral3PromptProcessor(const ModelArgs& args) {
  merge_size_ = args.mm_spatial_merge_size();
  image_token_id_ = args.image_token_id();
}

bool Mistral3PromptProcessor::process(std::string& prompt,
                                      const MMData& mm_data) {
  torch::Tensor image_grid_thw;
  if (auto res = mm_data.get<torch::Tensor>("image_grid_thw")) {
    image_grid_thw = res.value();
  }

  if (!image_grid_thw.defined()) {
    return true;
  }

  auto merge_length = merge_size_ * merge_size_;

  // Replace each [IMG] placeholder with the correct number of [IMG] tokens
  std::string data;
  data.reserve(prompt.size() +
               image_grid_thw.size(0) * 256 * image_token_.size());

  int32_t image_index = 0;
  size_t begin = 0;
  size_t pos = find_image_token(prompt, begin);

  while (pos != std::string::npos) {
    if (image_index >= image_grid_thw.size(0)) {
      return false;
    }
    data.append(prompt, begin, pos - begin);
    int32_t token_num =
        image_grid_thw[image_index].prod().item<int32_t>() / merge_length;
    while (token_num--) {
      data.append(image_token_);
    }
    ++image_index;
    begin = pos + image_token_.size();
    pos = find_image_token(prompt, begin);
  }

  if (begin < prompt.size()) {
    data.append(prompt, begin, std::string::npos);
  }

  prompt = std::move(data);
  return true;
}

bool Mistral3PromptProcessor::find_mm_spans(
    const std::vector<int32_t>& token_ids,
    MMData& mm_data) {
  if (mm_data.empty()) {
    return true;
  }
  auto& items = mm_data.items<MMItemVec>();
  size_t item_index = 0;
  size_t index = 0;
  while (index < token_ids.size()) {
    if (token_ids[index] != image_token_id_) {
      ++index;
      continue;
    }
    if (item_index >= items.size()) {
      return false;
    }
    auto& item = items[item_index];
    const auto grid = item.get<torch::Tensor>("image_grid_thw");
    if (item.type() != MMType::IMAGE || !grid.has_value()) {
      return false;
    }
    const int64_t length =
        grid->prod().item<int64_t>() / (merge_size_ * merge_size_);
    if (length <= 0 ||
        static_cast<uint64_t>(length) > token_ids.size() - index ||
        !std::all_of(token_ids.begin() + index,
                     token_ids.begin() + index + length,
                     [this](int32_t id) { return id == image_token_id_; })) {
      return false;
    }
    item.mutable_state().mutable_token_pos() = {static_cast<int32_t>(index),
                                                static_cast<int32_t>(length)};
    item.mutable_state().mutable_mm_token_mask() = torch::ones(
        {length},
        torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    item.mutable_state().mutable_mm_token_num() = static_cast<int32_t>(length);
    index += static_cast<size_t>(length);
    ++item_index;
  }
  return item_index == items.size();
}

size_t Mistral3PromptProcessor::find_image_token(const std::string& prompt,
                                                 size_t begin) {
  return prompt.find(image_token_, begin);
}

}  // namespace xllm

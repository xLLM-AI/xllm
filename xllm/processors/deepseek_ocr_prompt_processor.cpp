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

#include "processors/deepseek_ocr_prompt_processor.h"

#include <glog/logging.h>

namespace xllm {

DeepseekOCRPromptProcessor::DeepseekOCRPromptProcessor(const ModelArgs& args) {
  image_token_id_ = args.image_token_id();
  base_size_ = static_cast<int32_t>(args.mm_image_base_size());
  crop_size_ = static_cast<int32_t>(args.mm_image_size());
}

bool DeepseekOCRPromptProcessor::process(std::string& prompt,
                                         const MMData& mm_data) {
  torch::Tensor spatial_crop;
  if (auto res = mm_data.get<torch::Tensor>("image_spatial_crop")) {
    spatial_crop = res.value();
  }
  if (!spatial_crop.defined()) {
    return true;
  }

  const int64_t num_images = spatial_crop.sizes()[0];

  size_t total_grid_len = 0;
  for (int64_t i = 0; i < num_images; ++i) {
    const int32_t width_crops =
        static_cast<int32_t>(spatial_crop[i][0].item<int64_t>());
    const int32_t height_crops =
        static_cast<int32_t>(spatial_crop[i][1].item<int64_t>());
    total_grid_len +=
        static_cast<size_t>(grid_length(width_crops, height_crops));
  }

  std::string data;
  data.reserve(prompt.size() + total_grid_len * image_token_.size());

  size_t image_index = 0;
  size_t begin = 0;
  size_t pos = prompt.find(image_token_);
  while (pos != std::string::npos) {
    data.append(prompt, begin, pos - begin);
    if (image_index >= static_cast<size_t>(num_images)) {
      LOG(ERROR) << "More <image> placeholders than processed images.";
      return false;
    }
    const int32_t width_crops =
        static_cast<int32_t>(spatial_crop[image_index][0].item<int64_t>());
    const int32_t height_crops =
        static_cast<int32_t>(spatial_crop[image_index][1].item<int64_t>());

    // Emit exactly grid_length() placeholder tokens; every grid token is
    // the same image_token_ string, so only the count matters here.
    // find_mm_spans cuts the spans by the same grid_length().
    const int32_t length = grid_length(width_crops, height_crops);
    for (int32_t r = 0; r < length; ++r) {
      data.append(image_token_);
    }
    ++image_index;
    begin = pos + image_token_.size();
    pos = prompt.find(image_token_, begin);
  }
  if (image_index < static_cast<size_t>(num_images)) {
    LOG(ERROR) << "Fewer <image> placeholders than processed images.";
    return false;
  }
  if (begin < prompt.size()) {
    data.append(prompt, begin, std::string::npos);
  }
  prompt = std::move(data);
  return true;
}

bool DeepseekOCRPromptProcessor::find_mm_spans(
    const std::vector<int32_t>& token_ids,
    MMData& mm_data) {
  auto& mm_items = mm_data.items<MMItemVec>();

  // Expected token count per item from the single grid_length() formula.
  // Spans are cut by these lengths so two adjacent <image> placeholders (no
  // text between them, one contiguous token run) still map to separate items.
  std::vector<int32_t> expected;
  expected.reserve(mm_items.size());
  for (const auto& item : mm_items) {
    const auto crop = item.get<torch::Tensor>("image_spatial_crop");
    CHECK(crop.has_value() && crop->numel() == 2)
        << "image_spatial_crop missing for a multimodal item";
    // numel()==2 but the shape may carry a leading batch dim ((1,2) from the
    // image processor); flatten before the scalar reads.
    const torch::Tensor flat = crop->view({2});
    const int32_t width_crops = static_cast<int32_t>(flat[0].item<int64_t>());
    const int32_t height_crops = static_cast<int32_t>(flat[1].item<int64_t>());
    expected.push_back(grid_length(width_crops, height_crops));
  }

  size_t item_index = 0;
  auto close_span = [&](int32_t offset, int32_t length) {
    auto& item = mm_items[item_index++];
    item.mutable_state().mutable_token_pos() = {offset, length};
    item.mutable_state().mutable_mm_token_mask() = torch::ones(
        {length},
        torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    item.mutable_state().mutable_mm_token_num() = length;
  };

  const size_t tokens_num = token_ids.size();
  size_t idx = 0;
  while (idx < tokens_num) {
    if (token_ids[idx] != image_token_id_) {
      ++idx;
      continue;
    }
    const int32_t run_start = static_cast<int32_t>(idx);
    while (idx < tokens_num && token_ids[idx] == image_token_id_) {
      ++idx;
    }
    const int32_t run_length = static_cast<int32_t>(idx - run_start);
    int32_t consumed = 0;
    while (consumed < run_length) {
      if (item_index >= mm_items.size() ||
          consumed + expected[item_index] > run_length) {
        LOG(ERROR) << "Image token run does not match the item grid lengths.";
        return false;
      }
      close_span(run_start + consumed, expected[item_index]);
      consumed += expected[item_index];
    }
  }
  if (item_index != mm_items.size()) {
    LOG(ERROR) << "Prompt image spans (" << item_index
               << ") differ from multimodal items (" << mm_items.size() << ").";
    return false;
  }
  return true;
}

}  // namespace xllm

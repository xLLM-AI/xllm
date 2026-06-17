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

#include "processors/minicpmv_prompt_processor.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <regex>

#include "processors/minicpmv_image_processor.h"

namespace xllm::npu::model {

MiniCPMPromptProcessor::MiniCPMPromptProcessor(const ModelArgs& args) {
  image_feature_size_ = args.mm_image_feature_size();
  max_slice_nums_ = args.vision_max_slice_nums();
  slice_mode_ = args.mm_slice_mode();
  use_image_id_ = args.mm_use_image_id();
  scale_resolution_ = args.mm_scale_resolution();
}

void MiniCPMPromptProcessor::process(std::string& prompt,
                                     const MMData& mm_data) {
  std::vector<torch::Tensor> image_sizes;
  mm_data.get("image_sizes", image_sizes);

  const std::regex pattern(R"(\(<image>[\s\S]*?</image>\))");
  std::sregex_iterator image_tag_begin(prompt.begin(), prompt.end(), pattern);
  std::sregex_iterator image_tag_end;
  if (image_tag_begin == image_tag_end) {
    return;
  }

  std::vector<std::pair<int32_t, int32_t>> image_size_list;
  image_size_list.reserve(image_sizes.size());
  for (auto& image_size : image_sizes) {
    if (image_size.dim() != 1 || image_size.size(0) != 2) {
      const auto& sizes = image_size.sizes();
      LOG(FATAL) << "image_size must be a 1D tensor with 2 elements "
                    "representing height and width; now sizes: "
                 << sizes;
    }
    image_size_list.emplace_back(std::make_pair(image_size[0].item<int32_t>(),
                                                image_size[1].item<int32_t>()));
  }

  std::vector<std::string> text_chunks;
  size_t last_pos = 0;
  for (auto it = image_tag_begin; it != image_tag_end; ++it) {
    auto match = *it;
    text_chunks.push_back(prompt.substr(last_pos, match.position() - last_pos));
    last_pos = match.position() + match.length();
  }
  text_chunks.push_back(prompt.substr(last_pos));

  std::string new_prompt;
  for (size_t i = 0; i < image_size_list.size(); ++i) {
    new_prompt += text_chunks[i];
    new_prompt += get_slice_image_placeholder(image_size_list[i], i);
  }
  new_prompt += text_chunks.back();
  prompt = std::move(new_prompt);
}

void MiniCPMPromptProcessor::find_mm_spans(
    const std::vector<int32_t>& token_ids,
    MMData& mm_data) {
  int32_t global_mm_index = 0;
  int32_t offset = 0;
  auto& mm_items = mm_data.items<MMItemVec>();
  bool in_image = false;
  std::vector<int32_t> span_tokens;
  const size_t tokens_num = token_ids.size();
  for (size_t idx = 0; idx < tokens_num; ++idx) {
    int32_t token = token_ids[idx];
    if (token == im_start_id_) {
      in_image = true;
      offset = static_cast<int32_t>(idx) + 1;
      span_tokens.clear();
      continue;
    }
    if (token == im_end_id_) {
      if (!in_image) {
        continue;
      }
      in_image = false;
      int32_t length = static_cast<int32_t>(span_tokens.size());
      auto& item = mm_items[global_mm_index++];
      item.mutable_state().mutable_token_pos() = {offset, length};
      if (length == 0) {
        item.mutable_state().mutable_mm_token_num() = 0;
        continue;
      }
      std::vector<uint8_t> mask_vec;
      mask_vec.reserve(length);
      for (const auto token_id : span_tokens) {
        mask_vec.push_back(static_cast<uint8_t>(token_id == unk_id_));
      }
      auto mask =
          torch::from_blob(
              mask_vec.data(),
              {static_cast<int64_t>(length)},
              torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU))
              .clone();
      item.mutable_state().mutable_mm_token_mask() = mask;
      item.mutable_state().mutable_mm_token_num() =
          static_cast<int32_t>(mask.sum().item<int64_t>());
      continue;
    }
    if (!in_image) {
      continue;
    }
    span_tokens.push_back(token);
  }
}

std::string MiniCPMPromptProcessor::get_image_id_placeholder(
    int32_t idx) const {
  return im_id_start_ + std::to_string(idx) + im_id_end_;
}

std::string MiniCPMPromptProcessor::get_grid_placeholder(
    const std::pair<int32_t, int32_t>& grid) const {
  if (grid.first == 0 || grid.second == 0) {
    return "";
  }

  std::string slice_placeholder = slice_start_token_;
  for (int32_t i = 0; i < image_feature_size_; ++i) {
    slice_placeholder += unk_token_;
  }
  slice_placeholder += slice_end_token_;

  std::string grid_placeholder;
  for (int32_t i = 0; i < grid.second; ++i) {
    for (int32_t j = 0; j < grid.first; ++j) {
      grid_placeholder += slice_placeholder;
    }
    if (i < grid.second - 1) {
      grid_placeholder += "\n";
    }
  }

  return grid_placeholder;
}

std::string MiniCPMPromptProcessor::get_slice_image_placeholder(
    const std::pair<int32_t, int32_t>& image_size,
    int32_t image_idx,
    int32_t max_slice_nums,
    std::optional<bool> use_image_id_opt) const {
  if (max_slice_nums < 0) {
    max_slice_nums = max_slice_nums_;
  }

  bool use_image_id =
      use_image_id_opt.has_value() ? use_image_id_opt.value() : use_image_id_;

  CHECK_GE(max_slice_nums, 0);
  auto grid = MiniCPMVImageProcessor::get_sliced_grid(
      image_size, max_slice_nums, scale_resolution_);

  std::string image_placeholder = im_start_token_;
  for (int32_t i = 0; i < image_feature_size_; ++i) {
    image_placeholder += unk_token_;
  }
  image_placeholder += im_end_token_;

  std::string final_placeholder;
  if (use_image_id) {
    final_placeholder = get_image_id_placeholder(image_idx) + image_placeholder;
  } else {
    final_placeholder = image_placeholder;
  }

  if (slice_mode_) {
    final_placeholder += get_grid_placeholder(grid);
  }

  return final_placeholder;
}

}  // namespace xllm::npu::model

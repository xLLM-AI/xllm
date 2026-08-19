/* Copyright 2025-2026 The xLLM Authors.

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

#include "processors/qwen2_vl_prompt_processor.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "core/framework/tokenizer/tokenizer.h"

namespace xllm {
namespace {

int32_t count_mm_tokens(const torch::Tensor& grid_thw, int32_t merge_length) {
  if (!grid_thw.defined()) {
    return 0;
  }

  int32_t total = 0;
  const int64_t count = grid_thw.sizes()[0];
  for (int64_t idx = 0; idx < count; ++idx) {
    total += grid_thw[idx].prod().item<int32_t>() / merge_length;
  }
  return total;
}

}  // namespace

Qwen2VLPromptProcessor::Qwen2VLPromptProcessor(const ModelArgs& args) {
  merge_size_ = args.mm_image_merge_size();
  vision_start_token_id_ = args.vision_start_token_id();
  vision_end_token_id_ = args.vision_end_token_id();
  image_token_id_ = args.image_token_id();
  video_token_id_ = args.video_token_id();
}

void Qwen2VLPromptProcessor::process(std::string& /*prompt*/,
                                     const MMData& mm_data) {
  // Token-level expansion is handled in expand_mm_tokens().
  DLOG_IF(WARNING, !mm_data.empty())
      << "Qwen2VLPromptProcessor::process is unused when token-level "
         "expansion is enabled";
}

void Qwen2VLPromptProcessor::expand_mm_tokens(std::vector<int32_t>& token_ids,
                                              MMData& mm_data,
                                              const Tokenizer* /*tokenizer*/) {
  torch::Tensor image_grid_thw;
  if (auto res = mm_data.get<torch::Tensor>("image_grid_thw")) {
    image_grid_thw = res.value();
  }

  torch::Tensor video_grid_thw;
  if (auto res = mm_data.get<torch::Tensor>("video_grid_thw")) {
    video_grid_thw = res.value();
  }

  if (!image_grid_thw.defined() && !video_grid_thw.defined()) {
    return;
  }

  const int32_t merge_length = merge_size_ * merge_size_;
  const int32_t expanded_mm_tokens =
      count_mm_tokens(image_grid_thw, merge_length) +
      count_mm_tokens(video_grid_thw, merge_length);
  int32_t placeholder_count = 0;
  if (image_grid_thw.defined()) {
    placeholder_count += static_cast<int32_t>(image_grid_thw.sizes()[0]);
  }
  if (video_grid_thw.defined()) {
    placeholder_count += static_cast<int32_t>(video_grid_thw.sizes()[0]);
  }

  std::vector<int32_t> expanded;
  const int32_t extra_tokens =
      std::max(expanded_mm_tokens - placeholder_count, 0);
  expanded.reserve(token_ids.size() + static_cast<size_t>(extra_tokens));

  int32_t image_index = 0;
  int32_t video_index = 0;

  for (size_t index = 0; index < token_ids.size(); ++index) {
    if (token_ids[index] != vision_start_token_id_) {
      expanded.push_back(token_ids[index]);
      continue;
    }

    CHECK_LT(index + 2, token_ids.size())
        << "vision_start without matching placeholder and vision_end";
    CHECK_EQ(token_ids[index + 2], vision_end_token_id_)
        << "vision_start must be followed by one placeholder and vision_end";

    const int32_t placeholder_token_id = token_ids[index + 1];
    const torch::Tensor* grid_thw = nullptr;
    int32_t* modality_index = nullptr;
    if (placeholder_token_id == image_token_id_) {
      CHECK(image_grid_thw.defined());
      grid_thw = &image_grid_thw;
      modality_index = &image_index;
    } else if (placeholder_token_id == video_token_id_) {
      CHECK(video_grid_thw.defined());
      grid_thw = &video_grid_thw;
      modality_index = &video_index;
    } else {
      LOG(FATAL) << "Unexpected placeholder token id between vision bounds: "
                 << placeholder_token_id;
    }

    CHECK_LT(*modality_index, grid_thw->sizes()[0]);
    const int32_t token_num =
        (*grid_thw)[(*modality_index)].prod().item<int32_t>() / merge_length;
    CHECK_GT(token_num, 0) << "mm placeholder must expand to a positive span";

    expanded.push_back(vision_start_token_id_);
    expanded.insert(
        expanded.end(), static_cast<size_t>(token_num), placeholder_token_id);
    expanded.push_back(vision_end_token_id_);

    ++(*modality_index);
    index += 2;
  }

  token_ids = std::move(expanded);
}

void Qwen2VLPromptProcessor::find_mm_spans(
    const std::vector<int32_t>& token_ids,
    MMData& mm_data) {
  auto start = token_ids.begin();
  int32_t global_mm_index = 0;
  int32_t offset = 0;
  int32_t length = 0;
  auto& mm_items = mm_data.items<MMItemVec>();
  while (true) {
    auto vision_start_it =
        std::find(start, token_ids.end(), vision_start_token_id_);
    auto vision_end_it =
        std::find(start, token_ids.end(), vision_end_token_id_);
    if (vision_start_it == token_ids.end()) {
      break;
    }
    offset =
        static_cast<int32_t>(std::distance(token_ids.begin(), vision_start_it));
    length =
        static_cast<int32_t>(std::distance(vision_start_it + 1, vision_end_it));

    auto& item = mm_items[global_mm_index];
    const int32_t next_token_id = *(vision_start_it + 1);
    if (next_token_id == image_token_id_ || next_token_id == video_token_id_) {
      item.mutable_state().mutable_token_pos() = {offset + 1, length};
      const torch::Tensor mask = torch::ones(
          {length},
          torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
      item.mutable_state().mutable_mm_token_mask() = mask;
      item.mutable_state().mutable_mm_token_num() = length;
    }
    ++global_mm_index;
    start = std::next(vision_end_it);
  }
}

}  // namespace xllm

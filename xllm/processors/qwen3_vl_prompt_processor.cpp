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

#include "processors/qwen3_vl_prompt_processor.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cassert>
#include <cstdio>

#include "core/framework/tokenizer/tokenizer.h"

namespace xllm {
namespace {

void append_encoded_tokens(const Tokenizer* tokenizer,
                           std::vector<int32_t>& token_ids,
                           const std::string& text) {
  CHECK(tokenizer != nullptr) << "timestamp expansion requires tokenizer";
  std::vector<int32_t> encoded;
  CHECK(tokenizer->encode(text, &encoded));
  token_ids.insert(token_ids.end(), encoded.begin(), encoded.end());
}

void append_video_frames(std::vector<int32_t>& expanded,
                         const torch::Tensor& video_grid_thw,
                         int32_t video_index,
                         const std::vector<VideoMetadata>& video_metadata,
                         int32_t merge_length,
                         int32_t temporal_patch_size,
                         int32_t vision_start_token_id,
                         int32_t vision_end_token_id,
                         int32_t video_token_id,
                         const Tokenizer* tokenizer) {
  const int32_t num_frames = video_grid_thw[video_index][0].item<int32_t>();
  const int32_t token_num = video_grid_thw[video_index][1].item<int32_t>() *
                            video_grid_thw[video_index][2].item<int32_t>() /
                            merge_length;
  const auto& timestamps = video_metadata[video_index].timestamps;
  CHECK(!timestamps.empty());

  std::vector<double> ts = timestamps;
  const size_t rem = ts.size() % static_cast<size_t>(temporal_patch_size);
  if (rem != 0) {
    ts.insert(
        ts.end(), static_cast<size_t>(temporal_patch_size) - rem, ts.back());
  }
  std::vector<double> selected;
  selected.reserve(ts.size() / static_cast<size_t>(temporal_patch_size));
  for (size_t i = 0; i < ts.size();
       i += static_cast<size_t>(temporal_patch_size)) {
    selected.push_back(
        (ts[i] + ts[i + static_cast<size_t>(temporal_patch_size) - 1]) / 2.0);
  }
  if (selected.size() > static_cast<size_t>(num_frames)) {
    selected.resize(static_cast<size_t>(num_frames));
  }
  while (selected.size() < static_cast<size_t>(num_frames)) {
    selected.push_back(selected.back());
  }

  for (int32_t idx = 0; idx < num_frames; ++idx) {
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "<%.1f seconds>", selected[idx]);
    append_encoded_tokens(tokenizer, expanded, buffer);
    expanded.push_back(vision_start_token_id);
    expanded.insert(
        expanded.end(), static_cast<size_t>(token_num), video_token_id);
    expanded.push_back(vision_end_token_id);
  }
}

}  // namespace

Qwen3VLPromptProcessor::Qwen3VLPromptProcessor(const ModelArgs& args) {
  merge_size_ = args.mm_image_merge_size();
  vision_start_token_id_ = args.vision_start_token_id();
  vision_end_token_id_ = args.vision_end_token_id();
  image_token_id_ = args.image_token_id();
  video_token_id_ = args.video_token_id();
  temporal_patch_size_ = args.mm_temporal_patch_size();
}

void Qwen3VLPromptProcessor::process(std::string& /*prompt*/,
                                     const MMData& mm_data) {
  // Token-level expansion is handled in expand_mm_tokens().
  DLOG_IF(WARNING, !mm_data.empty())
      << "Qwen3VLPromptProcessor::process is unused when token-level "
         "expansion is enabled";
}

void Qwen3VLPromptProcessor::expand_mm_tokens(std::vector<int32_t>& token_ids,
                                              MMData& mm_data,
                                              const Tokenizer* tokenizer) {
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

  std::vector<VideoMetadata> video_metadata;
  mm_data.get_metadata(MMType::VIDEO, video_metadata);
  if (video_grid_thw.defined()) {
    CHECK(video_metadata.size() == static_cast<size_t>(video_grid_thw.size(0)));
  }

  const int32_t merge_length = merge_size_ * merge_size_;
  std::vector<int32_t> expanded;
  expanded.reserve(token_ids.size());

  int32_t image_index = 0;
  int32_t video_index = 0;

  for (size_t index = 0; index < token_ids.size(); ++index) {
    if (token_ids[index] == vision_start_token_id_ &&
        index + 2 < token_ids.size() &&
        token_ids[index + 2] == vision_end_token_id_) {
      const int32_t placeholder_token_id = token_ids[index + 1];
      if (placeholder_token_id == image_token_id_ && image_grid_thw.defined() &&
          image_index < image_grid_thw.size(0)) {
        const int32_t token_num =
            image_grid_thw[image_index].prod().item<int32_t>() / merge_length;
        expanded.push_back(vision_start_token_id_);
        expanded.insert(
            expanded.end(), static_cast<size_t>(token_num), image_token_id_);
        expanded.push_back(vision_end_token_id_);
        ++image_index;
        index += 2;
        continue;
      }

      if (placeholder_token_id == video_token_id_ && video_grid_thw.defined() &&
          video_index < video_grid_thw.size(0)) {
        append_video_frames(expanded,
                            video_grid_thw,
                            video_index,
                            video_metadata,
                            merge_length,
                            temporal_patch_size_,
                            vision_start_token_id_,
                            vision_end_token_id_,
                            video_token_id_,
                            tokenizer);
        ++video_index;
        index += 2;
        continue;
      }
    }

    if (token_ids[index] == image_token_id_ && image_grid_thw.defined() &&
        image_index < image_grid_thw.size(0)) {
      const bool in_vision_block =
          index >= 1 && token_ids[index - 1] == vision_start_token_id_ &&
          index + 1 < token_ids.size() &&
          token_ids[index + 1] == vision_end_token_id_;
      if (!in_vision_block) {
        const int32_t token_num =
            image_grid_thw[image_index].prod().item<int32_t>() / merge_length;
        expanded.insert(
            expanded.end(), static_cast<size_t>(token_num), image_token_id_);
        ++image_index;
        continue;
      }
    }

    if (token_ids[index] == video_token_id_ && video_grid_thw.defined() &&
        video_index < video_grid_thw.size(0)) {
      const bool in_vision_block =
          index >= 1 && token_ids[index - 1] == vision_start_token_id_ &&
          index + 1 < token_ids.size() &&
          token_ids[index + 1] == vision_end_token_id_;
      if (!in_vision_block) {
        append_video_frames(expanded,
                            video_grid_thw,
                            video_index,
                            video_metadata,
                            merge_length,
                            temporal_patch_size_,
                            vision_start_token_id_,
                            vision_end_token_id_,
                            video_token_id_,
                            tokenizer);
        ++video_index;
        continue;
      }
    }

    expanded.push_back(token_ids[index]);
  }

  token_ids = std::move(expanded);
}

void Qwen3VLPromptProcessor::find_mm_spans(
    const std::vector<int32_t>& token_ids,
    MMData& mm_data) {
  auto start = token_ids.begin();
  int32_t global_mm_index = 0;
  int32_t offset = 0;
  int32_t length = 0;
  auto& mm_items = mm_data.items<MMItemVec>();

  torch::Tensor video_grid_thw;
  if (auto res = mm_data.get<torch::Tensor>("video_grid_thw")) {
    video_grid_thw = res.value();
  }

  int32_t video_index = 0;
  int32_t video_frames_left = 0;
  int32_t video_span_start = -1;
  int32_t video_span_end = -1;
  std::vector<uint8_t> video_mask;

  while (true) {
    auto vision_start_it =
        std::find(start, token_ids.end(), vision_start_token_id_);
    if (vision_start_it == token_ids.end()) {
      break;
    }
    auto vision_end_it =
        std::find(vision_start_it + 1, token_ids.end(), vision_end_token_id_);
    CHECK(vision_end_it != token_ids.end());

    offset = std::distance(token_ids.begin(), vision_start_it);
    length = std::distance(vision_start_it + 1, vision_end_it);

    int32_t first_token = *(vision_start_it + 1);
    if (first_token == image_token_id_) {
      CHECK(global_mm_index < mm_items.size());
      auto& item = mm_items[global_mm_index];
      item.mutable_state().mutable_token_pos() = {offset + 1, length};
      item.mutable_state().mutable_mm_token_mask() = torch::ones(
          {length},
          torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
      item.mutable_state().mutable_mm_token_num() = length;
      ++global_mm_index;

    } else if (first_token == video_token_id_) {
      if (video_frames_left == 0) {
        CHECK(video_grid_thw.defined() && video_grid_thw.numel() > 0)
            << "video token exists but video_grid_thw is missing";
        CHECK(video_index < video_grid_thw.size(0));
        CHECK(global_mm_index < mm_items.size());

        video_frames_left = video_grid_thw[video_index][0].item<int32_t>();
        video_span_start = offset + 1;
        video_span_end = video_span_start;
        video_mask.clear();
      }

      CHECK(video_frames_left > 0);
      const int32_t frame_span_start = offset + 1;
      const int32_t frame_span_end = frame_span_start + length;
      if (video_span_end < frame_span_start) {
        video_mask.insert(
            video_mask.end(), frame_span_start - video_span_end, 0);
      }
      for (auto token_it = vision_start_it + 1; token_it != vision_end_it;
           ++token_it) {
        video_mask.push_back(
            static_cast<uint8_t>(*token_it == video_token_id_));
      }
      video_span_end = frame_span_end;
      --video_frames_left;
      if (video_frames_left == 0) {
        auto& item = mm_items[global_mm_index];
        item.mutable_state().mutable_token_pos() = {
            video_span_start, video_span_end - video_span_start};
        item.mutable_state().mutable_mm_token_mask() =
            torch::from_blob(
                video_mask.data(),
                {static_cast<int64_t>(video_mask.size())},
                torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU))
                .clone();
        const torch::Tensor& mask = item.state().mm_token_mask();
        item.mutable_state().mutable_mm_token_num() =
            static_cast<int32_t>(mask.sum().item<int64_t>());

        ++global_mm_index;
        ++video_index;
      }
    }

    start = std::next(vision_end_it);
  }
}

std::pair<Qwen3VLPromptProcessor::TokenType, size_t>
Qwen3VLPromptProcessor::find_vision_token(const std::string& prompt,
                                          size_t begin) {
  auto img_pos = prompt.find(image_token_, begin);
  auto vid_pos = prompt.find(video_token_, begin);

  if (img_pos == std::string::npos && vid_pos == std::string::npos)
    return {TokenType::INVALID, std::string::npos};
  else if (vid_pos == std::string::npos)
    return {TokenType::IMAGE, img_pos};
  else if (img_pos == std::string::npos)
    return {TokenType::VIDEO, vid_pos};
  else
    return img_pos < vid_pos ? std::make_pair(TokenType::IMAGE, img_pos)
                             : std::make_pair(TokenType::VIDEO, vid_pos);
}

std::vector<double> Qwen3VLPromptProcessor::build_timestamps(
    const std::vector<double>& timestamps,
    size_t num_frames,
    int32_t merge_size) {
  CHECK_GT(merge_size, 0);

  if (timestamps.empty()) {
    return std::vector<double>(num_frames, 0.0);
  }

  std::vector<double> ts = timestamps;
  const size_t rem = ts.size() % static_cast<size_t>(merge_size);
  if (rem != 0) {
    ts.insert(ts.end(), static_cast<size_t>(merge_size) - rem, ts.back());
  }

  std::vector<double> out;
  out.reserve(ts.size() / static_cast<size_t>(merge_size));

  for (size_t i = 0; i < ts.size(); i += static_cast<size_t>(merge_size)) {
    out.push_back((ts[i] + ts[i + static_cast<size_t>(merge_size) - 1]) / 2.0);
  }

  if (out.size() > num_frames) {
    out.resize(num_frames);
  }
  while (out.size() < num_frames) {
    out.push_back(out.back());
  }

  return out;
}

std::string Qwen3VLPromptProcessor::format_timestamp_str(double timestamp) {
  char buffer[32];
  snprintf(buffer, sizeof(buffer), "<%.1f seconds>", timestamp);
  return buffer;
}

}  // namespace xllm

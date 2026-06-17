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

#include "processors/qwen3_vl_prompt_processor.h"

#include <torch/torch.h>

#include <algorithm>
#include <cassert>
#include <cstdio>

namespace xllm {

Qwen3VLPromptProcessor::Qwen3VLPromptProcessor(const ModelArgs& args) {
  merge_size_ = args.mm_image_merge_size();
  vision_start_token_id_ = args.vision_start_token_id();
  vision_end_token_id_ = args.vision_end_token_id();
  image_token_id_ = args.image_token_id();
  video_token_id_ = args.video_token_id();
  temporal_patch_size_ = args.mm_temporal_patch_size();
}

void Qwen3VLPromptProcessor::process(std::string& prompt,
                                     const MMData& mm_data) {
  torch::Tensor image_grid_thw;
  if (auto res = mm_data.get<torch::Tensor>("image_grid_thw"))
    image_grid_thw = res.value();

  torch::Tensor video_grid_thw;
  if (auto res = mm_data.get<torch::Tensor>("video_grid_thw"))
    video_grid_thw = res.value();

  if (!image_grid_thw.defined() && !video_grid_thw.defined()) return;

  std::vector<VideoMetadata> video_metadata;
  mm_data.get_metadata(MMType::VIDEO, video_metadata);
  if (video_grid_thw.defined()) {
    CHECK(video_metadata.size() == static_cast<size_t>(video_grid_thw.size(0)));
  }

  const int32_t merge_length = merge_size_ * merge_size_;

  int32_t total_image_token = 0;
  if (image_grid_thw.defined()) {
    int32_t count = image_grid_thw.size(0);
    for (int32_t idx = 0; idx < count; ++idx) {
      total_image_token +=
          image_grid_thw[idx].prod().item<int32_t>() / merge_length;
    }
  }

  int32_t total_video_token = 0;
  if (video_grid_thw.defined()) {
    int32_t count = video_grid_thw.size(0);
    for (int32_t idx = 0; idx < count; ++idx) {
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

  size_t begin = 0;
  auto pair = find_vision_token(prompt, begin);

  while (pair.second != std::string::npos) {
    if (pair.first == TokenType::IMAGE) {
      data.append(prompt, begin, pair.second - begin);

      auto token_num =
          image_grid_thw[image_index].prod().item<int32_t>() / merge_length;
      while (token_num--) {
        data.append(image_token_);
      }

      ++image_index;
      begin = pair.second + image_token_.size();

    } else if (pair.first == TokenType::VIDEO) {
      const size_t pos = pair.second;
      const size_t vs_len = vision_start_token_.size();
      const size_t ve_len = vision_end_token_.size();
      const size_t vt_len = video_token_.size();

      size_t replace_begin = pos;
      size_t replace_end = pos + vt_len;

      if (pos >= vs_len &&
          prompt.compare(pos - vs_len, vs_len, vision_start_token_) == 0 &&
          prompt.compare(pos + vt_len, ve_len, vision_end_token_) == 0) {
        replace_begin = pos - vs_len;
        replace_end = pos + vt_len + ve_len;
      }

      data.append(prompt, begin, replace_begin - begin);

      const int32_t num_frames = video_grid_thw[video_index][0].item<int32_t>();
      const int32_t token_num = video_grid_thw[video_index][1].item<int32_t>() *
                                video_grid_thw[video_index][2].item<int32_t>() /
                                merge_length;

      const auto& timestamps = video_metadata[video_index].timestamps;
      CHECK(!timestamps.empty());

      auto selected = build_timestamps(
          timestamps, static_cast<size_t>(num_frames), temporal_patch_size_);

      for (int32_t idx = 0; idx < num_frames; ++idx) {
        data.append(format_timestamp_str(selected[idx]));
        data.append(vision_start_token_);
        int32_t num = token_num;

        while (num--) {
          data.append(video_token_);
        }
        data.append(vision_end_token_);
      }

      ++video_index;
      begin = replace_end;
    } else {
      LOG(FATAL) << "Unexpected token type encountered.";
    }

    pair = find_vision_token(prompt, begin);
  }

  if (begin < prompt.size()) data.append(prompt, begin, std::string::npos);
  prompt = std::move(data);
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

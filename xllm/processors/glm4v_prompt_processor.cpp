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

#include "processors/glm4v_prompt_processor.h"

#include <cstdint>
#include <cstdio>

namespace xllm {

GLM4VPromptProcessor::GLM4VPromptProcessor(const ModelArgs& args) {
  merge_size_ = args.mm_image_merge_size();
  image_start_token_id_ = args.image_start_token_id();
  image_end_token_id_ = args.image_end_token_id();
  video_start_token_id_ = args.video_start_token_id();
  video_end_token_id_ = args.video_end_token_id();
  image_token_id_ = args.image_token_id();
}

void GLM4VPromptProcessor::process(std::string& prompt, const MMData& mm_data) {
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
  if (!video_metadata.empty()) {
    CHECK(video_metadata.size() ==
          static_cast<size_t>(video_grid_thw.sizes()[0]));
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
      total_video_token += video_grid_thw[idx].prod().item<int32_t>() /
                           merge_length /
                           video_grid_thw[idx][0].item<int32_t>();
    }
  }

  size_t total_token_len = total_image_token * image_token_.size() +
                           total_video_token * image_token_.size();
  std::string data;
  data.reserve(prompt.size() + total_token_len);

  int32_t image_index = 0;
  int32_t video_index = 0;
  size_t begin = 0;
  auto pair = find_vision_token(prompt, begin);

  while (pair.second != std::string::npos) {
    data.append(prompt, begin, pair.second - begin);
    if (pair.first == TokenType::IMAGE) {
      auto token_num =
          image_grid_thw[image_index].prod().item<int32_t>() / merge_length;
      while (token_num--) {
        data.append(image_token_);
      }

      ++image_index;
      begin = pair.second + image_token_.size();
    } else if (pair.first == TokenType::VIDEO) {
      auto num_frames = video_grid_thw[video_index][0].item<int32_t>();
      auto timestamps = video_metadata[video_index].timestamps;
      CHECK(!timestamps.empty());

      auto selected = build_timestamps(timestamps, num_frames);
      auto token_num = video_grid_thw[video_index].prod().item<int32_t>() /
                       merge_length / num_frames;
      for (int32_t idx = 0; idx < num_frames; ++idx) {
        data.append(begin_of_image_token_);
        auto num = token_num;
        while (num--) {
          data.append(image_token_);
        }
        data.append(end_of_image_token_);
        data.append(format_timestamp_str(selected[idx]));
      }

      ++video_index;
      begin = pair.second + video_token_.size();
    } else {
      LOG(FATAL) << "Unexpected token type encountered.";
    }
    pair = find_vision_token(prompt, begin);
  }

  if (begin < prompt.size()) {
    data.append(prompt, begin, std::string::npos);
  }
  prompt = std::move(data);
}

void GLM4VPromptProcessor::find_mm_spans(const std::vector<int32_t>& token_ids,
                                         MMData& mm_data) {
  const size_t tokens_num = token_ids.size();
  int32_t global_mm_index = 0;
  int32_t image_span_offset = 0;
  int32_t image_span_length = 0;
  bool is_video = false;
  int32_t video_offset = 0;
  std::vector<uint8_t> video_mask;
  auto& mm_items = mm_data.items<MMItemVec>();

  for (size_t idx = 0; idx < tokens_num; ++idx) {
    auto token = token_ids[idx];
    if (token == video_start_token_id_) {
      is_video = true;
      video_offset = idx + 1;
      video_mask.clear();
      continue;
    } else if (token == video_end_token_id_) {
      if (is_video) {
        auto& item = mm_items[global_mm_index++];
        int32_t video_length = static_cast<int32_t>(video_mask.size());
        item.mutable_state().mutable_token_pos() = {video_offset, video_length};
        auto mask =
            torch::from_blob(
                video_mask.data(),
                {static_cast<int64_t>(video_length)},
                torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU))
                .clone();
        item.mutable_state().mutable_mm_token_mask() = mask;
        item.mutable_state().mutable_mm_token_num() =
            static_cast<int32_t>(mask.sum().item<int64_t>());
      }
      is_video = false;
      continue;
    }
    if (is_video) {
      video_mask.push_back(token == image_token_id_);
      continue;
    }
    if (token == image_start_token_id_) {
      image_span_offset = idx + 1;
    }
    if (token == image_token_id_) {
      ++image_span_length;
    } else if (token == image_end_token_id_) {
      auto& item = mm_items[global_mm_index++];
      item.mutable_state().mutable_token_pos() = {image_span_offset,
                                                  image_span_length};
      auto mask = torch::ones(
          {image_span_length},
          torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
      item.mutable_state().mutable_mm_token_mask() = mask;
      item.mutable_state().mutable_mm_token_num() = image_span_length;
      image_span_length = 0;
    }
  }
}

std::pair<GLM4VPromptProcessor::TokenType, size_t>
GLM4VPromptProcessor::find_vision_token(const std::string& prompt,
                                        size_t begin) {
  auto img_pos = prompt.find(image_token_, begin);
  auto vid_pos = prompt.find(video_token_, begin);

  if (img_pos == std::string::npos && vid_pos == std::string::npos) {
    return {TokenType::INVALID, std::string::npos};
  } else if (vid_pos == std::string::npos) {
    return {TokenType::IMAGE, img_pos};
  } else if (img_pos == std::string::npos) {
    return {TokenType::VIDEO, vid_pos};
  } else {
    return img_pos < vid_pos ? std::make_pair(TokenType::IMAGE, img_pos)
                             : std::make_pair(TokenType::VIDEO, vid_pos);
  }
}

std::vector<double> GLM4VPromptProcessor::build_timestamps(
    const std::vector<double>& timestamps,
    size_t num_frames) {
  std::vector<double> vec;
  vec.reserve(num_frames);

  for (size_t i = 0; i < timestamps.size(); i += 2) {
    vec.push_back(timestamps[i]);
    if (vec.size() == num_frames) break;
  }

  while (vec.size() < num_frames) {
    vec.push_back(vec.back());
  }

  return vec;
}

std::string GLM4VPromptProcessor::format_timestamp_str(double timestamp) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%.1f seconds", timestamp);
  return buffer;
}

}  // namespace xllm

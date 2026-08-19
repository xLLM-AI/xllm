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

#include "processors/glm4v_prompt_processor.h"

#include <glog/logging.h>

#include <cstdint>
#include <cstdio>

#include "core/framework/tokenizer/tokenizer.h"

namespace xllm {
namespace {

void append_encoded_tokens(const Tokenizer* tokenizer,
                           std::vector<int32_t>& token_ids,
                           const std::string& text) {
  CHECK(tokenizer != nullptr) << "video expansion requires tokenizer";
  std::vector<int32_t> encoded;
  CHECK(tokenizer->encode(text, &encoded));
  token_ids.insert(token_ids.end(), encoded.begin(), encoded.end());
}

int32_t resolve_video_placeholder_id(const Tokenizer* tokenizer,
                                     const std::string& video_token) {
  std::vector<int32_t> encoded;
  CHECK(tokenizer->encode(video_token, &encoded));
  CHECK_EQ(encoded.size(), 1U)
      << "video placeholder must tokenize to a single id";
  return encoded[0];
}

}  // namespace

GLM4VPromptProcessor::GLM4VPromptProcessor(const ModelArgs& args) {
  merge_size_ = args.mm_image_merge_size();
  image_start_token_id_ = args.image_start_token_id();
  image_end_token_id_ = args.image_end_token_id();
  video_start_token_id_ = args.video_start_token_id();
  video_end_token_id_ = args.video_end_token_id();
  image_token_id_ = args.image_token_id();
}

void GLM4VPromptProcessor::process(std::string& /*prompt*/,
                                   const MMData& mm_data) {
  // Token-level expansion is handled in expand_mm_tokens().
  DLOG_IF(WARNING, !mm_data.empty())
      << "GLM4VPromptProcessor::process is unused when token-level "
         "expansion is enabled";
}

void GLM4VPromptProcessor::expand_mm_tokens(std::vector<int32_t>& token_ids,
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
  if (!video_metadata.empty()) {
    CHECK(video_metadata.size() ==
          static_cast<size_t>(video_grid_thw.sizes()[0]));
  }

  const int32_t merge_length = merge_size_ * merge_size_;
  int32_t video_placeholder_id = 0;
  if (video_grid_thw.defined()) {
    CHECK(tokenizer != nullptr);
    video_placeholder_id =
        resolve_video_placeholder_id(tokenizer, video_token_);
  }

  std::vector<int32_t> expanded;
  expanded.reserve(token_ids.size());

  int32_t image_index = 0;
  int32_t video_index = 0;

  for (size_t index = 0; index < token_ids.size(); ++index) {
    if (token_ids[index] == image_start_token_id_ &&
        index + 2 < token_ids.size() &&
        token_ids[index + 2] == image_end_token_id_ &&
        token_ids[index + 1] == image_token_id_ && image_grid_thw.defined() &&
        image_index < image_grid_thw.size(0)) {
      const int32_t token_num =
          image_grid_thw[image_index].prod().item<int32_t>() / merge_length;
      expanded.push_back(image_start_token_id_);
      expanded.insert(
          expanded.end(), static_cast<size_t>(token_num), image_token_id_);
      expanded.push_back(image_end_token_id_);
      ++image_index;
      index += 2;
      continue;
    }

    if (video_grid_thw.defined() && video_index < video_grid_thw.size(0) &&
        token_ids[index] == video_placeholder_id) {
      const int32_t num_frames = video_grid_thw[video_index][0].item<int32_t>();
      const auto& timestamps = video_metadata[video_index].timestamps;
      CHECK(!timestamps.empty());

      const auto selected =
          build_timestamps(timestamps, static_cast<size_t>(num_frames));
      const int32_t token_num =
          video_grid_thw[video_index].prod().item<int32_t>() / merge_length /
          num_frames;
      for (int32_t idx = 0; idx < num_frames; ++idx) {
        expanded.push_back(image_start_token_id_);
        expanded.insert(
            expanded.end(), static_cast<size_t>(token_num), image_token_id_);
        expanded.push_back(image_end_token_id_);
        append_encoded_tokens(
            tokenizer, expanded, format_timestamp_str(selected[idx]));
      }
      ++video_index;
      continue;
    }

    expanded.push_back(token_ids[index]);
  }

  token_ids = std::move(expanded);
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
      video_offset = static_cast<int32_t>(idx) + 1;
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
      image_span_offset = static_cast<int32_t>(idx) + 1;
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

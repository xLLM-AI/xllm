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

#include "processors/qwen3_omni_moe_prompt_processor.h"

#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "processors/qwen3_audio_common.h"

namespace xllm {

namespace {

class ModalityCursor final {
 public:
  ModalityCursor(const std::string& token, const torch::Tensor& sizes)
      : token_(token), sizes_(sizes) {
    if (sizes_.defined()) {
      count_ = sizes_.size(0);
    }
  }

  torch::Tensor next_size() {
    CHECK(index_ < count_) << "The index of " << token_
                           << " modality is out of range, have " << count_
                           << " modality inputs but try to access index "
                           << index_;
    return sizes_[index_++];
  }

  const std::string& token() const { return token_; }

  void verify_consumed() const {
    CHECK_EQ(index_, count_) << "The number of " << token_
                             << " placeholders does not match modality inputs.";
  }

 private:
  const std::string& token_;
  const torch::Tensor& sizes_;
  int64_t index_ = 0;
  int64_t count_ = 0;
};

void append_tokens(std::string& output,
                   const std::string& token,
                   int64_t count) {
  CHECK_GE(count, 0);
  for (int64_t index = 0; index < count; ++index) {
    output.append(token);
  }
}

void set_item_span(MMDataItem& item, int32_t offset, int32_t length) {
  CHECK_GE(length, 0);
  item.mutable_state().mutable_token_pos() = {offset, length};
  item.mutable_state().mutable_mm_token_mask() = torch::ones(
      {length}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
  item.mutable_state().mutable_mm_token_num() = length;
}

}  // namespace

Qwen3OmniMoePromptProcessor::Qwen3OmniMoePromptProcessor(const ModelArgs& args)
    : vision_start_token_id_(args.vision_start_token_id()),
      vision_end_token_id_(args.vision_end_token_id()),
      image_token_id_(args.image_token_id()),
      video_token_id_(args.video_token_id()),
      audio_token_id_(args.audio_token_id()),
      audio_start_token_id_(args.audio_start_token_id()),
      audio_end_token_id_(args.audio_end_token_id()),
      merge_size_(static_cast<int32_t>(args.mm_image_merge_size())),
      position_id_per_seconds_(args.mm_position_id_per_seconds()),
      use_audio_in_video_(args.mm_use_audio_in_video()) {
  CHECK_GT(merge_size_, 0);
  CHECK_GT(args.mm_fps(), 0.0);
  video_second_per_grid_ =
      static_cast<double>(args.mm_temporal_patch_size()) / args.mm_fps();
}

void Qwen3OmniMoePromptProcessor::process(std::string& prompt,
                                          const MMData& mm_data) {
  torch::Tensor image_grid_thw;
  if (std::optional<torch::Tensor> value =
          mm_data.get<torch::Tensor>("image_grid_thw")) {
    image_grid_thw = value.value();
  }

  torch::Tensor video_grid_thw;
  if (std::optional<torch::Tensor> value =
          mm_data.get<torch::Tensor>("video_grid_thw")) {
    video_grid_thw = value.value();
  }

  torch::Tensor feature_lengths;
  if (std::optional<torch::Tensor> value =
          mm_data.get<torch::Tensor>(qwen3_audio::kFeatureLengthKey)) {
    feature_lengths = value.value();
  }

  if (!image_grid_thw.defined() && !video_grid_thw.defined() &&
      !feature_lengths.defined()) {
    return;
  }

  const int32_t merge_length = merge_size_ * merge_size_;

  int64_t total_audio_tokens = 0;
  if (feature_lengths.defined()) {
    const int64_t count = feature_lengths.size(0);
    for (int64_t index = 0; index < count; ++index) {
      total_audio_tokens += feature_lengths[index].item<int64_t>();
    }
  }

  int64_t total_image_tokens = 0;
  if (image_grid_thw.defined()) {
    const int64_t count = image_grid_thw.size(0);
    for (int64_t index = 0; index < count; ++index) {
      total_image_tokens +=
          image_grid_thw[index].prod().item<int64_t>() / merge_length;
    }
  }

  int64_t total_video_tokens = 0;
  if (video_grid_thw.defined()) {
    const int64_t count = video_grid_thw.size(0);
    for (int64_t index = 0; index < count; ++index) {
      total_video_tokens +=
          video_grid_thw[index].prod().item<int64_t>() / merge_length;
    }
  }

  const size_t reserve_size =
      prompt.size() +
      static_cast<size_t>(total_image_tokens) * image_token_.size() +
      static_cast<size_t>(total_video_tokens) * video_token_.size() +
      static_cast<size_t>(total_audio_tokens) * audio_token_.size();
  std::string output;
  output.reserve(reserve_size);

  ModalityCursor audio_cursor(audio_token_, feature_lengths);
  ModalityCursor image_cursor(image_token_, image_grid_thw);
  ModalityCursor video_cursor(video_token_, video_grid_thw);

  size_t begin = 0;
  std::pair<TokenType, size_t> special_token =
      find_special_token(prompt, begin);
  while (special_token.second != std::string::npos) {
    output.append(prompt, begin, special_token.second - begin);

    ModalityCursor* modality = nullptr;
    if (special_token.first == TokenType::AUDIO) {
      modality = &audio_cursor;
    } else if (special_token.first == TokenType::IMAGE) {
      modality = &image_cursor;
    } else if (special_token.first == TokenType::VIDEO) {
      modality = &video_cursor;
    }
    CHECK(modality != nullptr);
    const torch::Tensor modality_size = modality->next_size();
    const std::string& modality_token = modality->token();
    if (special_token.first == TokenType::AUDIO) {
      append_tokens(output, modality_token, modality_size.item<int64_t>());
    } else if (special_token.first == TokenType::VIDEO && use_audio_in_video_) {
      const torch::Tensor audio_size = audio_cursor.next_size();
      const std::string& audio_token = audio_cursor.token();
      const torch::Tensor audio_token_indices =
          torch::arange(audio_size.item<int32_t>(), torch::kInt32);

      const int32_t temporal = modality_size[0].item<int32_t>();
      const int32_t height = modality_size[1].item<int32_t>() / merge_size_;
      const int32_t width = modality_size[2].item<int32_t>() / merge_size_;
      CHECK_GT(temporal, 0);
      CHECK_GT(height, 0);
      CHECK_GT(width, 0);

      torch::Tensor video_token_indices =
          torch::arange(temporal, torch::kFloat32).view({temporal, 1, 1});
      video_token_indices =
          video_token_indices.expand({temporal, height, width}).reshape({-1});
      video_token_indices = video_token_indices * video_second_per_grid_ *
                            position_id_per_seconds_;
      auto video_indices = video_token_indices.accessor<float, 1>();
      auto audio_indices = audio_token_indices.accessor<int32_t, 1>();

      std::string placeholder = audio_start_token_;
      size_t video_index = 0;
      size_t audio_index = 0;
      const size_t video_length = video_indices.size(0);
      const size_t audio_length = audio_indices.size(0);
      while (video_index < video_length && audio_index < audio_length) {
        if (video_indices[video_index] <= audio_indices[audio_index]) {
          placeholder.append(modality_token);
          ++video_index;
        } else {
          placeholder.append(audio_token);
          ++audio_index;
        }
      }
      append_tokens(placeholder,
                    modality_token,
                    static_cast<int64_t>(video_length - video_index));
      append_tokens(placeholder,
                    audio_token,
                    static_cast<int64_t>(audio_length - audio_index));
      placeholder.append(audio_end_token_);
      output.append(placeholder);
    } else {
      append_tokens(output,
                    modality_token,
                    modality_size.prod().item<int64_t>() / merge_length);
    }

    begin = special_token.second + modality_token.size();
    special_token = find_special_token(prompt, begin);
  }

  if (begin < prompt.size()) {
    output.append(prompt, begin, std::string::npos);
  }
  audio_cursor.verify_consumed();
  image_cursor.verify_consumed();
  video_cursor.verify_consumed();
  prompt = std::move(output);
}

void Qwen3OmniMoePromptProcessor::find_mm_spans(
    const std::vector<int32_t>& token_ids,
    MMData& mm_data) {
  auto search_begin = token_ids.begin();
  int32_t global_mm_index = 0;
  MMItemVec& mm_items = mm_data.items<MMItemVec>();
  while (true) {
    auto vision_start =
        std::find(search_begin, token_ids.end(), vision_start_token_id_);
    auto vision_end =
        vision_start == token_ids.end()
            ? token_ids.end()
            : std::find(
                  vision_start + 1, token_ids.end(), vision_end_token_id_);
    auto audio_start =
        std::find(search_begin, token_ids.end(), audio_start_token_id_);
    auto audio_end =
        audio_start == token_ids.end()
            ? token_ids.end()
            : std::find(audio_start + 1, token_ids.end(), audio_end_token_id_);
    if (vision_start == token_ids.end() && audio_start == token_ids.end()) {
      break;
    }

    auto span_start = std::min(vision_start, audio_start);
    auto span_end = std::min(vision_end, audio_end);
    auto outer_span_end = std::max(vision_end, audio_end);
    CHECK(span_start != token_ids.end());
    if (span_start == vision_start) {
      CHECK(vision_end != token_ids.end());
    } else {
      CHECK(audio_end != token_ids.end());
    }
    CHECK(span_end != token_ids.end());
    CHECK(global_mm_index < static_cast<int32_t>(mm_items.size()));

    const int32_t offset =
        static_cast<int32_t>(std::distance(token_ids.begin(), span_start));
    const int32_t length =
        static_cast<int32_t>(std::distance(span_start + 1, span_end));
    MMDataItem& item = mm_items[global_mm_index];
    int32_t consumed_item_count = 1;

    if (*span_start == vision_start_token_id_ &&
        span_start + 1 != token_ids.end() &&
        *(span_start + 1) == audio_start_token_id_) {
      CHECK(item.is_type(MMType::VIDEO));
      CHECK(global_mm_index + 1 < static_cast<int32_t>(mm_items.size()));
      CHECK(mm_items[global_mm_index + 1].is_type(MMType::AUDIO));
      CHECK_GT(length, 1);
      CHECK_EQ(*(span_start + 2), video_token_id_)
          << "Audio-in-video placeholder must start with a video token.";
      set_item_span(item, offset + 2, length - 1);
      std::vector<int32_t> audio_in_video_token_ids(
          token_ids.begin() + offset + 2,
          token_ids.begin() + offset + 2 + length - 1);
      item.add(qwen3_omni_moe::kAudioInVideoTokenIdsKey,
               torch::tensor(audio_in_video_token_ids, torch::kInt32));
      set_item_span(mm_items[global_mm_index + 1], offset + 2, 0);
      consumed_item_count = 2;
      span_end = outer_span_end;
    } else {
      const int32_t first_token = *(span_start + 1);
      if (*span_start == vision_start_token_id_) {
        if (first_token == image_token_id_) {
          CHECK(item.is_type(MMType::IMAGE));
        } else {
          CHECK_EQ(first_token, video_token_id_);
          CHECK(item.is_type(MMType::VIDEO));
        }
      } else {
        CHECK(item.is_type(MMType::AUDIO));
        CHECK_EQ(first_token, audio_token_id_);
      }
      set_item_span(item, offset + 1, length);
    }
    global_mm_index += consumed_item_count;
    search_begin = std::next(span_end);
  }
  CHECK_EQ(global_mm_index, static_cast<int32_t>(mm_items.size()));
}

std::pair<Qwen3OmniMoePromptProcessor::TokenType, size_t>
Qwen3OmniMoePromptProcessor::find_special_token(const std::string& prompt,
                                                size_t begin) const {
  struct TokenInfo {
    const std::string& token;
    TokenType type;
    size_t position = std::string::npos;
  };

  std::array<TokenInfo, 3> tokens = {{{image_token_, TokenType::IMAGE},
                                      {video_token_, TokenType::VIDEO},
                                      {audio_token_, TokenType::AUDIO}}};
  for (TokenInfo& token : tokens) {
    token.position = prompt.find(token.token, begin);
  }

  auto earliest =
      std::min_element(tokens.begin(),
                       tokens.end(),
                       [](const TokenInfo& lhs, const TokenInfo& rhs) {
                         if (lhs.position == std::string::npos) {
                           return false;
                         }
                         if (rhs.position == std::string::npos) {
                           return true;
                         }
                         return lhs.position < rhs.position;
                       });
  if (earliest == tokens.end() || earliest->position == std::string::npos) {
    return {TokenType::INVALID, std::string::npos};
  }
  return {earliest->type, earliest->position};
}

}  // namespace xllm

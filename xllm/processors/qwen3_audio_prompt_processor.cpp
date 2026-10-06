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

#include "processors/qwen3_audio_prompt_processor.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#include "processors/qwen3_audio_common.h"

namespace xllm {

namespace {

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

Qwen3AudioPromptProcessor::Qwen3AudioPromptProcessor(const ModelArgs& args)
    : audio_token_id_(args.audio_token_id()),
      audio_start_token_id_(args.audio_start_token_id()),
      audio_end_token_id_(args.audio_end_token_id()) {}

void Qwen3AudioPromptProcessor::process(std::string& prompt,
                                        const MMData& mm_data) {
  torch::Tensor feature_lengths;
  if (std::optional<torch::Tensor> value =
          mm_data.get<torch::Tensor>(qwen3_audio::kFeatureLengthKey)) {
    feature_lengths = value.value();
  }
  if (!feature_lengths.defined()) {
    return;
  }

  int64_t total_audio_tokens = 0;
  const int64_t audio_count = feature_lengths.size(0);
  for (int64_t index = 0; index < audio_count; ++index) {
    total_audio_tokens += feature_lengths[index].item<int64_t>();
  }

  std::string output;
  output.reserve(prompt.size() +
                 static_cast<size_t>(total_audio_tokens) * audio_token_.size());

  size_t begin = 0;
  int64_t audio_index = 0;
  size_t audio_position = prompt.find(audio_token_, begin);
  while (audio_position != std::string::npos) {
    CHECK_LT(audio_index, audio_count)
        << "The prompt contains more audio placeholders than audio inputs.";
    output.append(prompt, begin, audio_position - begin);
    append_tokens(
        output, audio_token_, feature_lengths[audio_index].item<int64_t>());
    ++audio_index;
    begin = audio_position + audio_token_.size();
    audio_position = prompt.find(audio_token_, begin);
  }
  output.append(prompt, begin, std::string::npos);
  CHECK_EQ(audio_index, audio_count)
      << "The number of audio placeholders does not match audio inputs.";
  prompt = std::move(output);
}

void Qwen3AudioPromptProcessor::find_mm_spans(
    const std::vector<int32_t>& token_ids,
    MMData& mm_data) {
  auto search_begin = token_ids.begin();
  int32_t audio_index = 0;
  MMItemVec& mm_items = mm_data.items<MMItemVec>();
  while (true) {
    auto audio_start =
        std::find(search_begin, token_ids.end(), audio_start_token_id_);
    if (audio_start == token_ids.end()) {
      break;
    }
    auto audio_end =
        std::find(audio_start + 1, token_ids.end(), audio_end_token_id_);
    CHECK(audio_end != token_ids.end());
    CHECK(audio_start + 1 != audio_end);
    CHECK_EQ(*(audio_start + 1), audio_token_id_);
    CHECK_LT(audio_index, static_cast<int32_t>(mm_items.size()));
    CHECK(mm_items[audio_index].is_type(MMType::AUDIO));

    const int32_t offset =
        static_cast<int32_t>(std::distance(token_ids.begin(), audio_start + 1));
    const int32_t length =
        static_cast<int32_t>(std::distance(audio_start + 1, audio_end));
    set_item_span(mm_items[audio_index], offset, length);
    ++audio_index;
    search_begin = std::next(audio_end);
  }
  CHECK_EQ(audio_index, static_cast<int32_t>(mm_items.size()));
}

}  // namespace xllm

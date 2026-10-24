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

#include "processors/joyai_asr_prompt_processor.h"

#include <glog/logging.h>

#include <cstring>

#include "core/framework/config/model_config.h"

namespace xllm {

namespace {
const std::string kAudioPlaceholder = "<|AUDIO|>";
}  // namespace

JoyaiASRPromptProcessor::JoyaiASRPromptProcessor(const ModelArgs& args)
    : audio_token_(kAudioPlaceholder) {
  audio_token_id_ = args.audio_token_id();
  CHECK_NE(audio_token_id_, 0)
      << "joyai_asr requires audio_token_id (config.json \"audio_token_id\", "
         "e.g. 151647 for <|AUDIO|>)";
  ctc_pad_token_id_ = args.mm_audio_ctc_pad_token_id();
  ctc_enable_ = ModelConfig::get_instance().use_ctc();
}

bool JoyaiASRPromptProcessor::process(std::string& prompt,
                                      const MMData& mm_data) {
  torch::Tensor token_nums;
  if (auto res = mm_data.get<torch::Tensor>("audio_token_num")) {
    token_nums = res.value();
  }
  if (!token_nums.defined()) {
    return true;  // text-only request
  }
  const int64_t num_audios = token_nums.numel();

  // Checked on the UNEXPANDED prompt: a literal <|AUDIO|> typed by the
  // user is indistinguishable from generated ones after expansion.
  int64_t placeholders = 0;
  for (size_t pos = prompt.find(audio_token_); pos != std::string::npos;
       pos = prompt.find(audio_token_, pos + audio_token_.size())) {
    ++placeholders;
  }
  if (placeholders != num_audios) {
    LOG(ERROR) << "JoyaiASRPromptProcessor: " << placeholders
               << " <|AUDIO|> placeholders in prompt do not match "
               << num_audios << " audio inputs.";
    return false;
  }

  std::string data;
  data.reserve(prompt.size());

  int64_t audio_index = 0;
  size_t begin = 0;
  while (true) {
    const size_t pos = prompt.find(audio_token_, begin);
    if (pos == std::string::npos) {
      break;
    }
    const int64_t token_num = token_nums[audio_index].item<int64_t>();
    data.append(prompt, begin, pos - begin);
    for (int64_t i = 0; i < token_num; ++i) {
      data.append(audio_token_);
    }
    ++audio_index;
    begin = pos + audio_token_.size();
  }
  if (begin < prompt.size()) {
    data.append(prompt, begin, std::string::npos);
  }

  prompt = std::move(data);
  return true;
}

bool JoyaiASRPromptProcessor::find_mm_spans(
    const std::vector<int32_t>& token_ids,
    MMData& mm_data) {
  auto& mm_items = mm_data.items<MMItemVec>();

  // No wrapper tokens: one maximal audio-token run can cover several
  // audios — consume items from the run by their expected length.
  std::vector<int64_t> assigned_pad_len(mm_items.size(), 0);

  size_t item_index = 0;
  size_t i = 0;
  while (i < token_ids.size()) {
    if (token_ids[i] != audio_token_id_) {
      ++i;
      continue;
    }
    const size_t run_start = i;
    while (i < token_ids.size() && token_ids[i] == audio_token_id_) {
      ++i;
    }
    const size_t run_end = i;

    size_t pos = run_start;
    while (pos < run_end) {
      if (item_index >= mm_items.size()) {
        LOG(ERROR) << "JoyaiASRPromptProcessor: audio tokens in prompt do "
                      "not match audio inputs ("
                   << mm_items.size() << "); more tokens than inputs.";
        return false;
      }
      auto& item = mm_items[item_index];
      const auto token_num = item.get<torch::Tensor>("audio_token_num");
      if (!token_num.has_value() || token_num.value().numel() != 1) {
        LOG(ERROR) << "JoyaiASRPromptProcessor: audio input " << item_index
                   << " is missing its audio_token_num.";
        return false;
      }
      const int64_t length = token_num.value().item<int64_t>();
      if (length < 1 || static_cast<int64_t>(run_end - pos) < length) {
        LOG(ERROR) << "JoyaiASRPromptProcessor: audio token run does not "
                      "match audio input "
                   << item_index << " (run has " << (run_end - pos)
                   << " tokens left, audio_token_num is " << length << ").";
        return false;
      }
      item.mutable_state().mutable_token_pos() = {static_cast<int32_t>(pos),
                                                  static_cast<int32_t>(length)};
      item.mutable_state().mutable_mm_token_mask() = torch::ones(
          {length},
          torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
      item.mutable_state().mutable_mm_token_num() =
          static_cast<int32_t>(length);
      pos += static_cast<size_t>(length);
      ++item_index;
    }
  }

  // Fold each item's first trailing ctc-pad run (bounded by the next item)
  // into its entry: the span covers audio + text + pads, the mask marks
  // the audio and pad rows, and mm_token_num counts both. No run found ->
  // plain audio; CTC off -> the pad token (also the BOS id) stays ordinary
  // text.
  if (ctc_enable_) {
    for (size_t k = 0; k < item_index; ++k) {
      auto& state = mm_items[k].mutable_state();
      const auto audio_span = state.token_pos();
      const int32_t search_begin = audio_span.offset + audio_span.length;
      const int32_t search_end =
          k + 1 < item_index ? mm_items[k + 1].state().token_pos().offset
                             : static_cast<int32_t>(token_ids.size());
      for (int32_t j = search_begin; j < search_end; ++j) {
        if (token_ids[j] != ctc_pad_token_id_) {
          continue;
        }
        int32_t pad_length = 0;
        while (j + pad_length < search_end &&
               token_ids[j + pad_length] == ctc_pad_token_id_) {
          ++pad_length;
        }
        state.mutable_token_pos().length = j + pad_length - audio_span.offset;
        state.mutable_mm_token_mask() = torch::ones(
            {state.token_pos().length}, state.mm_token_mask().options());
        state.mutable_mm_token_mask()
            .narrow(0, audio_span.length, j - search_begin)
            .fill_(false);
        state.mutable_mm_token_num() = audio_span.length + pad_length;
        assigned_pad_len[k] = pad_length;
        break;
      }
    }
  }

  if (item_index != mm_items.size()) {
    LOG(ERROR) << "JoyaiASRPromptProcessor: assigned " << item_index
               << " audio spans for " << mm_items.size() << " audio inputs.";
    return false;
  }

  // One [hash, ctc_pad_num] assembly row per audio item — all of them
  // (pad-less entries included) so the batch aggregation keeps item order.
  // add() is insert-only: a re-parse would silently keep the stale row, so
  // fail loudly instead.
  for (size_t i = 0; i < mm_items.size(); ++i) {
    auto& item = mm_items[i];
    if (item.type() != MMType::AUDIO) {
      continue;
    }
    CHECK(!item.get<torch::Tensor>("audio_encode_meta").has_value())
        << "audio item " << i << " already carries audio_encode_meta; "
        << "find_mm_spans must run exactly once per item";
    int64_t hash_lo = 0;
    std::memcpy(&hash_lo, item.state().schedule_data().key.data, 8);
    if (hash_lo == 0) {
      LOG(ERROR) << "audio item " << i << " without a content-hash key";
      return false;
    }
    item.add("audio_encode_meta",
             torch::tensor({{hash_lo, assigned_pad_len[i]}}, torch::kInt64));
  }
  return true;
}

}  // namespace xllm

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

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/model/model_args.h"
#include "processors/prompt_processor.h"

namespace xllm {

namespace qwen3_omni_moe {
inline constexpr char kAudioInVideoTokenIdsKey[] = "audio_in_video_token_ids";
}  // namespace qwen3_omni_moe

class Qwen3OmniMoePromptProcessor final : public PromptProcessor {
 public:
  explicit Qwen3OmniMoePromptProcessor(const ModelArgs& args);

  void process(std::string& prompt, const MMData& mm_data) override;
  void find_mm_spans(const std::vector<int32_t>& token_ids,
                     MMData& mm_data) override;

 private:
  enum class TokenType { INVALID, IMAGE, VIDEO, AUDIO };

  std::pair<TokenType, size_t> find_special_token(const std::string& prompt,
                                                  size_t begin) const;

  const std::string image_token_ = "<|image_pad|>";
  const std::string video_token_ = "<|video_pad|>";
  const std::string audio_token_ = "<|audio_pad|>";
  const std::string audio_start_token_ = "<|audio_start|>";
  const std::string audio_end_token_ = "<|audio_end|>";

  int32_t vision_start_token_id_ = 0;
  int32_t vision_end_token_id_ = 0;
  int32_t image_token_id_ = 0;
  int32_t video_token_id_ = 0;
  int32_t audio_token_id_ = 0;
  int32_t audio_start_token_id_ = 0;
  int32_t audio_end_token_id_ = 0;
  int32_t merge_size_ = 0;
  int32_t position_id_per_seconds_ = 0;
  bool use_audio_in_video_ = false;
  double video_second_per_grid_ = 0.0;
};

}  // namespace xllm

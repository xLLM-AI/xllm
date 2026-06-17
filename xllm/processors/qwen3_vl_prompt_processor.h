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

class Qwen3VLPromptProcessor final : public PromptProcessor {
  enum class TokenType {
    INVALID,
    IMAGE,
    VIDEO,
  };

 public:
  explicit Qwen3VLPromptProcessor(const ModelArgs& args);

  void process(std::string& prompt, const MMData& mm_data) override;
  void find_mm_spans(const std::vector<int32_t>& token_ids,
                     MMData& mm_data) override;

 private:
  std::pair<TokenType, size_t> find_vision_token(const std::string& prompt,
                                                 size_t begin);

  std::vector<double> build_timestamps(const std::vector<double>& timestamps,
                                       size_t num_frames,
                                       int32_t merge_size);
  std::string format_timestamp_str(double timestamp);

  const std::string image_token_ = "<|image_pad|>";
  const std::string video_token_ = "<|video_pad|>";
  const std::string vision_start_token_ = "<|vision_start|>";
  const std::string vision_end_token_ = "<|vision_end|>";
  int32_t vision_start_token_id_;
  int32_t vision_end_token_id_;
  int32_t image_token_id_;
  int32_t video_token_id_;
  int32_t merge_size_ = 0;
  int32_t temporal_patch_size_ = 0;
};

}  // namespace xllm

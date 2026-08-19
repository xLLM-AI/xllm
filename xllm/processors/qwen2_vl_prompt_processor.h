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

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/framework/model/model_args.h"
#include "processors/prompt_processor.h"

namespace xllm {

class Qwen2VLPromptProcessor final : public PromptProcessor {
 public:
  explicit Qwen2VLPromptProcessor(const ModelArgs& args);

  void process(std::string& prompt, const MMData& mm_data) override;
  void find_mm_spans(const std::vector<int32_t>& token_ids,
                     MMData& mm_data) override;
  bool uses_token_level_expansion() const override {
    return !force_legacy_mm_expansion();
  }
  void expand_mm_tokens(std::vector<int32_t>& token_ids,
                        MMData& mm_data,
                        const Tokenizer* tokenizer = nullptr) override;

 private:
  int32_t vision_start_token_id_;
  int32_t vision_end_token_id_;
  int32_t image_token_id_;
  int32_t video_token_id_;
  int32_t merge_size_ = 0;
};

}  // namespace xllm

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

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "processors/prompt_processor.h"

namespace xllm {

class ModelArgs;

class TextPromptProcessor final : public PromptProcessor {
 public:
  explicit TextPromptProcessor(const ModelArgs& /*model_args*/) {}

  bool process(std::string& /*prompt*/, const MMData& mm_data) override {
    return mm_data.empty();
  }

  bool find_mm_spans(const std::vector<int32_t>& /*token_ids*/,
                     MMData& mm_data) override {
    return mm_data.empty();
  }
};

}  // namespace xllm

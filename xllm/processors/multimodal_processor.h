/* Copyright 2025-2026 The xLLM Authors.

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
#include <memory>
#include <string>
#include <vector>

#include "core/framework/tokenizer/tokenizer_args.h"

namespace xllm {
class ModelArgs;
class Tokenizer;
class MMInput;
class MMData;

class MultimodalProcessorBase {
 public:
  virtual ~MultimodalProcessorBase() = default;
  virtual bool process_prompt(std::string& prompt,
                              MMData& mm_data,
                              std::vector<int32_t>& token_ids) = 0;
  virtual bool process_multimodal(const MMInput& inputs,
                                  MMData& data) const = 0;
};

std::unique_ptr<MultimodalProcessorBase> create_multimodal_processor(
    const ModelArgs& model_args,
    std::shared_ptr<Tokenizer> tokenizer,
    int64_t max_cache_items,
    const TokenizerArgs& tokenizer_args,
    // Nonpositive lengths disable left padding.
    int32_t max_sequence_length = 0);

}  // namespace xllm

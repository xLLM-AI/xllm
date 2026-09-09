/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "core/framework/tokenizer/tokenizer.h"
#include "core/util/slice.h"

namespace xllm {

// a stateful decoder that can decode tokens incrementally.
class IncrementalDecoder final {
 public:
  IncrementalDecoder(const std::string_view& prompt,
                     size_t num_prompt_tokens,
                     bool echo,
                     bool skip_special_tokens);

  // decode the token ids incrementally
  // return the decoded delta text since last call.
  std::string decode(const Slice<int32_t>& token_ids,
                     const Tokenizer& tokenizer);

  // get the offset of the output text
  size_t output_offset() const { return output_offset_; }

  // get the offset of the prefix text
  size_t prefix_offset() const { return prefix_offset_; }

  // Enable checking whether to skip the prefill token
  void enable_checking_prefill_token() { checking_prefill_token_ = true; }

 private:
  // the original prompt string, used to skip the prompt decoding when streaming
  std::string_view prompt_;

  // the length of the prompt tokens
  size_t num_prompt_tokens_ = 0;

  // whether to skip special tokens when decoding
  bool skip_special_tokens_ = true;

  // variables to keep track of output text, should be accessed by single thread
  // prefix offset is used to defeat cleanup algorithms in the decode which
  // decide to add a space or not based on surrounding tokens.
  size_t prefix_offset_ = 0;
  // all tokens before output_offset_ have been decoded
  size_t output_offset_ = 0;

  // whether to check skipping prefill token in decode instance.
  bool checking_prefill_token_ = false;
};

}  // namespace xllm

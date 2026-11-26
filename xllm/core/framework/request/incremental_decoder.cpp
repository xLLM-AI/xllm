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

#include "incremental_decoder.h"

#include <absl/strings/match.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace xllm {

IncrementalDecoder::IncrementalDecoder(const std::string_view& prompt,
                                       size_t num_prompt_tokens,
                                       bool echo,
                                       bool skip_special_tokens)
    : prompt_(prompt),
      num_prompt_tokens_(num_prompt_tokens),
      skip_special_tokens_(skip_special_tokens) {
  // if echo is true, set prefix_offset_ and output_offset_ to 0 to print the
  // whole sequence, otherwise set them to the length of the prompt to skip the
  // prompt.
  prefix_offset_ = echo ? 0 : num_prompt_tokens_;
  output_offset_ = echo ? 0 : num_prompt_tokens_;
}

std::string IncrementalDecoder::decode(const Slice<int32_t>& token_ids,
                                       const Tokenizer& tokenizer) {
  // Delayed callbacks can call this method again before the sequence has
  // changed.  Repeating the tokenizer work is especially costly when a
  // byte-fallback token is waiting for its continuation.  Compare the exact
  // suffix that is relevant after the current offsets; earlier token changes
  // cannot affect this decode call.
  if (last_decode_valid_ && last_tokenizer_ == &tokenizer &&
      token_ids.size() >= prefix_offset_) {
    const auto current_ids = token_ids.slice(prefix_offset_);
    if (current_ids == last_token_ids_) {
      return {};
    }
  }

  std::string result;
  // return prompt directly if prompt string is not empty
  if (output_offset_ < num_prompt_tokens_ && !prompt_.empty()) {
    // leave 6 tokens for the prefix to defeat cleanup algorithms in decode
    // which decide to add a space or not depending on the surrounding ids.
    prefix_offset_ = num_prompt_tokens_ <= 6 ? 0 : num_prompt_tokens_ - 6;
    output_offset_ = num_prompt_tokens_;
    result.append(prompt_);
  }

  const auto prefix_text = tokenizer.decode(
      token_ids.slice(prefix_offset_, output_offset_), skip_special_tokens_);
  const auto new_text =
      tokenizer.decode(token_ids.slice(prefix_offset_), skip_special_tokens_);
  // utf-8 char � at the end means it is a potential unfinished byte sequence
  // from byte fallback tokenization.
  if (new_text.size() > prefix_text.size() && !absl::EndsWith(new_text, "�")) {
    prefix_offset_ = output_offset_;
    output_offset_ = token_ids.size();
    // only print the delta text
    result.append(new_text, prefix_text.size(), std::string::npos);
  }

  last_tokenizer_ = &tokenizer;
  last_token_ids_.clear();
  if (token_ids.size() > prefix_offset_) {
    last_token_ids_.assign(token_ids.begin() + prefix_offset_, token_ids.end());
  }
  last_decode_valid_ = true;
  return result;
}

}  // namespace xllm

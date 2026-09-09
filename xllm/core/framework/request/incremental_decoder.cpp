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
#include <sstream>
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
  std::stringstream ss;
  // return prompt directly if prompt string is not empty
  if (output_offset_ < num_prompt_tokens_ && !prompt_.empty()) {
    // leave 6 tokens for the prefix to defeat cleanup algorithms in decode
    // which decide to add a space or not depending on the surrounding ids.
    prefix_offset_ = num_prompt_tokens_ <= 6 ? 0 : num_prompt_tokens_ - 6;
    output_offset_ = num_prompt_tokens_;
    ss << prompt_;
  }

  // In PD mode, if a prefill token can directly generate characters, the decode
  // phase needs to skip that token. If it cannot, the decode token and that
  // token need to generate characters together.
  if (checking_prefill_token_) {
    const auto prefill_token_text =
        tokenizer.decode(token_ids.slice(output_offset_, output_offset_ + 1),
                         skip_special_tokens_);
    if (!absl::EndsWith(prefill_token_text, "�")) {
      output_offset_ += 1;
    }
    checking_prefill_token_ = false;
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
    ss << new_text.substr(prefix_text.size());
  }
  return ss.str();
}

}  // namespace xllm

/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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
#include <absl/strings/escaping.h>
#include <absl/strings/str_join.h>

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "core/common/macros.h"

namespace xllm {

using SpecialToken = std::pair<std::string, int32_t>;

struct TokenizerArgs {
  // Type of tokenizer to use. valid values are "fast", "sentencepiece" and
  // "tiktoken".
  PROPERTY(std::string, tokenizer_type) = "sentencepiece";

  // Vocab file name.
  PROPERTY(std::string, vocab_file) = "tokenizer.model";

  // Special tokens to add to the vocabulary.
  PROPERTY(std::vector<SpecialToken>, special_tokens);

  // Special tokens that should still be rendered even when
  // skip_special_tokens is enabled during decoding.
  PROPERTY(std::vector<std::string>, visible_special_tokens);

  // Regex pattern used by tiktok tokenizer only.
  PROPERTY(std::string, pattern);

  // tokens to add to the beginning of the input sequence.
  PROPERTY(std::vector<std::string>, prefix_tokens);

  // chat template
  PROPERTY(std::string, chat_template);

  // add_bos_token
  PROPERTY(bool, add_bos_token) = false;

  // add_eos_token
  PROPERTY(bool, add_eos_token) = false;

  // bos_token
  PROPERTY(std::string, bos_token);

  // eos_token
  PROPERTY(std::string, eos_token);

  // pad_token
  PROPERTY(std::string, pad_token);

  // tokenizer_class
  PROPERTY(std::string, tokenizer_class);
};

inline std::ostream& operator<<(std::ostream& os, const TokenizerArgs& args) {
  os << "TokenizerArgs: [";
  os << "tokenizer_type: " << args.tokenizer_type();
  //  os << ", chat_template: " << args.chat_template();
  os << ", add_bos_token: " << args.add_bos_token();
  os << ", add_eos_token: " << args.add_eos_token();
  os << ", bos_token: " << args.bos_token();
  os << ", eos_token: " << args.eos_token();
  os << ", pad_token: " << args.pad_token();
  os << ", tokenizer_class: " << args.tokenizer_class();
  if (!args.special_tokens().empty()) {
    os << ", special_tokens: [";
    for (const auto& [token, id] : args.special_tokens()) {
      os << "(" << token << ", " << id << ") ";
    }
    os << "]";
  }
  if (!args.visible_special_tokens().empty()) {
    os << ", visible_special_tokens: ["
       << absl::StrJoin(args.visible_special_tokens(), ", ") << "]";
  }
  os << ", pattern: " << absl::CEscape(args.pattern());
  if (!args.prefix_tokens().empty()) {
    os << ", prefix_tokens: [" << absl::StrJoin(args.prefix_tokens(), ", ")
       << "]";
  }
  os << "]";
  return os;
}

}  // namespace xllm

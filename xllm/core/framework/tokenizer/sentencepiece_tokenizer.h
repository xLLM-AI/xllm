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
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <re2/re2.h>

#include <cstdint>

#include "sentencepiece/sentencepiece_processor.h"
#include "tokenizer.h"
#include "tokenizer_args.h"

namespace xllm {

// a tokenizer that uses google/SentencePiece
class SentencePieceTokenizer : public Tokenizer {
 public:
  SentencePieceTokenizer(const std::string_view& dir_path,
                         const TokenizerArgs& args);

  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids,
              bool add_special_tokens = true) const override;

  std::string decode(const Slice<int32_t>& ids,
                     bool skip_special_tokens) const override;

  std::optional<int32_t> token_to_id(
      const std::string_view& token) const override;

  std::string id_to_token(int32_t id) const override;

  size_t vocab_size() const override;

  std::unique_ptr<Tokenizer> clone() const override;

 private:
  void load_special_tokens(const std::vector<SpecialToken>& special_tokens);
  void load_visible_special_tokens(
      const std::vector<std::string>& visible_special_tokens);

  bool encode_internal(const std::string_view& text,
                       std::vector<int32_t>* ids) const;
  void decode_internal(const Slice<int32_t>& ids,
                       size_t start,
                       size_t end,
                       std::stringstream* ss) const;

  std::string dir_path_;

  TokenizerArgs args_;

  sentencepiece::SentencePieceProcessor sp_processor_;

  // special tokens to ids
  absl::flat_hash_map<std::string, int32_t> special_token_encoder_;

  // special token ids to tokens
  absl::flat_hash_map<int32_t, std::string> special_token_decoder_;

  // special tokens that should be preserved in decoded output
  absl::flat_hash_set<int32_t> visible_special_token_ids_;

  // special token regex (optional)
  std::unique_ptr<re2::RE2> special_token_regex_;

  // token ids to add to the beginning of the input sequence
  std::vector<int32_t> prefix_token_ids_;
};

}  // namespace xllm

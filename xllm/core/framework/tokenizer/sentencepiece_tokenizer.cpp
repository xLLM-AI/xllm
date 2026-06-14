/* Copyright 2025-2026 The xLLM Authors.
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

#include "sentencepiece_tokenizer.h"

#include <absl/strings/str_cat.h>
#include <absl/strings/str_join.h>
#include <absl/strings/string_view.h>
#include <glog/logging.h>
#include <re2/re2.h>

#include <string_view>

#include "sentencepiece.pb.h"
#include "sentencepiece/sentencepiece_processor.h"

#define RETURN_FALSE_IF_ERROR(expr)  \
  do {                               \
    const auto _status = expr;       \
    if (!_status.ok()) return false; \
  } while (0)

#define RETURN_IF_ERROR(expr)  \
  do {                         \
    const auto _status = expr; \
    if (!_status.ok()) return; \
  } while (0)

namespace xllm {

SentencePieceTokenizer::SentencePieceTokenizer(const std::string_view& dir_path,
                                               const TokenizerArgs& args)
    : dir_path_(dir_path), args_(args) {
  const std::string vocab_file_path =
      dir_path.empty() ? args.vocab_file()
                       : absl::StrCat(dir_path_, "/", args.vocab_file());
  const auto status = sp_processor_.Load(vocab_file_path);
  if (!status.ok()) {
    LOG(FATAL) << "Failed to load SentencePiece model from " << vocab_file_path
               << ": " << status.ToString() << ", error " << status.ToString();
  }

  // add special tokens and construct special token regex
  if (!args.special_tokens().empty()) {
    load_special_tokens(args.special_tokens());
  }

  if (!args.visible_special_tokens().empty()) {
    load_visible_special_tokens(args.visible_special_tokens());
  }

  // construct prefix tokens
  if (!args.prefix_tokens().empty()) {
    for (const auto& token : args.prefix_tokens()) {
      if (token.empty()) {
        continue;
      }
      const auto token_id = token_to_id(token);
      if (token_id.has_value()) {
        prefix_token_ids_.push_back(token_id.value());
        LOG(INFO) << "Prefix token: " << token << ", id: " << token_id.value();
      } else {
        LOG(ERROR) << "Failed to find prefix token: " << token;
      }
    }
  }
}

void SentencePieceTokenizer::load_special_tokens(
    const std::vector<SpecialToken>& special_tokens) {
  // for each special token, add to encoder and decoder
  for (const auto& [token, id] : special_tokens) {
    if (token.empty()) {
      continue;
    }

    if (!special_token_encoder_.try_emplace(token, id).second) {
      LOG(WARNING) << "Duplicate special token: " << token << ", id: " << id;
    }

    if (!special_token_decoder_.try_emplace(id, token).second) {
      LOG(WARNING) << "Duplicate special token: " << token << ", id: " << id;
    }
  }

  // build special token regex
  std::vector<std::string> escaped_tokens;
  escaped_tokens.reserve(special_tokens.size());
  for (const auto& [token, id] : special_tokens) {
    if (token.empty()) {
      continue;
    }
    // escape each token
    const auto escaped_token = re2::RE2::QuoteMeta(token);
    escaped_tokens.push_back(escaped_token);
  }
  if (!escaped_tokens.empty()) {
    const auto special_token_regex_str = absl::StrJoin(escaped_tokens, "|");
    // surround with () to match special tokens
    const auto regex_str = absl::StrCat("(", special_token_regex_str, ")");
    special_token_regex_ = std::make_unique<re2::RE2>(regex_str);
  }
}

void SentencePieceTokenizer::load_visible_special_tokens(
    const std::vector<std::string>& visible_special_tokens) {
  for (const auto& token : visible_special_tokens) {
    if (token.empty()) {
      continue;
    }

    const auto token_id = token_to_id(token);
    if (!token_id.has_value()) {
      LOG(WARNING) << "Failed to find visible special token: " << token;
      continue;
    }
    visible_special_token_ids_.insert(token_id.value());
  }
}

bool SentencePieceTokenizer::encode_internal(const std::string_view& text,
                                             std::vector<int32_t>* ids) const {
  if (text.empty()) {
    // empty text, just return
    return true;
  }

  sentencepiece::SentencePieceText spt;
  RETURN_FALSE_IF_ERROR(sp_processor_.Encode({text.data(), text.size()}, &spt));
  for (const auto& sp : spt.pieces()) {
    ids->emplace_back(sp.id());
  }
  return true;
}

bool SentencePieceTokenizer::encode(const std::string_view& text,
                                    std::vector<int32_t>* ids,
                                    bool add_special_tokens,
                                    int32_t max_sequence_length) const {
  // prepend prefix tokens if exists
  if (!prefix_token_ids_.empty()) {
    ids->insert(
        ids->begin(), prefix_token_ids_.begin(), prefix_token_ids_.end());
  }

  if (special_token_regex_ == nullptr) {
    return encode_internal(text, ids);
  }

  absl::string_view input{text.data(), text.size()};
  absl::string_view special;
  while (true) {
    const auto* start = input.begin();
    if (!re2::RE2::FindAndConsume(&input, *special_token_regex_, &special)) {
      // no more special tokens
      break;
    }

    // encode text before special token if exists
    const std::string_view sub_input(start,
                                     input.begin() - start - special.size());
    if (!encode_internal(sub_input, ids)) {
      return false;
    }

    // add special token id if exists
    const auto sit = special_token_encoder_.find(special);
    if (sit != special_token_encoder_.end()) {
      // find one special token
      ids->push_back(sit->second);
    }
  }

  // encode remaining text if exists
  return encode_internal({input.data(), input.size()}, ids);
}

void SentencePieceTokenizer::decode_internal(const Slice<int32_t>& ids,
                                             size_t start,
                                             size_t end,
                                             std::stringstream* ss) const {
  if (start >= end) {
    // no text to decode
    return;
  }

  sentencepiece::SentencePieceText spt;
  std::vector<std::string> pieces;
  const int num_pieces = sp_processor_.GetPieceSize();
  pieces.reserve(end - start);
  for (size_t i = start; i < end; ++i) {
    const auto id = ids[i];
    if (id < 0 || id >= num_pieces) {
      LOG(ERROR) << "Invalid id: " << id;
      continue;
    }
    pieces.emplace_back(sp_processor_.IdToPiece(id));
  }
  RETURN_IF_ERROR(sp_processor_.Decode(pieces, &spt));
  (*ss) << spt.text();
}

std::string SentencePieceTokenizer::decode(const Slice<int32_t>& ids,
                                           bool skip_special_tokens) const {
  std::stringstream ss;
  size_t start = 0;
  for (size_t i = 0; i < ids.size(); ++i) {
    // identify special token
    const auto sit = special_token_decoder_.find(ids[i]);
    if (sit == special_token_decoder_.end()) {
      continue;
    }
    // decode text before special token if exists
    decode_internal(ids, start, i, &ss);

    if (!skip_special_tokens || visible_special_token_ids_.contains(ids[i])) {
      // output special token
      ss << sit->second;
    }
    start = i + 1;
  }

  // decode remaining text if exists
  decode_internal(ids, start, ids.size(), &ss);
  return ss.str();
}

std::optional<int32_t> SentencePieceTokenizer::token_to_id(
    const std::string_view& token) const {
  // encode special token
  const absl::string_view token_view{token.data(), token.size()};
  const auto sit = special_token_encoder_.find(token_view);
  if (sit != special_token_encoder_.end()) {
    return sit->second;
  }

  // encode token
  const auto token_id = sp_processor_.PieceToId(token_view);
  if (sp_processor_.IsUnknown(token_id)) {
    LOG(ERROR) << "Failed to find id for token: " << token;
    return std::nullopt;
  }
  return token_id;
}

std::string SentencePieceTokenizer::id_to_token(int32_t id) const {
  // decode special token
  const auto sit = special_token_decoder_.find(id);
  if (sit != special_token_decoder_.end()) {
    return sit->second;
  }

  // decode token
  return sp_processor_.IdToPiece(id);
}

size_t SentencePieceTokenizer::vocab_size() const {
  // vocab size = sentencepiece vocab size + special tokens
  return sp_processor_.GetPieceSize() + args_.special_tokens().size();
}

std::unique_ptr<Tokenizer> SentencePieceTokenizer::clone() const {
  return std::make_unique<SentencePieceTokenizer>(dir_path_, args_);
}

}  // namespace xllm

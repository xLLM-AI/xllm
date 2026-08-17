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

#include "fast_tokenizer.h"

#include <glog/logging.h>

#include <cstdint>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <string_view>

namespace xllm {

namespace {

bool contains_byte_level_decoder(const nlohmann::json& decoder) {
  if (decoder.is_array()) {
    for (const nlohmann::json& child : decoder) {
      if (contains_byte_level_decoder(child)) {
        return true;
      }
    }
    return false;
  }
  if (!decoder.is_object()) {
    return false;
  }

  const std::string type = decoder.value("type", "");
  if (type == "ByteLevel") {
    return true;
  }
  const auto decoders = decoder.find("decoders");
  return decoders != decoder.end() && contains_byte_level_decoder(*decoders);
}

bool has_byte_level_decoder(const std::string& tokenizer_path) {
  std::ifstream tokenizer_file(tokenizer_path);
  if (!tokenizer_file.is_open()) {
    return false;
  }

  const nlohmann::json tokenizer_json =
      nlohmann::json::parse(tokenizer_file, nullptr, false);
  if (tokenizer_json.is_discarded()) {
    return false;
  }
  const auto decoder = tokenizer_json.find("decoder");
  return decoder != tokenizer_json.end() &&
         contains_byte_level_decoder(*decoder);
}

bool is_byte_level_direct_byte(uint32_t byte) {
  return (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) ||
         (byte >= 174 && byte <= 255);
}

std::optional<uint8_t> byte_level_decode_codepoint(uint32_t codepoint) {
  if (codepoint <= 255 &&
      is_byte_level_direct_byte(static_cast<uint32_t>(codepoint))) {
    return static_cast<uint8_t>(codepoint);
  }

  uint32_t missing_index = 0;
  for (uint32_t byte = 0; byte < 256; ++byte) {
    if (is_byte_level_direct_byte(byte)) {
      continue;
    }
    if (codepoint == 256 + missing_index) {
      return static_cast<uint8_t>(byte);
    }
    ++missing_index;
  }
  return std::nullopt;
}

bool decode_utf8_codepoint(std::string_view input,
                           size_t* offset,
                           uint32_t* codepoint) {
  const size_t start = *offset;
  if (start >= input.size()) {
    return false;
  }

  const uint8_t first = static_cast<uint8_t>(input[start]);
  size_t length = 1;
  uint32_t value = first;
  if ((first & 0x80U) == 0) {
    length = 1;
  } else if ((first & 0xE0U) == 0xC0U) {
    length = 2;
    value = first & 0x1FU;
  } else if ((first & 0xF0U) == 0xE0U) {
    length = 3;
    value = first & 0x0FU;
  } else if ((first & 0xF8U) == 0xF0U) {
    length = 4;
    value = first & 0x07U;
  } else {
    return false;
  }

  if (start + length > input.size()) {
    return false;
  }
  for (size_t index = 1; index < length; ++index) {
    const uint8_t continuation = static_cast<uint8_t>(input[start + index]);
    if ((continuation & 0xC0U) != 0x80U) {
      return false;
    }
    value = (value << 6) | (continuation & 0x3FU);
  }

  *offset = start + length;
  *codepoint = value;
  return true;
}

std::string decode_byte_level_piece(std::string_view token) {
  std::string piece;
  size_t offset = 0;
  while (offset < token.size()) {
    const size_t start = offset;
    uint32_t codepoint = 0;
    if (!decode_utf8_codepoint(token, &offset, &codepoint)) {
      piece.push_back(token[start]);
      offset = start + 1;
      continue;
    }

    const std::optional<uint8_t> byte = byte_level_decode_codepoint(codepoint);
    if (byte.has_value()) {
      piece.push_back(static_cast<char>(byte.value()));
    } else {
      piece.append(token.substr(start, offset - start));
    }
  }
  return piece;
}

}  // namespace

FastTokenizer::FastTokenizer(const TokenizerArgs& tokenizer_args)
    : tokenizer_args_(tokenizer_args),
      byte_level_decoder_(has_byte_level_decoder(tokenizer_args.vocab_file())) {
  handle_ = tokenizers_new_from_path(tokenizer_args.vocab_file().c_str());
  CHECK(handle_ != nullptr)
      << "Failed to load tokenizer from file: " << tokenizer_args.vocab_file();
}

std::unique_ptr<Tokenizer> FastTokenizer::clone() const {
  return std::make_unique<FastTokenizer>(tokenizer_args_);
}

FastTokenizer::~FastTokenizer() { tokenizers_free(handle_); }

namespace {
// Helper function to add a special token to the beginning or end of ids
// Checks if token already exists before adding to avoid duplication
// Returns true on success, false if token is not found, empty, or already
// exists
bool add_special_token_id(const std::string& token,
                          std::optional<int32_t> token_id,
                          std::vector<int32_t>* ids,
                          bool prepend) {
  if (token.empty() || !token_id.has_value()) {
    if (!token.empty() && !token_id.has_value()) {
      LOG(WARNING) << "Failed to find token ID for token: " << token;
    }
    return false;
  }

  const int32_t id = token_id.value();

  // Check if token already exists at the expected position
  if (prepend) {
    // For BOS: check if already at the beginning
    if (!ids->empty() && ids->front() == id) {
      return false;  // Already exists, skip adding
    }
    ids->insert(ids->begin(), id);
  } else {
    // For EOS: check if already at the end
    if (!ids->empty() && ids->back() == id) {
      return false;  // Already exists, skip adding
    }
    ids->push_back(id);
  }
  return true;
}
}  // namespace

bool FastTokenizer::encode(const std::string_view& text,
                           std::vector<int32_t>* ids,
                           bool add_special_tokens) const {
  TokenizerEncodeResult result;
  tokenizers_encode(
      handle_, text.data(), text.size(), add_special_tokens, &result);

  std::vector<int32_t> ret(result.token_ids, result.token_ids + result.len);
  *ids = std::move(ret);

  // Free the memory allocated by Rust tokenizer
  // The token_ids pointer is allocated by Rust's Box::into_raw and must be
  // freed
  if (result.token_ids != nullptr && result.len > 0) {
    tokenizers_free_encode_results(&result, 1);
  }

  // Respect the call-level contract: false means text-only token IDs.
  if (add_special_tokens && tokenizer_args_.add_bos_token() &&
      !tokenizer_args_.bos_token().empty()) {
    const auto bos_id = token_to_id(tokenizer_args_.bos_token());
    add_special_token_id(tokenizer_args_.bos_token(),
                         bos_id,
                         ids,
                         /*prepend=*/true);
  }

  if (add_special_tokens && tokenizer_args_.add_eos_token() &&
      !tokenizer_args_.eos_token().empty()) {
    const auto eos_id = token_to_id(tokenizer_args_.eos_token());
    add_special_token_id(tokenizer_args_.eos_token(),
                         eos_id,
                         ids,
                         /*prepend=*/false);
  }

  return true;
}

std::string FastTokenizer::decode(const Slice<int32_t>& ids,
                                  bool skip_special_tokens) const {
  const char* data = nullptr;
  size_t len = 0;
  tokenizers_decode(handle_,
                    reinterpret_cast<const uint32_t*>(ids.data()),
                    ids.size(),
                    skip_special_tokens,
                    &data,
                    &len);
  return {data, len};
}

std::string FastTokenizer::decode_token(int32_t id) const {
  const std::string piece = id_to_token(id);
  return byte_level_decoder_ ? decode_byte_level_piece(piece) : piece;
}

std::optional<int32_t> FastTokenizer::token_to_id(
    const std::string_view& token) const {
  int32_t id = -1;
  tokenizers_token_to_id(handle_, token.data(), token.size(), &id);
  return id == -1 ? std::optional<int32_t>(std::nullopt)
                  : std::optional<int32_t>(id);
}

std::string FastTokenizer::id_to_token(int32_t id) const {
  const char* data = nullptr;
  size_t len = 0;
  tokenizers_id_to_token(handle_, id, &data, &len);
  return {data, len};
}

size_t FastTokenizer::vocab_size() const {
  size_t size;
  tokenizers_get_vocab_size(handle_, &size);
  CHECK(size > 0) << "vocab_size must be greater than 0.";
  return size;
}

}  // namespace xllm

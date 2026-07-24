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

namespace xllm {

FastTokenizer::FastTokenizer(const TokenizerArgs& tokenizer_args)
    : tokenizer_args_(tokenizer_args) {
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
                           bool add_special_tokens,
                           int32_t max_sequence_length) const {
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

  // Add BOS token if configured
  if (tokenizer_args_.add_bos_token() && !tokenizer_args_.bos_token().empty()) {
    const auto bos_id = token_to_id(tokenizer_args_.bos_token());
    add_special_token_id(tokenizer_args_.bos_token(),
                         bos_id,
                         ids,
                         /*prepend=*/true);
  }

  // Add EOS token if configured
  if (tokenizer_args_.add_eos_token() && !tokenizer_args_.eos_token().empty()) {
    const auto eos_id = token_to_id(tokenizer_args_.eos_token());
    add_special_token_id(tokenizer_args_.eos_token(),
                         eos_id,
                         ids,
                         /*prepend=*/false);
  }
  // Add pad to max_sequence_length if configured
  if (max_sequence_length > 0 && !tokenizer_args_.pad_token().empty()) {
    const auto pad_id = token_to_id(tokenizer_args_.pad_token());
    if (pad_id.has_value() &&
        static_cast<int32_t>(ids->size()) < max_sequence_length) {
      int32_t pad_count =
          max_sequence_length - static_cast<int32_t>(ids->size());
      ids->insert(ids->begin(), pad_count, pad_id.value());
    }
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

/* Copyright 2025-2026 The xLLM Authors.

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

#include "tokenizer_proxy.h"

#include <filesystem>

#include "fast_tokenizer.h"
#include "sentencepiece_tokenizer.h"
#include "tiktoken_tokenizer.h"

namespace xllm {

TokenizerProxy::TokenizerProxy(std::unique_ptr<Tokenizer> tokenizer)
    : tokenizer_(std::move(tokenizer)) {}

std::unique_ptr<Tokenizer> TokenizerProxy::clone() const {
  return std::make_unique<TokenizerProxy>(tokenizer_->clone());
}

bool TokenizerProxy::encode(const std::string_view& text,
                            std::vector<int32_t>* ids,
                            bool add_special_tokens,
                            int32_t max_sequence_length) const {
  return get_tls_tokenizer()->encode(
      text, ids, add_special_tokens, max_sequence_length);
}

bool TokenizerProxy::encode(int64_t item_id,
                            std::vector<int32_t>* token_ids) const {
  return get_tls_tokenizer()->encode(item_id, token_ids);
}

std::string TokenizerProxy::decode(const Slice<int32_t>& ids,
                                   bool skip_special_tokens) const {
  return get_tls_tokenizer()->decode(ids, skip_special_tokens);
}

bool TokenizerProxy::decode(const Slice<int32_t>& token_ids,
                            bool skip_special_tokens,
                            std::vector<int64_t>* item_ids) const {
  return get_tls_tokenizer()->decode(token_ids, skip_special_tokens, item_ids);
}

std::optional<int32_t> TokenizerProxy::token_to_id(
    const std::string_view& token) const {
  return get_tls_tokenizer()->token_to_id(token);
}

std::string TokenizerProxy::id_to_token(int32_t id) const {
  return get_tls_tokenizer()->id_to_token(id);
}

size_t TokenizerProxy::vocab_size() const {
  return get_tls_tokenizer()->vocab_size();
}

Tokenizer* TokenizerProxy::get_tls_tokenizer() const {
  thread_local std::unique_ptr<Tokenizer> tls_tokenizer(tokenizer_->clone());
  return tls_tokenizer.get();
}
}  // namespace xllm

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

#include "tokenizer_proxy.h"

#include <unordered_map>
#include <utility>

#include "fast_tokenizer.h"
#include "sentencepiece_tokenizer.h"
#include "tiktoken_tokenizer.h"

namespace xllm {

TokenizerProxy::TokenizerProxy(std::unique_ptr<Tokenizer> tokenizer)
    : tokenizer_(std::move(tokenizer)),
      tls_identity_(std::make_shared<ThreadLocalIdentity>()) {}

std::unique_ptr<Tokenizer> TokenizerProxy::clone() const {
  return std::make_unique<TokenizerProxy>(tokenizer_->clone());
}

bool TokenizerProxy::encode(const std::string_view& text,
                            std::vector<int32_t>* ids,
                            bool add_special_tokens) const {
  return get_tls_tokenizer()->encode(text, ids, add_special_tokens);
}

bool TokenizerProxy::parallel_batch_encode_enabled() const {
  return tokenizer_->parallel_batch_encode_enabled();
}

bool TokenizerProxy::batch_encode(const std::vector<std::string>& texts,
                                  std::vector<std::vector<int32_t>>* ids,
                                  bool add_special_tokens) const {
  return get_tls_tokenizer()->batch_encode(texts, ids, add_special_tokens);
}

std::string TokenizerProxy::decode(const Slice<int32_t>& ids,
                                   bool skip_special_tokens) const {
  return get_tls_tokenizer()->decode(ids, skip_special_tokens);
}

std::string TokenizerProxy::decode_token(int32_t id) const {
  return get_tls_tokenizer()->decode_token(id);
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

void TokenizerProxy::warmup() const { get_tls_tokenizer()->warmup(); }

Tokenizer* TokenizerProxy::get_tls_tokenizer() const {
  using ThreadLocalTokenizer =
      std::pair<std::weak_ptr<const ThreadLocalIdentity>,
                std::unique_ptr<Tokenizer>>;
  thread_local std::unordered_map<const ThreadLocalIdentity*,
                                  ThreadLocalTokenizer>
      tls_tokenizers;

  const ThreadLocalIdentity* tls_identity = tls_identity_.get();
  auto entry = tls_tokenizers.find(tls_identity);
  if (entry != tls_tokenizers.end()) {
    // A proxy identity remains alive while its methods are called. If this
    // address was reused after a previous proxy was destroyed, discard the
    // stale tokenizer before creating the replacement.
    if (!entry->second.first.expired()) {
      return entry->second.second.get();
    }
    tls_tokenizers.erase(entry);
  }

  auto new_entry =
      tls_tokenizers
          .emplace(tls_identity,
                   ThreadLocalTokenizer{tls_identity_, tokenizer_->clone()})
          .first;
  return new_entry->second.second.get();
}
}  // namespace xllm

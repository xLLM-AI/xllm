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
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/util/slice.h"

namespace xllm {

class Tokenizer {
 public:
  virtual ~Tokenizer() = default;

  virtual bool encode(const std::string_view& text,
                      std::vector<int32_t>* ids,
                      bool add_special_tokens = true) const {
    return false;
  }

  // On success, ids contains one result per input. On failure, ids contains
  // the successful prefix; texts[ids->size()] failed and later inputs have
  // not been encoded. Callers can continue the unattempted suffix without
  // retrying the failed input or discarding successful results.
  virtual bool batch_encode(const std::vector<std::string>& texts,
                            std::vector<std::vector<int32_t>>* ids,
                            bool add_special_tokens = true) const {
    ids->clear();
    ids->reserve(texts.size());
    for (const auto& text : texts) {
      ids->emplace_back();
      if (!encode(text, &ids->back(), add_special_tokens)) {
        ids->pop_back();
        return false;
      }
    }
    return true;
  }

  virtual std::string decode(const Slice<int32_t>& ids,
                             bool skip_special_tokens) const {
    return "";
  }

  // Return a token piece for token-level constraints. Implementations may
  // preserve raw byte-level token bytes instead of applying lossy UTF-8
  // replacement used by sequence decoding.
  virtual std::string decode_token(int32_t id) const {
    return decode(Slice<int32_t>(&id, 1), /*skip_special_tokens=*/false);
  }

  virtual std::optional<int32_t> token_to_id(
      const std::string_view& token) const {
    return std::nullopt;
  }

  virtual std::string id_to_token(int32_t id) const { return ""; }

  virtual size_t vocab_size() const { return 0; }

  // Initialize any thread-local tokenizer state before serving requests.
  virtual void warmup() const {}

  virtual std::unique_ptr<Tokenizer> clone() const { return nullptr; }

  // Whether batch encoding can use multiple workers in the current backend.
  // Query before dispatch so sequential backends can use request workers.
  virtual bool parallel_batch_encode_enabled() const { return false; }
};

}  // namespace xllm

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

  // max_sequence_length is currently honored by FastTokenizer only.
  // Other text tokenizer implementations may accept but ignore it.
  virtual bool encode(const std::string_view& text,
                      std::vector<int32_t>* ids,
                      bool add_special_tokens = true,
                      int32_t max_sequence_length = 0) const {
    return false;
  }

  virtual bool batch_encode(const std::vector<std::string>& texts,
                            std::vector<std::vector<int32_t>>* ids) const {
    for (const auto& text : texts) {
      std::vector<int32_t> single_ids;
      if (!encode(text, &single_ids)) {
        return false;
      }
      ids->push_back(single_ids);
    }
    return true;
  }

  virtual std::string decode(const Slice<int32_t>& ids,
                             bool skip_special_tokens) const {
    return "";
  }

  // Only for generative recommendation
  virtual bool encode(int64_t item_id, std::vector<int32_t>* token_ids) const {
    return false;
  }
  // Only for generative recommendation
  virtual bool decode(const Slice<int32_t>& token_ids,
                      bool skip_special_tokens,
                      std::vector<int64_t>* item_ids) const {
    return false;
  }

  virtual std::optional<int32_t> token_to_id(
      const std::string_view& token) const {
    return std::nullopt;
  }

  virtual std::string id_to_token(int32_t id) const { return ""; }

  virtual size_t vocab_size() const { return 0; }

  virtual std::unique_ptr<Tokenizer> clone() const { return nullptr; }
};

}  // namespace xllm

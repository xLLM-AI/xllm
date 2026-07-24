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

#include "tokenizer.h"
#include "tokenizer_args.h"

namespace xllm {

class TokenizerProxy : public Tokenizer {
 public:
  TokenizerProxy(std::unique_ptr<Tokenizer> tokenizer);

  ~TokenizerProxy() = default;

  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids,
              bool add_special_tokens = true,
              int32_t max_sequence_length = 0) const override;

  bool encode(int64_t item_id, std::vector<int32_t>* token_ids) const override;

  std::string decode(const Slice<int32_t>& ids,
                     bool skip_special_tokens) const override;

  bool decode(const Slice<int32_t>& token_ids,
              bool skip_special_tokens,
              std::vector<int64_t>* item_ids) const override;

  std::optional<int32_t> token_to_id(
      const std::string_view& token) const override;

  std::string id_to_token(int32_t id) const override;

  size_t vocab_size() const override;

  std::unique_ptr<Tokenizer> clone() const override;

 private:
  Tokenizer* get_tls_tokenizer() const;

  std::unique_ptr<Tokenizer> tokenizer_;
};
}  // namespace xllm

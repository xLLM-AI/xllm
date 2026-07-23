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

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "tokenizer.h"
#include "tokenizer_args.h"

namespace xllm {

// Trie tokenizer used by RWKV-5+ World models (rwkv_vocab_v20230424.txt).
class RwkvTokenizer final : public Tokenizer {
 public:
  RwkvTokenizer(const std::string_view& dir_path, const TokenizerArgs& args);

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
  struct VocabData;

  static std::shared_ptr<const VocabData> build_vocab_data(
      const std::string& vocab_file_path);

  static std::shared_ptr<const VocabData> load_vocab_data(
      const std::string& vocab_file_path);

  bool encode_bytes(const std::string_view& text,
                    std::vector<int32_t>* ids) const;

  std::shared_ptr<const VocabData> vocab_;
  std::string dir_path_;
  TokenizerArgs args_;
};

}  // namespace xllm

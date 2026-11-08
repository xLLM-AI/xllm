/* Copyright 2026 The xLLM Authors.

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
#include <optional>
#include <vector>

namespace xllm::util {

// Returns the first token id in `token_ids` that falls outside the valid
// vocabulary range [0, vocab_size); std::nullopt if all ids are in range.
// Callers must skip the scan when vocab_size <= 0 (range unknown).
inline std::optional<int32_t> find_out_of_vocab_token(
    const std::vector<int32_t>& token_ids,
    int64_t vocab_size) {
  for (int32_t token_id : token_ids) {
    if (token_id < 0 || token_id >= vocab_size) {
      return token_id;
    }
  }
  return std::nullopt;
}

}  // namespace xllm::util

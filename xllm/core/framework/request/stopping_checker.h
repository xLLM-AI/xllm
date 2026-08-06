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

#include <cstdint>
#include <unordered_set>
#include <vector>

#include "core/util/slice.h"
#include "finish_reason.h"

namespace xllm {

class StoppingChecker {
 public:
  StoppingChecker() = default;

  StoppingChecker(size_t max_generated_tokens,
                  size_t max_context_len,
                  int32_t eos_token,
                  bool ignore_eos,
                  const std::unordered_set<int32_t>& stop_tokens,
                  const std::vector<std::vector<int32_t>>& stop_sequences);

  FinishReason check(const Slice<int32_t>& token_ids,
                     size_t num_prompt_tokens,
                     size_t* matched_stop_token_count = nullptr) const;

  inline void set_max_generated_tokens(size_t tokens) {
    max_generated_tokens_ = tokens;
  }

  inline constexpr size_t get_max_generated_tokens() const {
    return max_generated_tokens_;
  }

  inline void set_max_context_len(size_t len) { max_context_len_ = len; }

  inline constexpr size_t get_max_context_len() const {
    return max_context_len_;
  }

  inline void set_eos_token(int32_t eos_token) { eos_token_ = eos_token; }

  inline constexpr int32_t get_eos_token() const { return eos_token_; }

  inline void set_ignore_eos(bool ignore_eos) { ignore_eos_ = ignore_eos; }

  inline constexpr bool get_ignore_eos() const { return ignore_eos_; }

  inline void set_stop_tokens(const std::unordered_set<int32_t>& tokens) {
    stop_tokens_ = std::move(tokens);
  }

  inline std::unordered_set<int32_t>& get_stop_tokens() { return stop_tokens_; }

  inline void set_stop_sequences(
      const std::vector<std::vector<int32_t>>& sequences) {
    stop_sequences_ = std::move(sequences);
  }

  inline std::vector<std::vector<int32_t>>& get_stop_sequences() {
    return stop_sequences_;
  }

  size_t get_max_stop_sequence_token_count() const;

 private:
  size_t max_generated_tokens_ = 5120;

  size_t max_context_len_ = 0;

  // eos token id
  int32_t eos_token_ = -1;

  // ignore eos token or not
  bool ignore_eos_ = false;

  // stopping token ids
  std::unordered_set<int32_t> stop_tokens_;

  // stopping sequences
  std::vector<std::vector<int32_t>> stop_sequences_;
};

}  // namespace xllm

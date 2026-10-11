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

#include "core/framework/request/stopping_checker.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace xllm {
namespace {

bool matches_suffix(const Slice<int32_t>& token_ids,
                    const std::vector<int32_t>& suffix) {
  return token_ids.size() >= suffix.size() &&
         std::equal(
             suffix.begin(), suffix.end(), token_ids.end() - suffix.size());
}

}  // namespace

StoppingChecker::StoppingChecker(
    size_t max_generated_tokens,
    size_t max_context_len,
    int32_t eos_token,
    bool ignore_eos,
    std::unordered_set<int32_t> stop_tokens,
    std::vector<std::vector<int32_t>> stop_sequences,
    std::vector<std::string> stop_strings,
    size_t min_generated_tokens)
    : max_generated_tokens_(max_generated_tokens),
      min_generated_tokens_(min_generated_tokens),
      max_context_len_(max_context_len),
      eos_token_(eos_token),
      ignore_eos_(ignore_eos),
      stop_tokens_(std::move(stop_tokens)),
      stop_sequences_(std::move(stop_sequences)),
      stop_strings_(std::move(stop_strings)) {
  CHECK(stop_strings_.empty() ||
        stop_strings_.size() == stop_sequences_.size());
}

size_t StoppingChecker::get_max_stop_sequence_token_count() const {
  size_t max_token_count = 0;
  for (const auto& sequence : stop_sequences_) {
    max_token_count = std::max(max_token_count, sequence.size());
  }
  for (const auto& sequence : generated_stop_sequences_) {
    max_token_count = std::max(max_token_count, sequence.size());
  }
  return max_token_count;
}

FinishReason StoppingChecker::check(const Slice<int32_t>& token_ids,
                                    size_t num_prompt_tokens,
                                    size_t* matched_stop_token_count,
                                    StopReason* stop_reason) const {
  CHECK(!token_ids.empty());
  if (matched_stop_token_count != nullptr) {
    *matched_stop_token_count = 0;
  }
  if (stop_reason != nullptr) {
    *stop_reason = std::monostate{};
  }

  // if enable_schedule_overlap, there might be pre scheduled fake token -1
  // need to figure out the valid token to check finish.
  size_t total_tokens = token_ids.size();
  while (total_tokens > 0 && token_ids[total_tokens - 1] < 0) {
    --total_tokens;
  }
  CHECK_GT(total_tokens, 0);
  const int32_t last_token_id = token_ids[total_tokens - 1];
  const Slice<int32_t> valid_token_ids(token_ids.data(), total_tokens);

  if (total_tokens - num_prompt_tokens < min_generated_tokens_) {
    return FinishReason::NONE;
  }

  // check eos token
  if (!ignore_eos_ && last_token_id == eos_token_) {
    if (matched_stop_token_count != nullptr) {
      *matched_stop_token_count = 1;
    }
    return FinishReason::STOP;
  }

  // Explicit stop tokens are independent of ignore_eos.
  if (stop_tokens_.count(last_token_id) > 0) {
    if (stop_reason != nullptr) {
      *stop_reason = static_cast<int32_t>(last_token_id);
    }
    if (matched_stop_token_count != nullptr) {
      *matched_stop_token_count = 1;
    }
    return FinishReason::STOP;
  }

  // check stop sequences
  for (size_t index = 0; index < stop_sequences_.size(); ++index) {
    const auto& seq = stop_sequences_[index];
    if (seq.back() == last_token_id && matches_suffix(valid_token_ids, seq)) {
      if (stop_reason != nullptr && index < stop_strings_.size() &&
          !stop_strings_[index].empty()) {
        *stop_reason = stop_strings_[index];
      }
      if (matched_stop_token_count != nullptr) {
        // A stop sequence may begin in the prompt and end in generated output.
        // Never hide prompt tokens, including when prompt echo is enabled.
        *matched_stop_token_count =
            std::min(seq.size(), total_tokens - num_prompt_tokens);
      }
      return FinishReason::STOP;
    }
  }

  if (total_tokens > num_prompt_tokens) {
    for (size_t index = 0; index < generated_stop_sequences_.size(); ++index) {
      const auto& seq = generated_stop_sequences_[index];
      if (seq.back() == last_token_id && matches_suffix(valid_token_ids, seq)) {
        if (stop_reason != nullptr && index < generated_stop_strings_.size()) {
          *stop_reason = generated_stop_strings_[index];
        }
        if (matched_stop_token_count != nullptr) {
          *matched_stop_token_count =
              std::min(seq.size(), total_tokens - num_prompt_tokens);
        }
        return FinishReason::STOP;
      }
    }
  }

  // Match explicit stop criteria before length limits. When a stop token lands
  // exactly on the length boundary, vLLM reports a stop and applies
  // include_stop_str_in_output to that token.
  if (max_generated_tokens_ > 0 &&
      total_tokens - num_prompt_tokens >= max_generated_tokens_) {
    return FinishReason::LENGTH;
  }

  if (max_context_len_ > 0 && total_tokens >= max_context_len_) {
    CHECK_GE(total_tokens, num_prompt_tokens) << "Unknown error";
    return FinishReason::LENGTH;
  }

  return FinishReason::NONE;
}

}  // namespace xllm

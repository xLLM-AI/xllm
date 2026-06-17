/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "stopping_checker.h"

#include <absl/strings/match.h>
#include <gflags/gflags_declare.h>

#include <cstdint>
#include <unordered_set>
#include <vector>

#include "core/util/utils.h"

namespace xllm {

StoppingChecker::StoppingChecker(
    size_t max_generated_tokens,
    size_t max_context_len,
    int32_t eos_token,
    bool ignore_eos,
    const std::unordered_set<int32_t>& stop_tokens,
    const std::vector<std::vector<int32_t>>& stop_sequences)
    : max_generated_tokens_(max_generated_tokens),
      max_context_len_(max_context_len),
      eos_token_(eos_token),
      ignore_eos_(ignore_eos),
      stop_tokens_(std::move(stop_tokens)),
      stop_sequences_(std::move(stop_sequences)) {}

FinishReason StoppingChecker::check(const Slice<int32_t>& token_ids,
                                    size_t num_prompt_tokens) const {
  CHECK(!token_ids.empty());

  // if enable_schedule_overlap, there might be pre scheduled fake token -1
  // need to figure out the valid token to check finish.
  size_t last_token_id;
  size_t total_tokens;
  for (auto i = token_ids.size() - 1; i >= 0; --i) {
    if (token_ids[i] >= 0) {
      last_token_id = token_ids[i];
      total_tokens = i + 1;
      break;
    }
  }

  // check max generated tokens
  if (max_generated_tokens_ > 0 &&
      total_tokens - num_prompt_tokens >= max_generated_tokens_) {
    return FinishReason::LENGTH;
  }

  // check max context tokens
  if (max_context_len_ > 0 && total_tokens >= max_context_len_) {
    CHECK_GE(total_tokens, num_prompt_tokens) << "Unknow error";
    return FinishReason::LENGTH;
  }

  // check eos token
  if (!ignore_eos_ && last_token_id == eos_token_) {
    return FinishReason::STOP;
  }

  // check stop tokens
  if (stop_tokens_.count(last_token_id) > 0) {
    return FinishReason::STOP;
  }

  // check stop sequences
  for (const auto& seq : stop_sequences_) {
    if (seq.back() == last_token_id && util::match_suffix(token_ids, seq)) {
      return FinishReason::STOP;
    }
  }

  return FinishReason::NONE;
}

}  // namespace xllm

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

#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace xllm {
namespace detail {

// Decode overlap encodes a prior output row as -(row + 1) in the input tokens.
// Failed sequences must have exactly one such placeholder before replacement.
struct OverlapPlaceholderLocation final {
  int64_t match_count = 0;
  int64_t token_offset = 0;
};

template <typename Token>
inline bool sanitize_cpu_failed_overlap_token_ids(
    torch::Tensor& token_ids,
    const std::vector<std::string>& sample_sequence_ids,
    const std::vector<int32_t>& prior_output_rows,
    const std::unordered_set<std::string_view>& failed_sequence_ids,
    std::string& error) {
  std::unordered_map<Token, OverlapPlaceholderLocation> placeholders;
  placeholders.reserve(failed_sequence_ids.size());
  for (size_t state_idx = 0; state_idx < sample_sequence_ids.size();
       ++state_idx) {
    if (!failed_sequence_ids.contains(sample_sequence_ids[state_idx]) ||
        prior_output_rows[state_idx] < 0) {
      continue;
    }
    const int32_t placeholder_token = -prior_output_rows[state_idx] - 1;
    placeholders.try_emplace(static_cast<Token>(placeholder_token));
  }
  if (placeholders.empty()) {
    return true;
  }

  const Token* input_tokens = token_ids.data_ptr<Token>();
  if (placeholders.size() == 1) {
    auto& [placeholder_token, location] = *placeholders.begin();
    for (int64_t token_offset = 0; token_offset < token_ids.numel();
         ++token_offset) {
      if (input_tokens[token_offset] != placeholder_token) {
        continue;
      }
      ++location.match_count;
      location.token_offset = token_offset;
    }
  } else {
    for (int64_t token_offset = 0; token_offset < token_ids.numel();
         ++token_offset) {
      const auto iter = placeholders.find(input_tokens[token_offset]);
      if (iter == placeholders.end()) {
        continue;
      }
      ++iter->second.match_count;
      iter->second.token_offset = token_offset;
    }
  }

  Token* sanitized_tokens = nullptr;
  for (size_t state_idx = 0; state_idx < sample_sequence_ids.size();
       ++state_idx) {
    if (!failed_sequence_ids.contains(sample_sequence_ids[state_idx]) ||
        prior_output_rows[state_idx] < 0) {
      continue;
    }
    const int32_t placeholder_token = -prior_output_rows[state_idx] - 1;
    auto& location = placeholders.at(static_cast<Token>(placeholder_token));
    if (location.match_count != 1) {
      error =
          "failed overlap row placeholder must identify exactly one input "
          "token";
      return false;
    }
    // Match the tensor path's row-by-row commit order without changing aliases
    // of the original input, including when a later row is invalid.
    if (sanitized_tokens == nullptr) {
      token_ids = token_ids.clone();
      sanitized_tokens = token_ids.data_ptr<Token>();
    }
    sanitized_tokens[location.token_offset] = 0;
    location.match_count = 0;
  }
  return true;
}

inline bool sanitize_failed_overlap_token_ids(
    torch::Tensor* token_ids,
    const std::vector<std::string>& sample_sequence_ids,
    const std::vector<int32_t>& prior_output_rows,
    const std::unordered_set<std::string_view>& failed_sequence_ids,
    std::string* error) {
  if (token_ids == nullptr || error == nullptr) {
    return false;
  }
  error->clear();
  if (failed_sequence_ids.empty() || !token_ids->defined()) {
    return true;
  }
  if (sample_sequence_ids.size() != prior_output_rows.size()) {
    *error = "overlap error sanitization metadata must align";
    return false;
  }

  if (token_ids->device().is_cpu() && token_ids->is_contiguous()) {
    if (token_ids->scalar_type() == torch::kInt32) {
      return sanitize_cpu_failed_overlap_token_ids<int32_t>(*token_ids,
                                                            sample_sequence_ids,
                                                            prior_output_rows,
                                                            failed_sequence_ids,
                                                            *error);
    }
    if (token_ids->scalar_type() == torch::kInt64) {
      return sanitize_cpu_failed_overlap_token_ids<int64_t>(*token_ids,
                                                            sample_sequence_ids,
                                                            prior_output_rows,
                                                            failed_sequence_ids,
                                                            *error);
    }
  }

  for (size_t state_idx = 0; state_idx < sample_sequence_ids.size();
       ++state_idx) {
    if (!failed_sequence_ids.contains(sample_sequence_ids[state_idx])) {
      continue;
    }
    const int32_t prior_output_row = prior_output_rows[state_idx];
    if (prior_output_row < 0) {
      continue;
    }
    const int32_t placeholder_token = -prior_output_row - 1;
    const torch::Tensor& input_token_ids = *token_ids;
    const torch::Tensor placeholder_mask = input_token_ids == placeholder_token;
    const int64_t match_count = placeholder_mask.sum().item<int64_t>();
    if (match_count != 1) {
      *error =
          "failed overlap row placeholder must identify exactly one input "
          "token";
      return false;
    }
    *token_ids = torch::where(
        placeholder_mask, torch::zeros_like(input_token_ids), input_token_ids);
  }
  return true;
}

}  // namespace detail
}  // namespace xllm

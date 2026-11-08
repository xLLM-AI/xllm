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

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/framework/constrained/json_object_grammar.h"
#include "core/runtime/json_object_output_error.h"

namespace xllm {

namespace detail {

constexpr int32_t kNoPriorJsonObjectOutputRow = -1;
constexpr int32_t kInvalidJsonObjectOutputRow = -2;

inline bool resolve_json_object_output_rows(
    const std::vector<JsonObjectGrammarState>& states,
    const std::vector<std::string>& current_sample_sequence_ids,
    const std::vector<int32_t>& current_prior_output_rows,
    const std::vector<std::string>& last_sample_sequence_ids,
    std::vector<int32_t>* output_rows,
    std::vector<JsonObjectOutputError>* output_errors,
    std::string* error) {
  if (output_rows == nullptr || output_errors == nullptr || error == nullptr) {
    return false;
  }
  output_rows->clear();
  output_errors->clear();
  error->clear();
  if (current_sample_sequence_ids.size() != states.size() ||
      current_prior_output_rows.size() != states.size()) {
    *error =
        "JSON grammar states, sampled sequence ids, and prior output rows "
        "must align";
    return false;
  }
  for (const int32_t prior_output_row : current_prior_output_rows) {
    if (prior_output_row < kNoPriorJsonObjectOutputRow) {
      *error = "JSON prior output row must be -1 or non-negative";
      return false;
    }
  }

  std::unordered_map<std::string_view, int32_t> last_step_output_rows;
  last_step_output_rows.reserve(last_sample_sequence_ids.size());
  for (size_t output_idx = 0; output_idx < last_sample_sequence_ids.size();
       ++output_idx) {
    const std::string& sequence_id = last_sample_sequence_ids[output_idx];
    const auto [iter, inserted] = last_step_output_rows.try_emplace(
        sequence_id, static_cast<int32_t>(output_idx));
    if (!inserted) {
      iter->second = kInvalidJsonObjectOutputRow;
    }
  }

  output_rows->reserve(states.size());
  for (size_t state_idx = 0; state_idx < states.size(); ++state_idx) {
    if (!states[state_idx].initialized()) {
      output_rows->emplace_back(kNoPriorJsonObjectOutputRow);
      continue;
    }

    const std::string& sequence_id = current_sample_sequence_ids[state_idx];
    if (sequence_id.empty()) {
      *error = "constrained sampled row requires a non-empty sequence id";
      return false;
    }
    const auto output_iter = last_step_output_rows.find(sequence_id);
    const int32_t expected_output_row = current_prior_output_rows[state_idx];
    if (expected_output_row == kNoPriorJsonObjectOutputRow) {
      if (output_iter != last_step_output_rows.end()) {
        output_rows->emplace_back(kInvalidJsonObjectOutputRow);
        output_errors->push_back(
            {sequence_id, "unexpected prior sampled output row"});
        continue;
      }
      output_rows->emplace_back(kNoPriorJsonObjectOutputRow);
      continue;
    }
    if (output_iter == last_step_output_rows.end()) {
      output_rows->emplace_back(kInvalidJsonObjectOutputRow);
      output_errors->push_back(
          {sequence_id, "missing prior sampled output row"});
      continue;
    }
    if (output_iter->second < 0) {
      output_rows->emplace_back(kInvalidJsonObjectOutputRow);
      output_errors->push_back(
          {sequence_id, "duplicate prior sampled output rows"});
      continue;
    }
    if (output_iter->second != expected_output_row) {
      output_rows->emplace_back(kInvalidJsonObjectOutputRow);
      output_errors->push_back(
          {sequence_id, "prior sampled output row identity mismatch"});
      continue;
    }
    output_rows->emplace_back(output_iter->second);
  }
  return true;
}

struct JsonObjectPlaceholderLocation final {
  int64_t match_count = 0;
  int64_t token_offset = 0;
};

template <typename Token>
inline bool sanitize_cpu_json_object_error_token_ids(
    torch::Tensor& token_ids,
    const std::vector<std::string>& sample_sequence_ids,
    const std::vector<int32_t>& prior_output_rows,
    const std::unordered_set<std::string_view>& failed_sequence_ids,
    std::string& error) {
  std::unordered_map<Token, JsonObjectPlaceholderLocation> placeholders;
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
          "failed JSON row placeholder must identify exactly one input "
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

inline bool sanitize_json_object_error_token_ids(
    torch::Tensor* token_ids,
    const std::vector<std::string>& sample_sequence_ids,
    const std::vector<int32_t>& prior_output_rows,
    const std::vector<JsonObjectOutputError>& errors,
    std::string* error) {
  if (token_ids == nullptr || error == nullptr) {
    return false;
  }
  error->clear();
  if (errors.empty() || !token_ids->defined()) {
    return true;
  }
  if (sample_sequence_ids.size() != prior_output_rows.size()) {
    *error = "JSON error sanitization metadata must align";
    return false;
  }

  std::unordered_set<std::string_view> failed_sequence_ids;
  failed_sequence_ids.reserve(errors.size());
  for (const JsonObjectOutputError& output_error : errors) {
    failed_sequence_ids.emplace(output_error.sample_sequence_id);
  }

  if (token_ids->device().is_cpu() && token_ids->is_contiguous()) {
    if (token_ids->scalar_type() == torch::kInt32) {
      return sanitize_cpu_json_object_error_token_ids<int32_t>(
          *token_ids,
          sample_sequence_ids,
          prior_output_rows,
          failed_sequence_ids,
          *error);
    }
    if (token_ids->scalar_type() == torch::kInt64) {
      return sanitize_cpu_json_object_error_token_ids<int64_t>(
          *token_ids,
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
          "failed JSON row placeholder must identify exactly one input "
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

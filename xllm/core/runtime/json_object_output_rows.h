/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/framework/sampling/json_object_grammar.h"

namespace xllm {

struct JsonObjectOutputError final {
  std::string sample_sequence_id;
  std::string message;
};

namespace detail {

constexpr int32_t kNoPriorJsonObjectOutputRow = -1;
constexpr int32_t kInvalidJsonObjectOutputRow = -2;

inline bool has_request_id_overlap(
    const std::vector<std::string>& current_request_ids,
    const std::vector<std::string>& last_request_ids) {
  if (current_request_ids.empty() || last_request_ids.empty()) {
    return true;
  }

  const auto first_current = std::find_if(
      current_request_ids.begin(),
      current_request_ids.end(),
      [](const std::string& request_id) { return !request_id.empty(); });
  if (first_current == current_request_ids.end()) {
    return false;
  }
  // Stable decode batches usually match near the front. Allocate a hash set
  // only after that common path misses and the batch may have churned.
  if (std::find(last_request_ids.begin(),
                last_request_ids.end(),
                *first_current) != last_request_ids.end()) {
    return true;
  }

  std::unordered_set<std::string> last_request_id_set;
  last_request_id_set.reserve(last_request_ids.size());
  for (const std::string& request_id : last_request_ids) {
    if (!request_id.empty()) {
      last_request_id_set.emplace(request_id);
    }
  }
  return std::any_of(std::next(first_current),
                     current_request_ids.end(),
                     [&last_request_id_set](const std::string& request_id) {
                       return !request_id.empty() &&
                              last_request_id_set.contains(request_id);
                     });
}

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

  std::unordered_map<std::string, int32_t> last_step_output_rows;
  last_step_output_rows.reserve(last_sample_sequence_ids.size());
  for (size_t output_idx = 0; output_idx < last_sample_sequence_ids.size();
       ++output_idx) {
    const std::string& sequence_id = last_sample_sequence_ids[output_idx];
    const auto iter = last_step_output_rows.find(sequence_id);
    if (iter == last_step_output_rows.end()) {
      last_step_output_rows.emplace(sequence_id,
                                    static_cast<int32_t>(output_idx));
    } else {
      iter->second = -2;
    }
  }

  output_rows->reserve(states.size());
  for (size_t state_idx = 0; state_idx < states.size(); ++state_idx) {
    if (!states[state_idx].initialized()) {
      output_rows->emplace_back(-1);
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

  std::unordered_set<std::string> failed_sequence_ids;
  failed_sequence_ids.reserve(errors.size());
  for (const JsonObjectOutputError& output_error : errors) {
    failed_sequence_ids.emplace(output_error.sample_sequence_id);
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

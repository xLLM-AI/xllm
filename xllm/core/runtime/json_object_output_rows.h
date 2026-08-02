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

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/framework/sampling/json_object_grammar.h"

namespace xllm::detail {

inline bool resolve_json_object_output_rows(
    const std::vector<JsonObjectGrammarState>& states,
    const std::vector<std::string>& current_sample_sequence_ids,
    const std::vector<std::string>& last_sample_sequence_ids,
    std::vector<int32_t>* output_rows,
    std::string* error) {
  if (output_rows == nullptr || error == nullptr) {
    return false;
  }
  output_rows->clear();
  error->clear();
  if (current_sample_sequence_ids.size() != states.size()) {
    *error = "JSON grammar states must align with sampled sequence ids";
    return false;
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
    if (output_iter == last_step_output_rows.end()) {
      output_rows->emplace_back(-1);
      continue;
    }
    if (output_iter->second < 0) {
      *error =
          "duplicate last-step constrained sampled sequence id: " + sequence_id;
      return false;
    }
    output_rows->emplace_back(output_iter->second);
  }
  return true;
}

}  // namespace xllm::detail

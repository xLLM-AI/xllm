/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <glog/logging.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/framework/sampling/json_object_grammar.h"

namespace xllm::detail {

struct JsonDraftValidationScratch final {
  // State after each draft step, indexed [draft_step][sequence].
  std::vector<std::vector<JsonObjectGrammarState>> states_after;
  // Invalid suffix flags in [draft_step][sequence] order.
  std::vector<uint8_t> invalid_draft_step_major;
};

inline std::vector<int32_t> copy_json_draft_token_ids(const int64_t* token_ids,
                                                      size_t token_count) {
  std::vector<int32_t> result(token_count);
  if (token_count == 0) {
    return result;
  }
  CHECK(token_ids != nullptr) << "JSON draft token buffer must be defined";
  std::transform(
      token_ids, token_ids + token_count, result.begin(), [](int64_t token_id) {
        return static_cast<int32_t>(token_id);
      });
  return result;
}

inline bool append_json_draft_step(
    std::vector<JsonObjectGrammarState>& current_states,
    std::vector<uint8_t>& invalid_suffix,
    const std::vector<int32_t>& token_ids,
    JsonDraftValidationScratch& scratch) {
  CHECK_EQ(token_ids.size(), current_states.size())
      << "JSON draft token count must match grammar state count";
  CHECK_EQ(invalid_suffix.size(), current_states.size())
      << "JSON invalid suffix count must match grammar state count";

  std::vector<JsonObjectGrammarState> next_states = current_states;
  bool halt_json_draft = false;
  for (size_t state_idx = 0; state_idx < next_states.size(); ++state_idx) {
    const JsonObjectGrammarState& state = current_states[state_idx];
    bool invalid = invalid_suffix[state_idx] != 0;
    if (!invalid && state.initialized()) {
      invalid = !state.can_accept_token(token_ids[state_idx]);
      if (!invalid) {
        CHECK(next_states[state_idx].accept_token(token_ids[state_idx]));
      } else {
        halt_json_draft = true;
      }
    }
    invalid_suffix[state_idx] = static_cast<uint8_t>(invalid);
    scratch.invalid_draft_step_major.push_back(static_cast<uint8_t>(invalid));
  }
  scratch.states_after.push_back(next_states);
  current_states = std::move(next_states);
  return halt_json_draft;
}

inline std::vector<JsonObjectGrammarState> build_json_validation_states(
    const std::vector<JsonObjectGrammarState>& initial_states,
    const JsonDraftValidationScratch& scratch,
    std::vector<uint8_t>& invalid_draft) {
  CHECK(!scratch.states_after.empty())
      << "JSON validation requires at least one draft state";
  const size_t num_sequences = initial_states.size();
  const size_t num_draft_steps = scratch.states_after.size();
  for (const auto& states : scratch.states_after) {
    CHECK_EQ(states.size(), num_sequences)
        << "JSON draft state rows must match validation sequences";
  }
  CHECK_EQ(scratch.invalid_draft_step_major.size(),
           num_sequences * num_draft_steps)
      << "JSON invalid draft flags must cover every draft row";

  std::vector<JsonObjectGrammarState> validation_states;
  validation_states.reserve(num_sequences * (num_draft_steps + 1));
  invalid_draft.clear();
  invalid_draft.reserve(num_sequences * num_draft_steps);
  for (size_t seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
    for (size_t draft_idx = 0; draft_idx < num_draft_steps; ++draft_idx) {
      validation_states.push_back(
          draft_idx == 0 ? initial_states[seq_idx]
                         : scratch.states_after[draft_idx - 1][seq_idx]);
      invalid_draft.push_back(
          scratch
              .invalid_draft_step_major[draft_idx * num_sequences + seq_idx]);
    }
    validation_states.push_back(scratch.states_after.back()[seq_idx]);
  }
  return validation_states;
}

}  // namespace xllm::detail

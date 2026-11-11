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

#include <glog/logging.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "core/framework/constrained/json_object_grammar.h"

namespace xllm::detail {

// One state per verify position, in [seq][verify_position] layout.
inline std::vector<JsonObjectGrammarState> replay_draft_states(
    const std::vector<JsonObjectGrammarState>& initial_states,
    const std::vector<std::vector<int32_t>>& draft_token_ids) {
  const size_t num_sequences = initial_states.size();
  const size_t num_speculative_tokens = draft_token_ids.size();
  for (const auto& step_ids : draft_token_ids) {
    CHECK_EQ(step_ids.size(), num_sequences)
        << "JSON draft token ids must match grammar state rows";
  }

  std::vector<JsonObjectGrammarState> validation_states;
  validation_states.reserve(num_sequences * (num_speculative_tokens + 1));
  for (size_t seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
    JsonObjectGrammarState state = initial_states[seq_idx].clone_state();
    validation_states.push_back(state);
    // Once a sequence mismatches, later draft positions stay at the last valid
    // prefix; skip the per-token checkpointing they would otherwise pay for.
    bool halted = false;
    for (size_t draft_idx = 0; draft_idx < num_speculative_tokens;
         ++draft_idx) {
      if (!halted && state.initialized()) {
        halted = !state.accept_token(draft_token_ids[draft_idx][seq_idx]);
      }
      validation_states.push_back(state);
    }
  }
  return validation_states;
}

inline std::vector<JsonTokenMismatch> find_json_token_mismatches(
    const std::vector<JsonObjectGrammarState>& initial_states,
    const int64_t* accepted_tokens,
    size_t num_rows,
    size_t row_width) {
  CHECK_EQ(initial_states.size(), num_rows)
      << "JSON accepted output rows must match grammar state rows";
  CHECK(accepted_tokens != nullptr || num_rows * row_width == 0)
      << "JSON accepted output token buffer must be defined";

  std::vector<JsonTokenMismatch> mismatches;
  mismatches.reserve(num_rows);
  for (size_t sequence_index = 0; sequence_index < num_rows; ++sequence_index) {
    const JsonObjectGrammarState& state = initial_states[sequence_index];
    if (!state.initialized()) {
      continue;
    }
    if (std::optional<JsonTokenMismatch> mismatch =
            state.clone_state().advance_tokens(
                state.num_accepted_tokens(),
                row_width,
                accepted_tokens + sequence_index * row_width)) {
      mismatch->sequence_index = sequence_index;
      mismatches.push_back(*mismatch);
    }
  }
  return mismatches;
}

}  // namespace xllm::detail

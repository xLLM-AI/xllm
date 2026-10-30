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

#include "core/framework/constrained/json_object_constraint_batch.h"

#include <glog/logging.h>

namespace xllm {

std::vector<JsonObjectConstraintViolation> advance_json_object_constraints(
    std::vector<JsonObjectGrammarState>& states,
    const torch::Tensor& accepted_tokens,
    const std::vector<int32_t>& token_rows) {
  CHECK_EQ(token_rows.size(), states.size())
      << "JSON constraint states must align with accepted-token rows";
  for (const int32_t token_row : token_rows) {
    CHECK_GE(token_row, -1)
        << "JSON constraint token rows must be nonnegative or -1";
  }
  if (!accepted_tokens.defined()) {
    for (size_t state_index = 0; state_index < states.size(); ++state_index) {
      if (!states[state_index].initialized()) {
        continue;
      }
      CHECK_EQ(token_rows[state_index], -1)
          << "JSON constraint advancement requires accepted tokens";
    }
    return {};
  }

  CHECK(accepted_tokens.device().is_cpu())
      << "JSON constraint tokens must already be ready on CPU";
  CHECK_EQ(accepted_tokens.scalar_type(), torch::kInt64)
      << "JSON constraint tokens must have int64 dtype";
  CHECK(accepted_tokens.dim() == 1 || accepted_tokens.dim() == 2)
      << "JSON constraint tokens must be 1-D or 2-D";
  for (size_t state_index = 0; state_index < states.size(); ++state_index) {
    if (!states[state_index].initialized()) {
      continue;
    }
    CHECK_LT(token_rows[state_index], accepted_tokens.size(0))
        << "JSON constraint token row is outside the accepted-token tensor";
  }

  const int64_t* token_data = accepted_tokens.const_data_ptr<int64_t>();
  const int64_t row_stride = accepted_tokens.stride(0);
  const bool has_multiple_tokens = accepted_tokens.dim() == 2;
  const int64_t tokens_per_row =
      has_multiple_tokens ? accepted_tokens.size(1) : 1;
  const int64_t token_stride =
      has_multiple_tokens ? accepted_tokens.stride(1) : 0;
  std::vector<JsonObjectConstraintViolation> violations;
  for (size_t state_index = 0; state_index < states.size(); ++state_index) {
    JsonObjectGrammarState& state = states[state_index];
    const int32_t token_row = token_rows[state_index];
    if (!state.initialized() || token_row == -1) {
      continue;
    }
    const int64_t row_offset = static_cast<int64_t>(token_row) * row_stride;
    for (int64_t token_offset = 0; token_offset < tokens_per_row;
         ++token_offset) {
      const int32_t token_id = static_cast<int32_t>(
          token_data[row_offset + token_offset * token_stride]);
      if (token_id < 0) {
        break;
      }
      if (!state.can_accept_token(token_id)) {
        const JsonObjectGrammarSnapshot snapshot = state.snapshot();
        if (violations.empty()) {
          violations.reserve(states.size());
        }
        violations.emplace_back(
            JsonObjectConstraintViolation{state_index,
                                          static_cast<size_t>(token_offset),
                                          token_id,
                                          snapshot.token_ids.size(),
                                          state.fingerprint()});
        state = JsonObjectGrammarState();
        break;
      }
      CHECK(state.accept_token(token_id));
    }
  }
  return violations;
}

}  // namespace xllm

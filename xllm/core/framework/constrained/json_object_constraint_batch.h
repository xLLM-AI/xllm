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

#include <torch/types.h>

#include <cstdint>
#include <vector>

#include "core/framework/constrained/json_object_grammar.h"
#include "core/framework/speculative/mtp_json_object_state.h"

namespace xllm {

// Reuses the accepted-token mismatch record from the speculative JSON state
// helpers: both scan accepted tokens against grammar states.
using JsonObjectConstraintViolation = detail::JsonAcceptedTokenMismatch;

// Advances the borrowed grammar states in place with their accepted tokens.
// The caller owns the states and any grammar objects they reference. Token
// tensors are read only during the call and are never retained.
//
// Tokens must already be ready on CPU, int64, and 1-D or 2-D. Each state maps
// to one token row, or -1 when it has no newly accepted tokens. Uninitialized
// states are skipped. An undefined tensor is allowed only when every
// initialized state maps to -1. A negative token ends its row. A state whose
// token violates the grammar is recorded and reset to uninitialized.
std::vector<JsonObjectConstraintViolation> advance_json_object_constraints(
    std::vector<JsonObjectGrammarState>& states,
    const torch::Tensor& accepted_tokens,
    const std::vector<int32_t>& token_rows);

}  // namespace xllm

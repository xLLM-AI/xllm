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

namespace xllm {

class JsonObjectGrammarState;

// The verify layout of one request row in a speculative verify output: how
// many draft slots the row was verified over, and whether it decoded under a
// JSON constraint.
struct VerifyRowLayout {
  // Verified draft slots after pruning; zero for a bonus-only sequence.
  // Suffix fallback tokens occupy verified slots even without a cache draft.
  int32_t verify_len = 0;
  bool json_constrained = false;
};

// Builds the per-row verify layout for one [batch, width] verify output.
// Rows default to the full width-1 drafts; per_seq_draft_counts carries the
// pruned per-row widths when adaptive pruning shrank individual rows.
std::vector<VerifyRowLayout> make_verify_layouts(
    const torch::Tensor& next_tokens,
    const std::vector<int32_t>* per_seq_draft_counts,
    const std::vector<JsonObjectGrammarState>& json_object_states);

std::vector<VerifyRowLayout> make_verify_layouts(
    const torch::Tensor& next_tokens);

}  // namespace xllm

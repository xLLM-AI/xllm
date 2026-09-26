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

#include "core/framework/speculative/verify_layout.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstddef>

#include "core/framework/sampling/json_object_grammar.h"

namespace xllm {

std::vector<VerifyRowLayout> make_verify_layouts(
    const torch::Tensor& next_tokens) {
  return make_verify_layouts(next_tokens,
                             /*per_seq_draft_counts=*/nullptr,
                             /*json_object_states=*/{});
}

std::vector<VerifyRowLayout> make_verify_layouts(
    const torch::Tensor& next_tokens,
    const std::vector<int32_t>* per_seq_draft_counts,
    const std::vector<JsonObjectGrammarState>& json_object_states) {
  CHECK(next_tokens.defined()) << "verify output tokens are undefined";
  CHECK_EQ(next_tokens.dim(), 2)
      << "verify output tokens should be [batch, width]";
  CHECK_GT(next_tokens.numel(), 0)
      << "verify output tokens should not be empty";
  const int64_t batch_size = next_tokens.size(0);
  const int32_t max_draft_tokens =
      static_cast<int32_t>(next_tokens.size(1)) - 1;
  const bool have_per_seq = per_seq_draft_counts != nullptr;
  const bool have_json_states = !json_object_states.empty();
  if (have_json_states) {
    CHECK_EQ(json_object_states.size(), static_cast<size_t>(batch_size))
        << "JSON state count does not match verify rows";
  }
  if (have_per_seq) {
    CHECK_EQ(per_seq_draft_counts->size(), static_cast<size_t>(batch_size))
        << "per-sequence draft count batch mismatch";
  }
  std::vector<VerifyRowLayout> row_layouts(static_cast<size_t>(batch_size));
  for (size_t seq_id = 0; seq_id < row_layouts.size(); ++seq_id) {
    row_layouts[seq_id].verify_len =
        have_per_seq
            ? std::clamp((*per_seq_draft_counts)[seq_id], 0, max_draft_tokens)
            : max_draft_tokens;
    row_layouts[seq_id].json_constrained =
        have_json_states && json_object_states[seq_id].initialized();
  }
  return row_layouts;
}

}  // namespace xllm

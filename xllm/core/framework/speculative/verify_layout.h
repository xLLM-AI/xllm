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

// Verify outcome of one speculative verify step: per-request stats.
struct SpecVerifyStats {
  struct RequestStats {
    // Verified draft slots after pruning; zero for a bonus-only sequence.
    int32_t num_draft_tokens = 0;
    // Accepted draft tokens, excluding the bonus token.
    int32_t num_accepted_tokens = 0;
    bool json_constrained = false;
  };
  std::vector<RequestStats> requests;
};

// Builds the per-request verify stats for one [batch, width] verify output.
// per_seq_draft_counts carries the pruned widths when adaptive pruning shrank
// individual sequences; null means every sequence used the full width-1 draft.
SpecVerifyStats make_verify_stats(
    const torch::Tensor& next_tokens,
    const std::vector<int32_t>* per_seq_draft_counts,
    const std::vector<int32_t>& num_accepted_tokens,
    const std::vector<bool>& constrained_mask);

}  // namespace xllm

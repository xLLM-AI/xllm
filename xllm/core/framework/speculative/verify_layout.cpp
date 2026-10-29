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

namespace xllm {

SpecVerifyStats make_verify_stats(
    const torch::Tensor& next_tokens,
    const std::vector<int32_t>* per_seq_draft_counts,
    const std::vector<int32_t>& num_accepted_tokens,
    const std::vector<bool>& constrained_mask) {
  CHECK(next_tokens.defined()) << "verify output tokens are undefined";
  CHECK_EQ(next_tokens.dim(), 2)
      << "verify output tokens should be [batch, width]";
  const int64_t batch_size = next_tokens.size(0);
  const int64_t num_verify_tokens = next_tokens.size(1);
  CHECK_GT(batch_size, 0) << "verify output needs at least one row";
  CHECK_GE(num_verify_tokens, 1)
      << "verify output needs a committed token column";
  const int32_t max_draft_tokens = static_cast<int32_t>(num_verify_tokens) - 1;
  const bool have_per_seq = per_seq_draft_counts != nullptr;
  const bool have_constrained = !constrained_mask.empty();
  if (have_constrained) {
    CHECK_EQ(constrained_mask.size(), static_cast<size_t>(batch_size))
        << "constrained_mask must be empty or match the batch size";
  }
  if (have_per_seq) {
    CHECK_EQ(per_seq_draft_counts->size(), static_cast<size_t>(batch_size))
        << "per-sequence draft count batch mismatch";
  }
  CHECK_EQ(num_accepted_tokens.size(), static_cast<size_t>(batch_size))
      << "accepted token count batch mismatch";
  SpecVerifyStats stats;
  stats.requests.resize(static_cast<size_t>(batch_size));
  for (size_t seq_id = 0; seq_id < stats.requests.size(); ++seq_id) {
    SpecVerifyStats::RequestStats& request = stats.requests[seq_id];
    request.num_draft_tokens =
        have_per_seq
            ? std::clamp((*per_seq_draft_counts)[seq_id], 0, max_draft_tokens)
            : max_draft_tokens;
    request.num_accepted_tokens = num_accepted_tokens[seq_id];
    // Accepted drafts always land inside the row's verified slots.
    CHECK_LE(request.num_accepted_tokens, request.num_draft_tokens)
        << "accepted token count exceeds the row's verified width";
    request.json_constrained = have_constrained && constrained_mask[seq_id];
  }
  return stats;
}

}  // namespace xllm

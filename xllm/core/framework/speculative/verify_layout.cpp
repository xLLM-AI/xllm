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

#include <cstddef>

#include "core/util/tensor_helper.h"

namespace xllm {

SpecVerifyStats make_verify_stats(const torch::Tensor& next_tokens,
                                  const std::vector<char>& constrained_mask) {
  const int64_t batch_size = next_tokens.size(0);
  const int64_t num_verify_tokens = next_tokens.size(1);
  const int32_t max_draft_tokens = static_cast<int32_t>(num_verify_tokens) - 1;
  const bool have_constrained = !constrained_mask.empty();
  const torch::Tensor host_tokens =
      to_cpu_contiguous(next_tokens, torch::kInt32);
  const Slice<int32_t> tokens = tensor_slice<int32_t>(host_tokens);
  SpecVerifyStats stats;
  stats.requests.resize(static_cast<size_t>(batch_size));
  for (size_t seq_id = 0; seq_id < stats.requests.size(); ++seq_id) {
    SpecVerifyStats::RequestStats& request = stats.requests[seq_id];
    request.num_draft_tokens = max_draft_tokens;
    int32_t accepted = 0;
    while (accepted < num_verify_tokens &&
           tokens[static_cast<size_t>(seq_id) * num_verify_tokens + accepted] >=
               0) {
      ++accepted;
    }
    request.num_accepted_tokens = accepted - 1;
    request.json_constrained =
        have_constrained && constrained_mask[seq_id] != 0;
  }
  return stats;
}

}  // namespace xllm

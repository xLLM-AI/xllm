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

#include <algorithm>
#include <cstdint>

#include "core/common/types.h"

namespace xllm {

// Token capacity shared by scheduling, graph warmup and cache burst
// reservation. Inputs are resolved at startup; chunking is the original switch
// before multi-SLO mode selection.
class PrefillTokenBudget final {
 public:
  PrefillTokenBudget(int64_t max_batch_tokens,
                     int64_t max_chunk_tokens,
                     bool enable_chunked_prefill)
      : max_batch_tokens_(max_batch_tokens),
        minimum_group_tokens_(enable_chunked_prefill
                                  ? std::min(max_chunk_tokens, max_batch_tokens)
                                  : max_batch_tokens) {}

  int64_t per_dp_group(int64_t dp_size,
                       InstanceRole role,
                       bool enable_fair_budget) const {
    if (!enable_fair_budget || dp_size <= 1 || role != InstanceRole::PREFILL) {
      return max_batch_tokens_;
    }
    // A whole chunk (or whole non-chunked prompt) must still make progress.
    const int64_t per_group_share =
        max_batch_tokens_ / dp_size + (max_batch_tokens_ % dp_size != 0);
    return std::max(per_group_share, minimum_group_tokens_);
  }

 private:
  int64_t max_batch_tokens_;
  int64_t minimum_group_tokens_;
};

}  // namespace xllm

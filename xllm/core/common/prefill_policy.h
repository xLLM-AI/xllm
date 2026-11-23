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

#include <algorithm>
#include <cstdint>

#include "core/common/types.h"

namespace xllm {

// Resolves the requested chunk and owns the token-budget rules shared by
// scheduling, graph warmup and cache reservation. Chunking is the original
// startup switch, before multi-SLO batch-mode selection.
class PrefillPolicy final {
 public:
  PrefillPolicy(int64_t max_batch_tokens,
                int64_t requested_chunk_tokens,
                bool enable_chunked_prefill,
                bool enable_fair_budget)
      : max_batch_tokens_(max_batch_tokens),
        chunk_tokens_(requested_chunk_tokens < 0 ? max_batch_tokens
                                                 : requested_chunk_tokens),
        enable_chunked_prefill_(enable_chunked_prefill),
        enable_fair_budget_(enable_fair_budget) {
    CHECK_GT(max_batch_tokens, 0) << "max_tokens_per_batch must be positive";
    CHECK_NE(requested_chunk_tokens, 0)
        << "max_tokens_per_chunk_for_prefill must be positive or negative for "
           "auto";
  }

  int64_t chunk_tokens() const { return chunk_tokens_; }
  bool enable_chunked_prefill() const { return enable_chunked_prefill_; }

  int64_t per_dp_group(int64_t dp_size, InstanceRole role) const {
    if (!enable_fair_budget_ || !enable_chunked_prefill_ || dp_size <= 1 ||
        role != InstanceRole::PREFILL) {
      return max_batch_tokens_;
    }
    // Round up each share and retain at least one chunk so long prompts
    // advance.
    const int64_t per_group_share =
        max_batch_tokens_ / dp_size + (max_batch_tokens_ % dp_size != 0);
    return std::max(per_group_share,
                    std::min(chunk_tokens_, max_batch_tokens_));
  }

 private:
  int64_t max_batch_tokens_;
  int64_t chunk_tokens_;
  bool enable_chunked_prefill_;
  bool enable_fair_budget_;
};

}  // namespace xllm

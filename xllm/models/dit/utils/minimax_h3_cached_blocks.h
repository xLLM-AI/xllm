/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <cstdint>
#include <stdexcept>
#include <utility>

#include "models/dit/utils/minimax_h3_qualified_execution_context.h"

namespace xllm {

struct H3QualifiedCachedBlockOutput {
  torch::Tensor hidden;
  SimilarityResidualCacheDecision decision;
};

// The callback executes one real block and must not modify its input in place.
// Keeping this controller independent of weights permits CPU tests to exercise
// the exact branch/ordering used by the resident transformer.
template <typename ExecuteBlock>
H3QualifiedCachedBlockOutput run_h3_qualified_cached_blocks(
    const torch::Tensor& hidden,
    int64_t step,
    int64_t block_count,
    SimilarityResidualCacheState& cache,
    const H3QualifiedCacheProbeRows& rows,
    ProcessGroup* consensus_group,
    ExecuteBlock&& execute_block) {
  const int64_t middle_start = cache.front_blocks();
  const int64_t back_blocks = cache.back_blocks();
  if (middle_start >= block_count ||
      back_blocks >= block_count - middle_start) {
    throw std::invalid_argument(
        "H3 CacheDiT requires a nonempty middle block stack");
  }
  const int64_t tail_start = block_count - back_blocks;
  torch::Tensor front = hidden;
  for (int64_t layer = 0; layer < middle_start; ++layer) {
    front = execute_block(layer, front);
  }
  const SimilarityResidualCacheDecision decision =
      cache.decide(step,
                   front - hidden,
                   rows.used_rows,
                   consensus_group,
                   rows.target_indices);
  torch::Tensor current;
  if (decision.hit) {
    current = cache.apply(step, front);
  } else {
    current = front;
    for (int64_t layer = middle_start; layer < tail_start; ++layer) {
      current = execute_block(layer, current);
    }
    // The reusable residual excludes the back blocks, which run on every step.
    cache.record_dense(step, front, current);
  }
  for (int64_t layer = tail_start; layer < block_count; ++layer) {
    current = execute_block(layer, current);
  }
  return {.hidden = std::move(current), .decision = decision};
}

}  // namespace xllm

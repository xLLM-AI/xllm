/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

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
#include <limits>

#include "framework/block/kv_cache_manager.h"
#include "framework/request/sequence.h"
#include "util/utils.h"

namespace xllm::scheduler {

// Returns the additional KV blocks needed after a decode cursor advances.
inline size_t estimate_decode_extra_blocks(Sequence* sequence,
                                           size_t updated_num_tokens,
                                           size_t block_size) {
  const size_t num_blocks = sequence->kv_state().num_blocks(BlockType::KV);
  const size_t num_blocks_needed =
      (updated_num_tokens + block_size - 1) / block_size;
  if (num_blocks_needed > num_blocks) {
    return num_blocks_needed - num_blocks;
  }
  if (sequence->check_beam_search() &&
      !sequence->kv_state().src_blocks().empty() &&
      sequence->kv_state().need_swap()) {
    return 1;
  }
  return 0;
}

// Includes the target-validation and prelaunch horizon when overlap is active.
inline size_t get_decode_allocation_tokens(
    Sequence* sequence,
    size_t baseline_tokens,
    int32_t num_speculative_tokens,
    int32_t min_speculative_tokens_required,
    bool enable_schedule_overlap) {
  if (!enable_schedule_overlap || num_speculative_tokens <= 0) {
    return baseline_tokens;
  }
  CHECK_GT(min_speculative_tokens_required, 0);
  const size_t overlap_horizon =
      static_cast<size_t>(min_speculative_tokens_required) + 1;
  const size_t kv_cache_tokens = sequence->kv_state().kv_cache_tokens_num();
  CHECK_LE(kv_cache_tokens,
           std::numeric_limits<size_t>::max() - overlap_horizon);
  return std::max(baseline_tokens, kv_cache_tokens + overlap_horizon);
}

inline size_t get_sequence_free_blocks_for_rank(
    KVCacheManager* kv_cache_manager,
    int32_t dp_rank) {
  const auto free_blocks = kv_cache_manager->num_free_blocks();
  if (free_blocks.empty()) {
    return 0;
  }
  if (dp_rank >= 0 && static_cast<size_t>(dp_rank) < free_blocks.size()) {
    return free_blocks[dp_rank];
  }
  return util::max(free_blocks);
}

}  // namespace xllm::scheduler

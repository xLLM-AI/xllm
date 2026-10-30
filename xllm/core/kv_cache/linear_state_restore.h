/* Copyright 2025-2026 The xLLM Authors.

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
#include <vector>

#include "core/kv_cache/linear_state_types.h"
#include "core/kv_cache/storage/kv_cache.h"

namespace xllm {

// Convert logical sequence KV cursors to the warm/cold state mask consumed by
// active linear-attention rows. Data-parallel expansion keeps each logical
// row contiguous in the execution batch.
std::vector<int64_t> build_linear_state_mask(
    const std::vector<int32_t>& cached_tokens,
    int64_t active_rows);

void restore_linear_state_slots(std::vector<KVCache>& kv_caches,
                                const std::vector<int32_t>& write_ids,
                                const std::vector<int32_t>& read_ids,
                                std::vector<int64_t>& validity_mask,
                                bool reads_distinct_state);

}  // namespace xllm

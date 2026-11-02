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
#include <cmath>
#include <cstdint>

namespace xllm {

struct VmmArenaCapacity {
  int64_t maximum_bytes = 0;
  int64_t initial_bytes = 0;
  int64_t growth_memory_reserve_bytes = 0;
};

// Inputs are validated by the coordinator before any worker is initialized.
inline VmmArenaCapacity calculate_vmm_arena_capacity(
    int64_t available_memory,
    int64_t total_memory,
    double max_memory_utilization,
    bool max_memory_utilization_configured,
    bool lazy,
    int64_t max_cache_size,
    int64_t reserved_memory_bytes = 0,
    int64_t resident_memory_bytes = 0) {
  VmmArenaCapacity capacity;
  const bool has_ratio_cap = !lazy || (max_memory_utilization_configured &&
                                       max_memory_utilization > 0.0);
  if (has_ratio_cap && max_memory_utilization < 1.0) {
    capacity.growth_memory_reserve_bytes = static_cast<int64_t>(
        std::ceil(total_memory * (1.0 - max_memory_utilization)));
  }
  // Keep addresses stable up to the eventual ceiling, while activating only
  // the slots supported by current memory. Physical growth must preserve the
  // ratio reserve even when non-pool device allocations change later.
  capacity.maximum_bytes = lazy ? total_memory : available_memory;
  capacity.maximum_bytes -= capacity.growth_memory_reserve_bytes;
  if (max_cache_size > 0) {
    capacity.maximum_bytes = std::min(capacity.maximum_bytes, max_cache_size);
  }
  // Keep non-pool allocations, such as the VLM encoder cache, outside both
  // the startup pool and future elastic growth.
  capacity.maximum_bytes -= reserved_memory_bytes;
  capacity.growth_memory_reserve_bytes += reserved_memory_bytes;
  // Persistent states are allocated before KV pages are mapped. They reduce
  // the arena and startup budget, but their resident bytes already reduce
  // live free memory and must not remain a physical-growth reserve.
  capacity.maximum_bytes -= resident_memory_bytes;
  capacity.maximum_bytes = std::max<int64_t>(capacity.maximum_bytes, 0);
  capacity.initial_bytes =
      std::min(capacity.maximum_bytes,
               std::max<int64_t>(available_memory -
                                     capacity.growth_memory_reserve_bytes -
                                     resident_memory_bytes,
                                 0));
  return capacity;
}

}  // namespace xllm

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
#include <string>

#include "core/framework/config/parallel_config.h"
#include "core/kv_cache/storage/kv_cache_utils.h"
#include "core/platform/platform.h"

namespace xllm {

// Adds an explicit logical descriptor to a cache tensor produced by KVCache.
// Returns false with an actionable error when the tensor cannot be represented
// without guessing an unsupported layout.
bool describe_cache_tensor(const CacheTensorLayoutContext& context,
                           KVCacheTensor* cache_tensor,
                           std::string* error);

inline int64_t indexer_pages_per_block() {
  if (!Platform::supports_dsa_indexer_cache_sharding()) {
    return 1;
  }
  return std::max<int64_t>(
      ParallelConfig::get_instance().kv_split_size_effective(), 1);
}

}  // namespace xllm

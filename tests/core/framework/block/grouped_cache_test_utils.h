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

#include <initializer_list>
#include <utility>
#include <vector>

#include "core/kv_cache/layout/paged_cache_partition.h"

namespace xllm::test {

inline std::vector<PagedCachePartition> grouped_cache_partitions(
    int32_t block_size,
    int32_t window_size,
    std::initializer_list<std::pair<int32_t, bool>> histories = {{4, true},
                                                                 {128, true}}) {
  std::vector<PagedCachePartition> partitions;
  partitions.reserve(histories.size() + 1);
  partitions.emplace_back(
      CachePartitionId{},
      block_size,
      /*enable_prefix_cache=*/true,
      std::vector<CacheComponent>{
          {CacheComponentRole::KEY, CacheScalarType::BFLOAT16, {0}}},
      /*compression_ratio=*/1,
      window_size);
  for (const auto& [ratio, prefix_cache] : histories) {
    partitions.emplace_back(
        CachePartitionId{"", static_cast<uint32_t>(partitions.size())},
        block_size * ratio,
        prefix_cache,
        std::vector<CacheComponent>{
            {CacheComponentRole::KEY, CacheScalarType::BFLOAT16, {0}}},
        ratio);
  }
  return partitions;
}

}  // namespace xllm::test

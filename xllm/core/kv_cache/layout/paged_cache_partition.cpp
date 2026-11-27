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

#include "core/kv_cache/layout/paged_cache_partition.h"

#include <glog/logging.h>

#include <algorithm>
#include <functional>
#include <unordered_set>
#include <utility>

namespace xllm {

PagedCachePartition::PagedCachePartition(CachePartitionId id,
                                         int32_t token_stride,
                                         bool enable_prefix_cache,
                                         std::vector<CacheComponent> components,
                                         int32_t compression_ratio,
                                         int32_t window_size)
    : id_(std::move(id)),
      token_stride_(token_stride),
      compression_ratio_(compression_ratio),
      window_size_(window_size),
      enable_prefix_cache_(enable_prefix_cache),
      components_(std::move(components)) {
  CHECK_GT(token_stride_, 0) << "Paged cache token stride must be positive";
  CHECK_GT(compression_ratio_, 0) << "Compression ratio must be positive";
  CHECK_EQ(token_stride_ % compression_ratio_, 0)
      << "Token stride must contain complete compressed rows";
  CHECK_GE(window_size_, 0) << "Window size cannot be negative";
  CHECK(!is_sliding_window() || compression_ratio_ == 1)
      << "Sliding windows use uncompressed token rows";
  CHECK(!components_.empty())
      << "Paged cache partition must contain components";
  std::unordered_set<CacheComponentRole> roles;
  for (const auto& component : components_) {
    CHECK(roles.insert(component.role).second) << "Duplicate cache component";
    CHECK(!component.layers.empty()) << "Cache component must own layers";
    CHECK_GE(component.layers.front(), 0);
    CHECK(std::adjacent_find(component.layers.begin(),
                             component.layers.end(),
                             std::greater_equal<int32_t>()) ==
          component.layers.end())
        << "Cache component layers must be strictly increasing";
  }
}

const CacheComponent* PagedCachePartition::find_component(
    CacheComponentRole role) const {
  const auto it = std::find_if(
      components_.begin(), components_.end(), [role](const auto& component) {
        return component.role == role;
      });
  return it == components_.end() ? nullptr : &*it;
}

}  // namespace xllm

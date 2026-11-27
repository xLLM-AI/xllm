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

#include <cstdint>
#include <vector>

#include "core/kv_cache/layout/cache_partition.h"

namespace xllm {

// Components in this partition grow and restore together using one block ID
// space. Logical token stride can differ from the tensor block size under KV
// split.
class PagedCachePartition final {
 public:
  PagedCachePartition(CachePartitionId id,
                      int32_t token_stride,
                      bool enable_prefix_cache,
                      std::vector<CacheComponent> components,
                      int32_t compression_ratio = 1,
                      int32_t window_size = 0);

  const CachePartitionId& id() const { return id_; }
  int32_t token_stride() const { return token_stride_; }
  int32_t compression_ratio() const { return compression_ratio_; }
  int32_t window_size() const { return window_size_; }
  bool is_sliding_window() const { return window_size() > 0; }
  bool participates_in_admission() const { return !is_sliding_window(); }
  // Engine settings and instance role may further restrict prefix reuse.
  bool enable_prefix_cache() const { return enable_prefix_cache_; }
  const std::vector<CacheComponent>& components() const { return components_; }
  const CacheComponent* find_component(CacheComponentRole role) const;

 private:
  CachePartitionId id_;
  int32_t token_stride_;
  int32_t compression_ratio_;
  // Zero denotes full history; positive values denote a sliding window.
  int32_t window_size_;
  bool enable_prefix_cache_;
  std::vector<CacheComponent> components_;
};

}  // namespace xllm

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

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace xllm {

// Provides physical capacity and mappings for KV pages. Reservations also
// account for other users of the same memory pool. Methods are synchronous;
// implementations must not call back into the KV page allocator while holding
// their budget lock. Mapping offsets are bytes within each layer's K/V region.
class KVCachePageBackend {
 public:
  virtual ~KVCachePageBackend() = default;

  virtual bool try_reserve_pages(const std::string& model_id,
                                 int32_t dp_rank,
                                 size_t physical_pages) = 0;
  virtual void release_pages(const std::string& model_id,
                             int32_t dp_rank,
                             size_t physical_pages) = 0;
  virtual size_t available_pages(const std::string& model_id,
                                 int32_t dp_rank) const = 0;
  virtual bool map_pages(const std::string& model_id,
                         int32_t dp_rank,
                         const std::vector<int64_t>& byte_offsets) = 0;
  virtual bool unmap_pages(const std::string& model_id,
                           int32_t dp_rank,
                           const std::vector<int64_t>& byte_offsets) = 0;
};

}  // namespace xllm

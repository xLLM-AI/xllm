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
#include <memory>
#include <string>
#include <vector>

#include "core/kv_cache/block/kv_cache_page_state.h"

namespace xllm {

// KV page operations supplied by the runtime. Implementations retain the
// shared memory budget and mapping coordination; cache block managers do not
// depend on model lifecycle, weight reservations or distributed RPCs.
class KVCachePageAllocator {
 public:
  virtual ~KVCachePageAllocator() = default;

  virtual bool is_initialized() const = 0;
  virtual void start_prealloc_thread() = 0;

  virtual std::unique_ptr<KVCachePageState> alloc_kv_cache_page(
      const std::string& model_id,
      int32_t dp_rank) = 0;
  virtual void free_kv_cache_pages(
      const std::string& model_id,
      int32_t dp_rank,
      const std::vector<int64_t>& virtual_page_ids) = 0;
  virtual void trim_kv_cache(const std::string& model_id, int32_t dp_rank) = 0;

  virtual size_t get_num_inuse_virt_pages(const std::string& model_id,
                                          int32_t dp_rank) const = 0;
  virtual size_t get_num_reserved_virt_pages(const std::string& model_id,
                                             int32_t dp_rank) const = 0;
  virtual int64_t get_virt_page_id(int64_t block_id,
                                   size_t block_memory_size) const = 0;
  virtual size_t page_size() const = 0;
  virtual size_t phy_pages_per_virt_page(const std::string& model_id) const = 0;
};

}  // namespace xllm

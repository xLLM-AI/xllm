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

#include "core/kv_cache/transfer/paged_kv_cache_transfer_memory_provider.h"

#include <glog/logging.h>

#include <limits>
#include <utility>

#include "core/framework/allocator/global_memory_region.h"
#include "core/kv_cache/storage/kv_cache_memory_registry.h"

namespace xllm {
namespace {

class PagedKVCacheTransferMemoryProvider final
    : public KVCacheTransferMemoryProvider {
 public:
  PagedKVCacheTransferMemoryProvider(KVCacheMemoryRegistry& registry,
                                     GlobalMemoryRegion& region)
      : registry_(registry), region_(region) {}

  KVCacheTransferMemoryRegion memory_region() const override {
    auto lifetime = region_.acquire_mapping_lease();
    return {region_.base_vaddr(), region_.total_size(), std::move(lifetime)};
  }

  std::pair<uint64_t, uint64_t> get_global_offsets_for_block(
      const std::string& model_id,
      int64_t layer_id,
      uint64_t block_id,
      size_t block_size) const override {
    if (block_id > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      LOG(ERROR) << "KV cache transfer block id exceeds int64_t range: "
                 << block_id;
      return {UINT64_MAX, UINT64_MAX};
    }
    if (!region_.is_initialized()) {
      LOG(ERROR) << "Global memory region is not initialized";
      return {UINT64_MAX, UINT64_MAX};
    }
    return registry_.get_global_offsets_for_block(
        model_id, layer_id, static_cast<int64_t>(block_id), block_size);
  }

 private:
  KVCacheMemoryRegistry& registry_;
  GlobalMemoryRegion& region_;
};

}  // namespace

std::unique_ptr<KVCacheTransferMemoryProvider>
create_paged_kv_cache_transfer_memory_provider(KVCacheMemoryRegistry& registry,
                                               GlobalMemoryRegion& region) {
  return std::make_unique<PagedKVCacheTransferMemoryProvider>(registry, region);
}

}  // namespace xllm

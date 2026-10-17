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
#include <optional>
#include <string>
#include <utility>

#include "core/kv_cache/layout/virtual_kv_cache_offsets.h"

namespace xllm {

// The optional lease keeps the mapping valid while registered for transfers.
struct VirtualKVCacheTransferMemoryRegion {
  void* base_address = nullptr;
  size_t size_bytes = 0;
  std::shared_ptr<void> lifetime;
  size_t page_size_bytes = 0;
};

// The runtime supplies shared mappings and physical byte offsets for virtual
// KV caches without exposing model lifecycle or distributed coordination.
class VirtualKVCacheTransferMemoryProvider {
 public:
  virtual ~VirtualKVCacheTransferMemoryProvider() = default;

  virtual VirtualKVCacheTransferMemoryRegion memory_region() const = 0;

  virtual std::optional<VirtualKVCacheBlockGlobalOffsets>
  get_global_offsets_for_block(const std::string& model_id,
                               int64_t layer_id,
                               uint64_t block_id) const = 0;

  // Returns {UINT64_MAX, UINT64_MAX} for an invalid or unmapped block.
  virtual std::pair<uint64_t, uint64_t> get_global_offsets_for_block(
      const std::string& model_id,
      int64_t layer_id,
      uint64_t block_id,
      size_t block_size) const = 0;
};

}  // namespace xllm

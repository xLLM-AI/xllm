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
#include <memory>
#include <unordered_set>
#include <vector>

#include "core/kv_cache/layout/static_kv_cache_layout.h"
#include "core/kv_cache/storage/kv_cache_tensor_allocator.h"

namespace xllm {

// Owns one eagerly allocated arena. Returned tensor views share its storage
// and keep it alive independently of the allocator's lifetime.
class StaticKVCacheTensorAllocator final : public KVCacheTensorAllocator {
 public:
  StaticKVCacheTensorAllocator(StaticKVCacheLayout layout,
                               const torch::Device& device);

  void begin_layer(int64_t layer_id) override;
  torch::Tensor allocate(KVCacheTensorRole role,
                         const std::vector<int64_t>& shape,
                         torch::ScalarType dtype,
                         const torch::Device& device) override;

  // Epochs correspond to the logical-id entries in rows. Existing leases,
  // including cached prefixes, retain their contents.
  bool needs_full_block_initialization(
      const std::vector<int32_t>& rows,
      const std::vector<uint64_t>& lease_epochs) const;
  void initialize_full_blocks(const std::vector<int32_t>& rows,
                              const std::vector<uint64_t>& lease_epochs);

  const StaticKVCacheLayout& layout() const { return layout_; }
  size_t bytes_used() const { return layout_.bytes_used(); }

 private:
  StaticKVCacheLayout layout_;
  torch::Tensor arena_;
  std::vector<uint64_t> initialized_full_block_epochs_;
  std::unordered_set<int64_t> started_layers_;
  int64_t current_layer_id_ = -1;
  uint32_t allocated_roles_ = 0;
};

std::shared_ptr<StaticKVCacheTensorAllocator> create_static_kv_tensor_allocator(
    const KVCacheShape& shape,
    const KVCacheCreateOptions& options);

}  // namespace xllm

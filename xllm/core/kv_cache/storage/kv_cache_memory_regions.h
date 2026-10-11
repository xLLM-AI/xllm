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

#include <torch/types.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "core/framework/allocator/virtual_memory/mapped_memory_region.h"

namespace xllm {

// Owns the layer-wise virtual memory regions of one model's KV cache.
// The caller synchronizes access and supplies the device and page geometry.
class KVCacheMemoryRegions final {
 public:
  KVCacheMemoryRegions() = default;
  ~KVCacheMemoryRegions() = default;

  KVCacheMemoryRegions(const KVCacheMemoryRegions&) = delete;
  KVCacheMemoryRegions& operator=(const KVCacheMemoryRegions&) = delete;
  KVCacheMemoryRegions(KVCacheMemoryRegions&&) noexcept = default;
  KVCacheMemoryRegions& operator=(KVCacheMemoryRegions&&) noexcept = default;

  std::vector<torch::Tensor> create_k_tensors(const std::vector<int64_t>& dims,
                                              torch::Dtype dtype,
                                              int64_t num_layers,
                                              const torch::Device& device,
                                              size_t page_size);
  std::vector<torch::Tensor> create_v_tensors(const std::vector<int64_t>& dims,
                                              torch::Dtype dtype,
                                              int64_t num_layers,
                                              const torch::Device& device,
                                              size_t page_size);

  bool map(const std::vector<offset_t>& offsets);
  bool unmap(const std::vector<offset_t>& offsets);

  // Returns pool-relative offsets, or {UINT64_MAX, UINT64_MAX} on failure.
  std::pair<uint64_t, uint64_t> get_global_offsets_for_block(
      int64_t layer_id,
      int64_t block_id,
      size_t block_size) const;

  int64_t num_layers() const { return num_layers_; }

 private:
  std::vector<torch::Tensor> create_tensors(
      const std::vector<int64_t>& dims,
      torch::Dtype dtype,
      int64_t num_layers,
      const torch::Device& device,
      size_t page_size,
      const char* name,
      std::vector<std::unique_ptr<MappedMemoryRegion>>& regions);

  std::vector<std::unique_ptr<MappedMemoryRegion>> k_regions_;
  std::vector<std::unique_ptr<MappedMemoryRegion>> v_regions_;
  int64_t num_layers_ = 0;
};

}  // namespace xllm

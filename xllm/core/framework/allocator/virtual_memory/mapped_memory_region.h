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
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/framework/allocator/virtual_memory/physical_page.h"
#include "core/platform/vmm_api.h"

namespace xllm {

// Type definitions (page_id_t is defined in physical_page.h)
using offset_t = page_id_t;

// Mapping operations require synchronization by the caller.
class MappedMemoryRegion final {
 public:
  MappedMemoryRegion(size_t size,
                     torch::Dtype dtype,
                     torch::Device dev,
                     size_t page_size);

  // Map externally selected pages in the supplied logical order.
  // The region owns this reservation and releases it during destruction.
  MappedMemoryRegion(std::vector<page_id_t> page_ids,
                     torch::Dtype dtype,
                     torch::Device dev,
                     size_t page_size);

  ~MappedMemoryRegion();

  bool map(offset_t offset);
  bool unmap(offset_t offset);

  // Validate every target and reserve all missing pages before changing any
  // mapping. Duplicate targets and existing mappings are idempotent.
  static bool map_pages(
      const std::vector<std::pair<MappedMemoryRegion*, offset_t>>& targets);

  // Map/unmap every page in the virtual range.
  bool map_all();
  bool unmap_all();

  // Map all pages using reserved physical page IDs.
  // page_ids: physical page IDs to use
  // Returns true on success
  bool map_with_page_ids(const std::vector<page_id_t>& page_ids);

  // Check whether this region owns a reservation of externally selected pages.
  bool is_using_preallocated_pages() const { return use_preallocated_pages_; }

  // Convert the underlying memory to a torch::Tensor.
  // For NPU devices, uses convert_to_torch_tensor; for others, uses from_blob.
  torch::Tensor to_torch_tensor() const;

  // Convert a portion of the underlying memory to a torch::Tensor.
  // offset: byte offset from the start of the tensor
  // dims: dimensions of the returned tensor
  torch::Tensor to_torch_tensor(size_t offset,
                                const std::vector<int64_t>& dims) const;

  inline size_t size() const noexcept { return size_; }
  inline size_t page_size() const noexcept { return page_size_; }
  inline VirPtr vaddr() const noexcept { return vaddr_; }

  // Geometry for non-owning tensor views.
  inline torch::Dtype dtype() const noexcept { return dtype_; }
  inline const torch::Device& device() const noexcept { return dev_; }

  // Non-owning base virtual address.
  inline VirPtr get_base_ptr() const noexcept { return vaddr_; }

  // Get the pool page_id for a mapped page-aligned byte offset.
  // Returns -1 if the offset is invalid or has no transferred page mapping.
  page_id_t get_phy_page_id(offset_t offset) const;

 private:
  bool valid_offset_(offset_t offset) const;

  VirPtr vaddr_;
  size_t size_;
  size_t page_size_;
  torch::Dtype dtype_;
  torch::Device dev_;

  // Maps page id -> PhysicalPage (page id = offset / page_size_)
  std::unordered_map<page_id_t, std::unique_ptr<PhysicalPage>> mapping_;

  // Reservation of externally selected pool pages.
  bool use_preallocated_pages_ = false;
  std::vector<page_id_t> preallocated_page_ids_;  // Stored for cleanup
  size_t mapped_preallocated_pages_ = 0;
};

}  // namespace xllm

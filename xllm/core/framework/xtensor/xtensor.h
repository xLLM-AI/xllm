/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <torch/torch.h>

#include <memory>
#include <unordered_map>
#include <vector>

#include "common/global_flags.h"
#include "common/macros.h"
#include "core/framework/config/kv_cache_config.h"
#include "phy_page.h"  // Includes page_id_t definition
#include "platform/vmm_api.h"

namespace xllm {

// Type definitions (page_id_t is defined in phy_page.h)
using offset_t = page_id_t;

/* NOTE: XTensorAllocator is thread-safe but XTensor is not. */
class XTensor {
 public:
  XTensor(size_t size, torch::Dtype dtype, torch::Device dev);

  // Constructor for weight tensor using pre-allocated page_ids (non-contiguous)
  // page_ids: physical page IDs from PhyPagePool (allocated via
  // allocate_pages_from_right)
  XTensor(const std::vector<page_id_t>& page_ids,
          torch::Dtype dtype,
          torch::Device dev);

  ~XTensor();

  bool map(offset_t offset);
  bool unmap(offset_t offset);

  // Map/unmap all pages (for weight tensors)
  bool map_all();
  bool unmap_all();

  // Map all pages using pre-allocated page_ids (for weight fallback)
  // page_ids: physical page IDs to use
  // Returns true on success
  bool map_with_page_ids(const std::vector<page_id_t>& page_ids);

  // Check if this XTensor uses pre-allocated pages (weight fallback mode)
  bool is_using_preallocated_pages() const { return use_preallocated_pages_; }

  // Allocate a chunk of memory from this tensor (bump allocator style)
  // Used for weight tensors where each layer allocates its own portion.
  // ptr: output parameter, set to the allocated memory address
  // size: size in bytes to allocate
  // Returns true on success, false on failure
  bool allocate(void*& ptr, size_t size);

  // Get the current allocation offset (for debugging/info)
  size_t alloc_offset() const noexcept { return alloc_offset_; }

  // Reset allocation offset (e.g., for reuse)
  void reset_alloc_offset() { alloc_offset_ = 0; }

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

  // Getters for compatibility
  inline torch::Dtype dtype() const noexcept { return dtype_; }
  inline const torch::Device& device() const noexcept { return dev_; }

  // Alias for vaddr() for backward compatibility
  inline VirPtr get_base_ptr() const noexcept { return vaddr_; }

  // Get the global physical page_id for a given offset within this XTensor.
  // Returns the page_id from PhyPagePool, or -1 if the offset is not mapped.
  // This is used for PD disaggregation to convert block offsets to
  // GlobalXTensor offsets.
  page_id_t get_phy_page_id(offset_t offset) const;

 private:
  // Map a single physical page at the given offset
  bool map_phy_page_(PhyPage* page, offset_t offset);

  VirPtr vaddr_;
  size_t size_;
  size_t
      page_size_;  // Page size
                   // (::xllm::KVCacheConfig::get_instance().phy_page_granularity_size())
  torch::Dtype dtype_;
  torch::Device dev_;

  // Maps page id -> PhyPage (page id = offset / page_size_)
  std::unordered_map<page_id_t, std::unique_ptr<PhyPage>> mapping_;

  // Bump allocator offset for weight allocation
  size_t alloc_offset_ = 0;

  // For weight fallback mode: use pre-allocated pages from PhyPagePool
  bool use_preallocated_pages_ = false;
  std::vector<page_id_t> preallocated_page_ids_;  // Stored for cleanup
};

}  // namespace xllm

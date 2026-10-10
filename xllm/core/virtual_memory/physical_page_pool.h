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

#include <deque>
#include <memory>
#include <mutex>
#include <vector>

#include "core/virtual_memory/physical_page.h"

namespace xllm {

/**
 * PhysicalPagePool manages a pool of pre-allocated physical pages.
 *
 * This is a singleton class that:
 * - Pre-allocates physical pages during initialization
 * - Each page has a unique page_id for tracking
 * - Provides get/put interface for MappedMemoryRegion to acquire/release
 * physical pages
 * - Avoids runtime allocation overhead during map operations
 */
class PhysicalPagePool final {
 public:
  // Get the global singleton instance
  static PhysicalPagePool& get_instance() {
    static PhysicalPagePool pool;
    return pool;
  }

  // Initialize the pool with specified number of pages
  // device: the device to allocate physical pages on
  // num_pages: number of physical pages to pre-allocate
  void init(const torch::Device& device, size_t num_pages, size_t page_size);

  // Release all pages and make the singleton reusable. Returns false when a
  // caller still owns pages from this pool.
  bool reset();

  // Check if initialized
  bool is_initialized() const { return initialized_; }

  // Get a physical page from the pool
  // Returns nullptr if pool is empty
  std::unique_ptr<PhysicalPage> get();

  // Get multiple physical pages from the pool in one lock (left-to-right)
  // Returns empty vector if not enough pages available
  // If partial allocation fails, all acquired pages are returned to pool
  std::vector<std::unique_ptr<PhysicalPage>> batch_get(size_t count);

  // Reserve contiguous pages by scanning from the right side.
  // Returns the starting page_id of the contiguous segment, or -1 if not found
  // The pages are marked as allocated but ownership remains in all_pages_
  page_id_t allocate_contiguous_from_right(size_t count);

  // Allocate non-contiguous pages from right side (fallback for fragmented
  // pool) Returns page_ids of allocated pages (may not be contiguous) Returns
  // empty vector if not enough pages available
  std::vector<page_id_t> allocate_pages_from_right(size_t count);

  // Free pages that were allocated via allocate_contiguous_from_right
  // or allocate_pages_from_right
  // page_ids: vector of page_ids to free
  void release_reserved_pages(const std::vector<page_id_t>& page_ids);

  // Put a physical page back to the pool
  void put(std::unique_ptr<PhysicalPage> page);

  // Put multiple physical pages back to the pool in one lock
  void batch_put(std::vector<std::unique_ptr<PhysicalPage>>& pages);

  // Get number of available pages in the pool
  size_t num_available() const;

  // Get total number of pages (available + in use)
  size_t num_total() const { return num_total_pages_; }
  size_t page_size() const { return page_size_; }

  // Get the device
  const torch::Device& device() const { return device_; }

  // ============== Shared Mapping Support ==============

  // Get non-owning page pointers in page_id order for shared mapping.
  // Pointers remain stable when ownership moves to a caller; every page
  // must stay alive while any mapping references it.
  const std::vector<PhysicalPage*>& get_all_pages() const;

 private:
  PhysicalPagePool() = default;
  ~PhysicalPagePool() = default;
  PhysicalPagePool(const PhysicalPagePool&) = delete;
  PhysicalPagePool& operator=(const PhysicalPagePool&) = delete;

  bool initialized_ = false;
  torch::Device device_{torch::kCPU};
  size_t num_total_pages_ = 0;
  size_t page_size_ = 0;

  mutable std::mutex mtx_;
  // Pages owned by the pool, indexed by page_id.
  // This owns the pages and provides O(1) lookup by page_id
  std::vector<std::unique_ptr<PhysicalPage>> all_pages_;

  // Stable non-owning page pointers, filled once at init.
  std::vector<PhysicalPage*> all_page_ptrs_;

  // Free page indices (page_ids of pages available for allocation)
  // Transferred pages are taken from the front of the free deque.
  std::deque<page_id_t> free_page_ids_;

  // Track which pages are allocated (for segment management)
  std::vector<bool> page_allocated_;
};

}  // namespace xllm

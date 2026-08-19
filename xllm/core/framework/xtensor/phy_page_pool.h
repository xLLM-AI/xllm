/* Copyright 2025-2026 The xLLM Authors.

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

#include <torch/torch.h>

#include <deque>
#include <memory>
#include <mutex>
#include <vector>

#include "phy_page.h"

namespace xllm {

/**
 * PhyPagePool manages a pool of pre-allocated physical pages.
 *
 * This is a singleton class that:
 * - Pre-allocates physical pages during initialization
 * - Each page has a unique page_id for tracking
 * - Provides get/put interface for XTensor to acquire/release physical pages
 * - Avoids runtime allocation overhead during map operations
 */
class PhyPagePool {
 public:
  // Get the global singleton instance
  static PhyPagePool& get_instance() {
    static PhyPagePool pool;
    return pool;
  }

  // Initialize the pool with specified number of pages
  // device: the device to allocate physical pages on
  // num_pages: number of physical pages to pre-allocate
  void init(const torch::Device& device, size_t num_pages);

  // Check if initialized
  bool is_initialized() const { return initialized_; }

  // Get a physical page from the pool
  // Returns nullptr if pool is empty
  std::unique_ptr<PhyPage> get();

  // Get multiple physical pages from the pool in one lock (left-to-right)
  // Returns empty vector if not enough pages available
  // If partial allocation fails, all acquired pages are returned to pool
  std::vector<std::unique_ptr<PhyPage>> batch_get(size_t count);

  // Find and allocate contiguous pages from right side (for weight allocation)
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
  void free_weight_pages(const std::vector<page_id_t>& page_ids);

  // Put a physical page back to the pool
  void put(std::unique_ptr<PhyPage> page);

  // Put multiple physical pages back to the pool in one lock
  void batch_put(std::vector<std::unique_ptr<PhyPage>>& pages);

  // Get number of available pages in the pool
  size_t num_available() const;

  // Get total number of pages (available + in use)
  size_t num_total() const { return num_total_pages_; }

  // Get the device
  const torch::Device& device() const { return device_; }

  // ============== Global XTensor Support ==============

  // Get all pages as raw pointers for GlobalXTensor mapping
  // Ownership remains with pool, pages are NOT marked as allocated
  const std::vector<PhyPage*>& get_all_pages() const;

 private:
  PhyPagePool() = default;
  ~PhyPagePool() = default;
  PhyPagePool(const PhyPagePool&) = delete;
  PhyPagePool& operator=(const PhyPagePool&) = delete;

  bool initialized_ = false;
  torch::Device device_{torch::kCPU};
  size_t num_total_pages_ = 0;

  mutable std::mutex mtx_;
  // All pages indexed by page_id (for jumbo xtensor)
  // This owns the pages and provides O(1) lookup by page_id
  std::vector<std::unique_ptr<PhyPage>> all_pages_;

  // Raw pointers to all pages (for GlobalXTensor, filled once at init)
  std::vector<PhyPage*> all_page_ptrs_;

  // Free page indices (page_ids of pages available for allocation)
  // For KV cache allocation, pages are taken from left to right.
  std::deque<page_id_t> free_page_ids_;

  // Track which pages are allocated (for segment management)
  std::vector<bool> page_allocated_;
};

}  // namespace xllm

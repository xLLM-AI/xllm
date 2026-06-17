/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/types.h"
#include "core/framework/config/kv_cache_config.h"
#include "virt_page.h"
#include "xtensor.h"  // For offset_t type definition

namespace xllm {

// Configuration constants
constexpr int32_t MIN_RESERVED_PAGES = 8;
constexpr int32_t MAX_RESERVED_PAGES = 32;
constexpr bool PAGE_PREALLOC_ENABLED = true;
constexpr double PREALLOC_THREAD_TIMEOUT = 2.0;  // seconds
/**
 * PageAllocator manages virtual page allocation for KV cache.
 *
 * Key concepts:
 * - VirtPage: Logical page for KV cache indexing, based on single-layer memory
 * - PhyPage: Physical memory page (2MB), managed by PhyPagePool
 *
 * Multi-model support:
 * - Each model has its own logical page_list (virtual pages)
 * - All models share physical pages (phy_pages)
 * - Model sleep: stops prealloc thread, unmaps and releases physical pages
 * - Model wakeup: restarts prealloc thread to refill physical pages
 *
 * Memory layout:
 * - For non-contiguous: each layer has its own K and V XTensor
 *   - mem_size_per_layer = total_phy_mem / (2 * num_layers)
 *   - num_virt_pages = mem_size_per_layer / virt_page_size
 *   - Allocating 1 virt_page consumes (2 * num_layers) phy_pages
 *
 * This is a singleton class shared by all XTensorBlockManagerImpl instances.
 */
class PageAllocator {
 public:
  // Get the global singleton instance
  static PageAllocator& get_instance() {
    static PageAllocator allocator;
    return allocator;
  }

  // Initialize the allocator (basic initialization)
  // num_phy_pages: total number of physical pages from PhyPagePool
  // dp_size: number of data parallel groups
  // max_world_size: maximum number of workers (for per-worker tracking)
  // enable_page_prealloc: whether to enable background preallocation
  void init(size_t num_phy_pages,
            int32_t dp_size = 1,
            int32_t max_world_size = 1,
            bool enable_page_prealloc = PAGE_PREALLOC_ENABLED);

  // Check if initialized
  bool is_initialized() const { return initialized_; }

  // ============ Multi-Model Management ============
  // Register a model with its layer count
  // model_id: unique identifier for the model (e.g. model name from options)
  // num_layers: number of transformer layers for this model
  // Returns true if registration successful
  bool register_model(const std::string& model_id,
                      int64_t num_layers,
                      MasterStatus master_status);

  // Put a model to sleep:
  // - Release weight pages (via free_weight_pages)
  // - Unmap all mapped KV cache virtual pages
  // - Release physical pages back to shared pool
  // - Stop preallocation for this model
  bool sleep_model(const std::string& model_id,
                   bool skip_weight_release = false);

  // Wake up a sleeping model:
  // - Re-map all previously mapped KV cache virtual pages
  // - Re-allocate weight pages (via alloc_weight_pages)
  // - Resume preallocation for this model
  bool wakeup_model(const std::string& model_id);

  // Check if a model is sleeping
  bool is_model_sleeping(const std::string& model_id) const;

  // Set model-specific parallel strategy (for fork master with different dp/tp)
  // This affects which workers are targeted during weight allocation
  void set_model_parallel_strategy(const std::string& model_id,
                                   int32_t dp_size,
                                   int32_t tp_size);

  // Get model-specific world_size (dp_size * tp_size)
  // Returns 0 if model not found or not set
  int32_t get_model_world_size(const std::string& model_id) const;

  // Get available physical pages for a specific model
  // This considers the model's world_size and returns the minimum free pages
  // among all workers that the model uses
  size_t get_free_phy_pages_for_model(const std::string& model_id) const;

  // Start preallocation thread (called after reserving null block)
  void start_prealloc_thread();

  // ============ KV Cache Page Allocation ============
  // Allocate a virtual page for KV cache
  // model_id: which model this allocation is for
  // dp_rank: which DP group this allocation is for
  // Consumes phy_pages_per_virt_page_ physical pages
  // Returns nullptr if no physical pages available
  std::unique_ptr<VirtPage> alloc_kv_cache_page(const std::string& model_id,
                                                int32_t dp_rank);

  // Free multiple KV cache virtual pages
  void free_kv_cache_pages(const std::string& model_id,
                           int32_t dp_rank,
                           const std::vector<int64_t>& virt_page_ids);

  // Trim reserved KV cache pages (unmap physical pages)
  void trim_kv_cache(const std::string& model_id, int32_t dp_rank);

  // ============ Weight Page Allocation ============
  // Allocate physical pages for weight tensor (full map)
  // model_id: which model this allocation is for
  // num_pages: number of physical pages (aligned up from weight size)
  // All-or-nothing: returns true if all pages allocated, false otherwise
  bool alloc_weight_pages(const std::string& model_id, size_t num_pages);

  // Free physical pages from weight tensor
  // model_id: which model
  // num_pages: same count used in alloc_weight_pages
  bool free_weight_pages(const std::string& model_id, size_t num_pages);

  // Get number of weight pages allocated for a model (not cleared on free)
  size_t get_weight_pages_allocated(const std::string& model_id) const;

  // Set weight pages count (for
  // MasterStatus::LIGHT_SLEEP/MasterStatus::DEEP_SLEEP mode without physical
  // allocation)
  void set_weight_pages_count(const std::string& model_id, size_t num_pages);

  // Virtual page getters (for specific model and DP group)
  size_t get_num_free_virt_pages(const std::string& model_id,
                                 int32_t dp_rank) const;
  size_t get_num_inuse_virt_pages(const std::string& model_id,
                                  int32_t dp_rank) const;
  size_t get_num_total_virt_pages(const std::string& model_id) const;
  size_t get_num_reserved_virt_pages(const std::string& model_id,
                                     int32_t dp_rank) const;

  // Physical page getters (shared across all models)
  size_t get_num_free_phy_pages() const;
  size_t get_num_total_phy_pages() const;

  // Get free pages for each worker (for etcd registration)
  // Returns a vector where index i = num_total_phy_pages -
  // worker_pages_used_[i]
  std::vector<size_t> get_all_worker_free_pages() const;

  // Convert block_id to virt_page_id
  int64_t get_virt_page_id(int64_t block_id, size_t block_mem_size) const;

  // Get offset for XTensor map/unmap (based on single-layer)
  offset_t get_offset(int64_t virt_page_id) const;

  // Get configuration
  size_t page_size() const { return page_size_; }
  int64_t num_layers(const std::string& model_id) const;

  // Get number of physical pages consumed per virtual page allocation
  size_t phy_pages_per_virt_page(const std::string& model_id) const;

 private:
  PageAllocator() = default;
  ~PageAllocator();
  PageAllocator(const PageAllocator&) = delete;
  PageAllocator& operator=(const PageAllocator&) = delete;

  // Per-DP group virtual page tracking
  struct DpGroupPages {
    size_t num_free_virt_pages{0};            // Protected by mtx_
    std::deque<int64_t> free_virt_page_list;  // Unmapped virtual pages
    std::deque<int64_t>
        reserved_virt_page_list;  // Mapped virtual pages ready for use
    std::unordered_set<int64_t>
        allocated_virt_page_list;  // Mapped virtual pages in use by block mgr
  };

  // Per-model state
  struct ModelState {
    int64_t num_layers = 0;
    size_t num_total_virt_pages = 0;
    size_t phy_pages_per_virt_page = 0;
    size_t weight_pages_allocated = 0;  // Not cleared on free, used for wakeup
    bool is_sleeping = false;
    // Count of pending map operations (for safe sleep)
    std::atomic<int> pending_map_ops{0};
    std::vector<DpGroupPages> dp_group_pages;
    // Model-specific parallel strategy (for fork master with different dp/tp)
    int32_t model_dp_size = 0;     // 0 means use global dp_size_
    int32_t model_tp_size = 0;     // 0 means use global tp_size
    int32_t model_world_size = 0;  // = dp_size * tp_size, 0 means use global
  };

  // Check if enough physical pages available for a specific DP group
  // model_id: which model (to get tp_size for worker range calculation)
  // dp_rank: which DP group
  bool has_enough_phy_pages_for_dp(const std::string& model_id,
                                   int32_t dp_rank,
                                   size_t num_phy_pages) const;

  // Consume/release physical pages for a specific DP group (update tracking)
  // Returns false if not enough physical pages available
  bool consume_phy_pages_for_dp(const std::string& model_id,
                                int32_t dp_rank,
                                size_t num_phy_pages);
  void release_phy_pages_for_dp(const std::string& model_id,
                                int32_t dp_rank,
                                size_t num_phy_pages);

  // Get worker range for a DP group [start, end)
  // Returns {start_worker, end_worker}
  std::pair<int32_t, int32_t> get_dp_group_worker_range(
      const std::string& model_id,
      int32_t dp_rank) const;

  // Get minimum free pages among workers in a range [start, end)
  size_t get_min_free_pages_in_range(int32_t start_worker,
                                     int32_t end_worker) const;

  // Preallocation worker thread function
  void prealloc_worker();

  // Start/stop preallocation thread
  void start_prealloc_thread_internal();
  void stop_prealloc_thread(double timeout = PREALLOC_THREAD_TIMEOUT);

  // Trigger preallocation
  void trigger_preallocation();

  // Map/unmap virtual pages (broadcasts to workers in dp_rank group)
  // Returns false if broadcast fails
  bool map_virt_pages(const std::string& model_id,
                      int32_t dp_rank,
                      const std::vector<int64_t>& virt_page_ids);
  bool unmap_virt_pages(const std::string& model_id,
                        int32_t dp_rank,
                        const std::vector<int64_t>& virt_page_ids);

  // Update memory usage tracking
  void update_memory_usage();

  // Get model state (throws if not found)
  ModelState& get_model_state(const std::string& model_id);
  const ModelState& get_model_state(const std::string& model_id) const;

  // Initialization state
  bool initialized_ = false;

  // Configuration
  int32_t dp_size_ = 1;
  size_t page_size_ =
      0;  // Page size (from
          // ::xllm::KVCacheConfig::get_instance().phy_page_granularity_size())
  bool enable_page_prealloc_ = PAGE_PREALLOC_ENABLED;

  // Physical page tracking (shared across all models)
  size_t num_total_phy_pages_ = 0;  // Total physical pages per worker

  // Per-worker physical page tracking
  // Each worker has independent PhyPagePool with the same total pages.
  // worker_pages_used_[i] = total pages used by worker i (weight + KV cache)
  // This tracks both weight allocation (by model world_size) and
  // KV cache allocation (by DP group's workers)
  std::vector<size_t> worker_pages_used_;
  int32_t max_world_size_ =
      0;  // Maximum number of workers (from initial nnodes)

  // Per-model state (key is model_id from options)
  std::unordered_map<std::string, ModelState> model_states_;

  // Reserved page limits
  int32_t min_reserved_pages_ = MIN_RESERVED_PAGES;
  int32_t max_reserved_pages_ = MAX_RESERVED_PAGES;

  // Threading
  mutable std::mutex mtx_;
  std::condition_variable cond_;
  std::atomic<bool> prealloc_running_{false};
  std::atomic<bool> prealloc_needed_{false};
  std::unique_ptr<std::thread> prealloc_thd_;
};

}  // namespace xllm

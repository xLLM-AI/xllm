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
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "core/framework/allocator/virtual_memory/physical_page.h"
#include "core/framework/allocator/vmm_memory_budget.h"

namespace xllm {

class SharedPhysicalPageMapping;

enum class PhysicalPagePoolFailureReason {
  kNone,
  kLogicalCapacity,
  kLogicalAccountingMismatch,
  kFreeMemoryQueryFailed,
  kRequestSizeOverflow,
  kInsufficientFreeMemory,
  kPhysicalPageCreation,
  kPostAllocationMemoryQueryFailed,
  kPostAllocationHeadroom,
  kSharedPageMapping,
};

const char* physical_page_pool_failure_reason_name(
    PhysicalPagePoolFailureReason reason);

struct PhysicalPageBatchResult {
  std::vector<std::unique_ptr<PhysicalPage>> pages;
  PhysicalPagePoolFailureReason failure_reason =
      PhysicalPagePoolFailureReason::kNone;
};

// Owns stable logical page slots and their optional physical backing. Eager
// initialization preserves fixed-pool callers; lazy initialization commits
// backing atomically when KV pages or weight reservations need it.
class PhysicalPagePool final {
 public:
  // Get the global singleton instance
  static PhysicalPagePool& get_instance() {
    static PhysicalPagePool pool;
    return pool;
  }

  // Initialize the pool with specified number of pages
  // device: the device to allocate physical pages on
  // num_pages: stable logical capacity, independent of physical residency
  // lazy reserves stable logical slots without allocating device memory.
  // The default preserves eager allocation for existing callers.
  void init(const torch::Device& device,
            size_t num_pages,
            size_t page_size,
            bool lazy = false);
  // Reserve num_pages stable slots, activating initial_num_pages at startup.
  // Lazy allocations activate further slots only after backing succeeds.
  void init(const torch::Device& device,
            size_t num_pages,
            size_t initial_num_pages,
            size_t page_size,
            bool lazy);

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
  // Return pages and the reason when this request cannot be satisfied.
  PhysicalPageBatchResult batch_get_with_reason(size_t count);

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

  // Number of activated unoccupied slots, including uncommitted backing.
  size_t num_available() const;

  // Stable logical capacity; growth and reclamation never renumber these slots
  size_t num_total() const { return num_total_pages_; }
  // Number of logical slots currently activated for allocation.
  size_t num_active() const;
  size_t num_committed() const;
  // Snapshot of idle backing plus growth allowed by current device memory.
  size_t num_allocatable() const;

  // Leave this many currently free device bytes outside new pool allocations.
  // Existing resident pages are not counted again against the free snapshot.
  void set_allocation_headroom(size_t bytes);
  // Keep this many device bytes free to enforce a configured utilization cap.
  void set_growth_reserve(size_t bytes);

  // Observe execution allocator residency/peaks and driver availability.
  // Pool growth is excluded from the fallback driver low watermark.
  void observe_execution_memory();

  // Release idle backing while retaining keep_pages resident idle pages.
  // Callers must return pages only after their device/transfer users finish.
  // Active shared-mapping leases pin the backing and make trimming a no-op.
  size_t trim_idle(size_t keep_pages = 0);
  size_t page_size() const { return page_size_; }

  // Get the device
  const torch::Device& device() const { return device_; }

  // ============== Shared Mapping Support ==============

  // Get non-owning page pointers in page_id order for shared mapping.
  // Uncommitted slots are null. Hold the corresponding KV ownership or weight
  // reservation while dereferencing a snapshot; idle backing can be reclaimed.
  std::vector<PhysicalPage*> get_all_pages() const;

  // Serialize alias publication and leases with physical growth/reclamation.
  // The mapping reserves all logical slots but maps only resident pages.
  void attach_shared_mapping(
      const std::shared_ptr<SharedPhysicalPageMapping>& mapping);
  bool detach_shared_mapping(const SharedPhysicalPageMapping* mapping);
  std::shared_ptr<void> acquire_shared_mapping_lease(
      bool require_contiguous = false);

 private:
  friend class PhysicalPagePoolTestPeer;

  bool materialize_pages_locked(
      const std::vector<page_id_t>& page_ids,
      PhysicalPagePoolFailureReason* failure_reason = nullptr);
  void activate_pages_locked(const std::vector<page_id_t>& page_ids);
  size_t allocation_headroom_locked(
      size_t free_bytes,
      std::optional<AllocatorMemoryStats> allocator_stats) const;
  void reserve_pages_locked(const std::vector<page_id_t>& page_ids);

  PhysicalPagePool() = default;
  ~PhysicalPagePool() = default;
  PhysicalPagePool(const PhysicalPagePool&) = delete;
  PhysicalPagePool& operator=(const PhysicalPagePool&) = delete;

  bool initialized_ = false;
  bool lazy_ = false;
  torch::Device device_{torch::kCPU};
  size_t num_total_pages_ = 0;
  size_t num_active_pages_ = 0;
  size_t num_allocated_pages_ = 0;
  size_t page_size_ = 0;
  size_t num_committed_pages_ = 0;
  size_t allocation_headroom_bytes_ = 0;
  size_t growth_reserve_bytes_ = 0;
  VmmMemoryBudget execution_memory_budget_;

  mutable std::mutex mtx_;
  // Pages owned by the pool, indexed by page_id.
  // This owns the pages and provides O(1) lookup by page_id
  std::vector<std::unique_ptr<PhysicalPage>> all_pages_;

  // Stable logical slots. A null pointer denotes uncommitted backing; live
  // page IDs and addresses never move when other slots grow or shrink.
  std::vector<PhysicalPage*> all_page_ptrs_;

  // A reserved arena can have inactive slots that are activated on demand.
  std::vector<bool> page_active_;

  // Free page indices (page_ids of pages available for allocation)
  // Transferred pages are taken from the front of the free deque.
  std::deque<page_id_t> free_page_ids_;

  // Track which pages are allocated (for segment management)
  std::vector<bool> page_allocated_;
  std::weak_ptr<SharedPhysicalPageMapping> shared_mapping_;

  // Only the slow physical-growth path uses these boundaries. Tests can inject
  // deterministic memory snapshots and partial allocation failures.
  std::function<int64_t()> free_memory_query_;
  std::function<std::optional<AllocatorMemoryStats>()> allocator_memory_query_;
  std::function<std::unique_ptr<PhysicalPage>(page_id_t)> page_factory_;
};

}  // namespace xllm

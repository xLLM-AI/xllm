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

#include <cstddef>
#include <mutex>
#include <vector>

#include "vmm_api.h"

namespace xllm {

/**
 * @brief SharedVMMAllocator - A shared VMM buffer allocator for CUDA Graph
 * memory reuse with multi-VA physical memory sharing.
 *
 * This allocator manages physical memory that can be mapped to multiple
 * virtual address spaces. Each CUDA Graph capture uses a new virtual address
 * space, but all spaces share the same physical memory. This achieves:
 * - Physical memory usage = max(shape) instead of sum(shape)
 * - No address conflicts with PyTorch's block management
 * - Stable virtual addresses within each capture (critical for CUDA Graph)
 *
 * Key features:
 * - Multiple virtual address spaces mapped to shared physical memory
 * - Physical memory extended on demand
 * - Thread-safe allocation and virtual space switching
 */
class SharedVMMAllocator {
 public:
  SharedVMMAllocator() = default;
  ~SharedVMMAllocator();

  // Non-copyable and non-movable
  SharedVMMAllocator(const SharedVMMAllocator&) = delete;
  SharedVMMAllocator& operator=(const SharedVMMAllocator&) = delete;
  SharedVMMAllocator(SharedVMMAllocator&&) = delete;
  SharedVMMAllocator& operator=(SharedVMMAllocator&&) = delete;

  /**
   * @brief Initialize the allocator by reserving virtual address space.
   * @param device_id The device ID to allocate memory on.
   * @param reserve_size Optional: size of virtual address space to reserve.
   *                     If 0, defaults to 1.125x device total memory.
   */
  void init(int32_t device_id, size_t reserve_size = 0);

  /**
   * @brief Switch to a new virtual address space for the next capture.
   * Creates a new virtual address space and maps existing physical memory
   * to it. This allows different captures to use different virtual addresses
   * while sharing the same physical memory.
   */
  void switch_to_new_virtual_space();

  /**
   * @brief Reset the allocation pointer to the beginning (deprecated).
   * Use switch_to_new_virtual_space() instead to avoid address conflicts.
   */
  void reset_allocation_pointer();

  /**
   * @brief Allocate memory from the VMM buffer.
   * @param size The size of memory to allocate.
   * @return Pointer to the allocated memory.
   */
  void* allocate(size_t size);

  /**
   * @brief Deallocate memory (no-op for VMM allocator).
   * Memory is not actually freed; it will be reused after reset.
   * @param ptr Pointer to deallocate (unused).
   */
  void deallocate(void* ptr);

  /**
   * @brief Get the high water mark (maximum allocated offset).
   */
  size_t high_water_mark() const { return high_water_mark_; }

  /**
   * @brief Get the current allocation offset.
   */
  size_t current_offset() const { return current_offset_; }

  /**
   * @brief Get the total mapped physical memory size.
   */
  size_t mapped_size() const { return mapped_size_; }

  /**
   * @brief Get the reserved virtual address space size.
   */
  size_t reserved_size() const { return reserved_size_; }

  /**
   * @brief Check if the allocator is initialized.
   */
  bool is_initialized() const { return initialized_; }

 private:
  /**
   * @brief Structure to track a virtual address space.
   */
  struct VirtualSpace {
    VirPtr base_ptr;       // Base virtual address
    size_t reserved_size;  // Size of reserved virtual address space
    size_t mapped_size;    // Size of physical memory mapped to this space
  };

  /**
   * @brief Extend the physical memory mapping to accommodate new allocations.
   * @param new_size The minimum required mapped size.
   */
  void extend_mapping(size_t new_size);

  /**
   * @brief Map physical memory to a virtual address space.
   * @param space The virtual space to map to.
   * @param start_handle_idx Start index in handles_ to map from.
   */
  void map_physical_to_virtual(VirtualSpace& space, size_t start_handle_idx);

  /**
   * @brief Get the granularity size for physical memory pages.
   */
  size_t get_granularity() const;

  bool initialized_ = false;
  int32_t device_id_ = 0;       // Device ID
  size_t reserved_size_ = 0;    // Size for each virtual address space
  size_t current_offset_ = 0;   // Current allocation pointer
  size_t high_water_mark_ = 0;  // Maximum allocation offset (physical)
  size_t granularity_ = 0;      // Physical page granularity size

  // Physical memory (shared across all virtual spaces)
  std::vector<PhyMemHandle> handles_;  // Physical memory handles
  size_t mapped_size_ = 0;             // Total mapped physical memory size

  // Virtual address spaces (each maps to the same physical memory)
  std::vector<VirtualSpace> virtual_spaces_;
  size_t current_space_index_ = 0;  // Index of current virtual space

  mutable std::mutex mutex_;  // Thread safety
};

}  // namespace xllm

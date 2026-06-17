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

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include "common/macros.h"
#include "framework/block/block_manager.h"
// #include "page_allocator.h"
#include "virt_page.h"

namespace xllm {

/**
 * XTensorBlockManagerImpl is a BlockManager implementation that uses
 * virtual memory management (VMM) for KV cache allocation.
 *
 * This implementation follows the kvcached KVCacheManager design:
 * - Pages: Large memory chunks (e.g., 2MB) that are mapped/unmapped to physical
 * memory
 * - Blocks: Smaller units within pages that are allocated to store KV cache
 * data
 *
 * Key features:
 * - Uses PageAllocator for physical page management
 * - Maps blocks to virtual pages
 * - Supports reserved blocks for pre-allocation
 * - Does NOT support prefix cache (prefix cache disabled)
 */
class XTensorBlockManagerImpl : public BlockManager {
 public:
  explicit XTensorBlockManagerImpl(const Options& options,
                                   int64_t num_layers,
                                   size_t block_mem_size,
                                   size_t page_size,
                                   int32_t dp_rank = 0,
                                   const std::string& model_id = "");
  ~XTensorBlockManagerImpl() override;

  // Allocate num_blocks blocks
  std::vector<Block> allocate(size_t num_blocks) override;

  // Deallocate blocks
  void deallocate(const Slice<Block>& blocks) override;

  // Allocate shared blocks (prefix cache not supported)
  std::vector<Block> allocate_shared(
      const Slice<int32_t>& token_ids,
      const Slice<Block>& existed_shared_blocks = {},
      const MMData& mm_data = MMData()) override;

  // Cache blocks (prefix cache not supported)
  void cache(const Slice<int32_t>& token_ids,
             std::vector<Block>& blocks,
             size_t existed_shared_blocks_num = 0,
             const MMData& mm_data = MMData()) override;
  void cache(const std::vector<Block>& blocks) override;

  // Get merged KV cache event
  void get_merged_kvcache_event(KvCacheEvent* event) const override;

  // Get number of blocks in prefix cache (always 0, not supported)
  size_t num_blocks_in_prefix_cache() const override { return 0; }

  // Get number of free blocks
  size_t num_free_blocks() const override;

  // Get number of used blocks
  size_t num_used_blocks() const override;

  // Get KV cache utilization
  double kv_cache_utilization() const override;

  // Free a single block by id
  void free(int32_t block_id) override;

  // Allocate a single block
  Block allocate() override;

  // Get total number of blocks
  size_t num_total_blocks() const override;

  // Get available size (free blocks that can be allocated)
  size_t available_size() const;

  // Try to reserve blocks for future allocation
  bool try_to_reserve(size_t need_size);

  // Free all reserved blocks
  void free_reserved();

  // Trim unused reserved pages
  void trim();

  // Get mapped memory size in bytes
  size_t get_mapped_memory_size() const;

  // Reserve padding block for padding tokens.
  // Should be called after KV tensors are created.
  void reserve_xtensor_padding_blocks();

 private:
  DISALLOW_COPY_AND_ASSIGN(XTensorBlockManagerImpl);

  // Internal allocation
  std::vector<int32_t> alloc_internal(size_t need_size);

  // Free blocks by their indices
  void free_blocks(const std::vector<int32_t>& indices);

  // Get number of allocated blocks
  size_t get_num_allocated_blocks() const;

  // Internal available_size (caller must hold mtx_)
  size_t available_size_internal() const;

 private:
  // Model ID (from options_.model_id())
  std::string model_id_;

  // Data parallel rank
  int32_t dp_rank_;

  // Number of layers
  int64_t num_layers_;

  // Page size
  size_t page_size_;

  // Block memory size in bytes
  size_t block_mem_size_;

  // Number of available blocks in avail_pages
  std::atomic<size_t> num_avail_blocks_;

  // Pages with free blocks (page_id -> VirtPage)
  std::unordered_map<int64_t, std::unique_ptr<VirtPage>> avail_pages_;

  // Pages that are fully allocated (page_id -> VirtPage)
  std::unordered_map<int64_t, std::unique_ptr<VirtPage>> full_pages_;

  // Reserved blocks for pre-allocation
  std::vector<int32_t> reserved_blocks_;

  // Padding block id (block 0, reserved for padding)
  std::optional<int32_t> padding_block_;

  // Mutex for thread safety
  mutable std::mutex mtx_;
};

}  // namespace xllm

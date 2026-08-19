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
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace xllm {

/**
 * VirtPage class represents a virtual memory page that can contain multiple
 * blocks. A page manages a range of blocks and tracks which blocks are free.
 */
class VirtPage {
 public:
  VirtPage(int64_t page_id, size_t page_size);
  ~VirtPage() = default;

  // Initialize the page with block memory size
  void init(size_t block_mem_size);

  // Allocate blocks from this page
  std::vector<int64_t> alloc(size_t num_blocks = 1);

  // Free a single block
  void free(int64_t block_id);

  // Free multiple blocks
  void free_batch(const std::vector<int64_t>& block_ids);

  // Check if page is empty (all blocks are free)
  bool empty() const;

  // Check if page is full (no free blocks)
  bool full() const;

  // Get number of free blocks
  size_t num_free_blocks() const;

  // Get list of free blocks
  const std::vector<int64_t>& get_free_blocks() const;

  // Get page id
  int64_t page_id() const { return page_id_; }

  // Get page size
  size_t page_size() const { return page_size_; }

  // Static utility functions
  /**
   * Get the block range of a page.
   * The page contains [start_block, end_block), which handles the case where
   * page_size is not divisible by block_mem_size.
   * For example, if page_size = 16 and block_mem_size = 6, the page 0
   * contains [0, 2) blocks, and the page 1 contains [3, 5) blocks.
   * Pages:  |      0-16       |        16-32        |
   *         | 0-6 | 6-12 | 12-18 | 18-24 | 24-30 | 30-32 |
   * Blocks: |  0  |  1   |2<skip>|   3   |   4   |5<skip>|
   */
  static std::pair<int64_t, int64_t> get_block_range(int64_t page_id,
                                                     size_t page_size,
                                                     size_t block_mem_size);

  /**
   * Calculate the number of blocks that can fit in a page.
   * This calculation is accurate even when page_size is not divisible by
   * block_mem_size.
   */
  static size_t get_num_blocks(size_t page_size, size_t block_mem_size);

 private:
  void require_init() const;

  int64_t page_id_;
  size_t page_size_;

  std::optional<int64_t> start_block_;
  std::optional<int64_t> end_block_;
  std::optional<size_t> num_kv_blocks_;
  std::vector<int64_t> free_list_;
};

}  // namespace xllm

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

#include "core/kv_cache/block/virtual_kv_page_state.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace xllm {
namespace {

void require_valid_geometry(size_t page_size, size_t block_mem_size) {
  CHECK_GT(page_size, 0) << "Page size must be positive";
  CHECK_GT(block_mem_size, 0) << "Block memory size must be positive";
  CHECK_LE(block_mem_size, page_size) << "KV blocks must fit within a page";
}

}  // namespace

VirtualKVPageState::VirtualKVPageState(int64_t page_id, size_t page_size)
    : page_id_(page_id), page_size_(page_size) {}

void VirtualKVPageState::require_init() const {
  CHECK(start_block_.has_value()) << "VirtualKVPageState not initialized";
  CHECK(end_block_.has_value()) << "VirtualKVPageState not initialized";
  CHECK(num_kv_blocks_.has_value()) << "VirtualKVPageState not initialized";
}

void VirtualKVPageState::require_allocated(int64_t block_id) const {
  CHECK_GE(block_id, *start_block_)
      << "Block " << block_id << " does not belong to page " << page_id_;
  CHECK_LT(block_id, *end_block_)
      << "Block " << block_id << " does not belong to page " << page_id_;
  CHECK(allocated_blocks_[static_cast<size_t>(block_id - *start_block_)])
      << "Block " << block_id << " is not allocated on page " << page_id_;
}

void VirtualKVPageState::init(size_t block_mem_size) {
  CHECK(!start_block_.has_value()) << "VirtualKVPageState already initialized";
  auto [start, end] = get_block_range(page_id_, page_size_, block_mem_size);
  start_block_ = start;
  end_block_ = end;
  num_kv_blocks_ = end - start;

  free_list_.clear();
  free_list_.reserve(*num_kv_blocks_);
  allocated_blocks_.assign(*num_kv_blocks_, 0);
  for (int64_t i = start; i < end; ++i) {
    free_list_.push_back(i);
  }
}

std::vector<int64_t> VirtualKVPageState::alloc(size_t num_blocks) {
  require_init();
  if (full()) {
    throw std::runtime_error("VirtualKVPageState " + std::to_string(page_id_) +
                             " is already full");
  }

  size_t actual_alloc = std::min(num_blocks, free_list_.size());
  std::vector<int64_t> block_ids(free_list_.begin(),
                                 free_list_.begin() + actual_alloc);
  for (int64_t block_id : block_ids) {
    allocated_blocks_[static_cast<size_t>(block_id - *start_block_)] = 1;
  }
  free_list_.erase(free_list_.begin(), free_list_.begin() + actual_alloc);
  return block_ids;
}

void VirtualKVPageState::free(int64_t block_id) {
  require_init();
  require_allocated(block_id);
  free_list_.push_back(block_id);
  allocated_blocks_[static_cast<size_t>(block_id - *start_block_)] = 0;
}

void VirtualKVPageState::free_batch(const std::vector<int64_t>& block_ids) {
  require_init();
  if (block_ids.empty()) {
    return;
  }

  std::vector<int64_t> sorted_block_ids(block_ids);
  std::sort(sorted_block_ids.begin(), sorted_block_ids.end());
  CHECK(std::adjacent_find(sorted_block_ids.begin(), sorted_block_ids.end()) ==
        sorted_block_ids.end())
      << "Duplicate blocks in free batch for page " << page_id_;
  for (int64_t block_id : block_ids) {
    require_allocated(block_id);
  }

  free_list_.insert(free_list_.end(), block_ids.begin(), block_ids.end());
  for (int64_t block_id : block_ids) {
    allocated_blocks_[static_cast<size_t>(block_id - *start_block_)] = 0;
  }
}

bool VirtualKVPageState::empty() const {
  require_init();
  return free_list_.size() == *num_kv_blocks_;
}

bool VirtualKVPageState::full() const {
  require_init();
  return free_list_.empty();
}

size_t VirtualKVPageState::num_free_blocks() const {
  require_init();
  return free_list_.size();
}

const std::vector<int64_t>& VirtualKVPageState::get_free_blocks() const {
  require_init();
  return free_list_;
}

std::pair<int64_t, int64_t> VirtualKVPageState::get_block_range(
    int64_t page_id,
    size_t page_size,
    size_t block_mem_size) {
  require_valid_geometry(page_size, block_mem_size);
  CHECK_GE(page_id, 0) << "Page ID must be nonnegative";
  uint64_t page_index = static_cast<uint64_t>(page_id);
  CHECK_LE(page_index + 1, std::numeric_limits<size_t>::max() / page_size)
      << "Page byte range overflows";
  size_t page_start = static_cast<size_t>(page_index) * page_size;
  size_t page_end = static_cast<size_t>(page_index + 1) * page_size;
  size_t start_block = page_start / block_mem_size;
  if (page_start % block_mem_size != 0) {
    ++start_block;
  }
  size_t end_block = page_end / block_mem_size;
  CHECK_LE(static_cast<uint64_t>(end_block),
           static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      << "Page block range exceeds block ID limit";
  return {static_cast<int64_t>(start_block), static_cast<int64_t>(end_block)};
}

size_t VirtualKVPageState::get_num_blocks(size_t page_size,
                                          size_t block_mem_size) {
  require_valid_geometry(page_size, block_mem_size);
  return page_size / block_mem_size;
}

}  // namespace xllm

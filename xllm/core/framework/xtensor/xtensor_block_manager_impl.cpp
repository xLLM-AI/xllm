/* Copyright 2025-2026 The xLLM Authors.

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

#include "xtensor_block_manager_impl.h"

#include <glog/logging.h>

#include <algorithm>
#include <chrono>

#include "common/global_flags.h"
#include "page_allocator.h"
#include "phy_page_pool.h"
#include "xtensor_allocator.h"

namespace xllm {

XTensorBlockManagerImpl::XTensorBlockManagerImpl(const Options& options,
                                                 int64_t num_layers,
                                                 size_t block_mem_size,
                                                 size_t page_size,
                                                 int32_t dp_rank,
                                                 const std::string& model_id)
    : BlockManager(options),
      model_id_(model_id),
      dp_rank_(dp_rank),
      num_layers_(num_layers),
      page_size_(page_size),
      block_mem_size_(block_mem_size),
      num_avail_blocks_(0) {
  LOG(INFO) << "XTensorBlockManagerImpl initialized: "
            << "model_id=" << model_id_ << ", dp_rank=" << dp_rank_
            << ", num_layers=" << num_layers_ << ", page_size=" << page_size_
            << ", block_size=" << options.block_size()
            << ", block_mem_size=" << block_mem_size_
            << ", num_blocks=" << options.num_blocks();
}

void XTensorBlockManagerImpl::reserve_xtensor_padding_blocks() {
  // Reserve the first block as padding block for padding tokens
  auto padding_blocks = alloc_internal(1);
  if (padding_blocks.empty() || padding_blocks[0] != 0) {
    LOG(FATAL) << "Failed to reserve padding block, got "
               << (padding_blocks.empty() ? -1 : padding_blocks[0]);
  }
  padding_block_ = padding_blocks[0];
}

XTensorBlockManagerImpl::~XTensorBlockManagerImpl() {
  // Release all pages
  std::lock_guard<std::mutex> lock(mtx_);
  std::vector<int64_t> page_ids;
  for (const auto& [page_id, page] : avail_pages_) {
    page_ids.push_back(page_id);
  }
  for (const auto& [page_id, page] : full_pages_) {
    page_ids.push_back(page_id);
  }
  avail_pages_.clear();
  full_pages_.clear();

  if (!page_ids.empty() && PageAllocator::get_instance().is_initialized()) {
    PageAllocator::get_instance().free_kv_cache_pages(
        model_id_, dp_rank_, page_ids);
  }
}

std::vector<int32_t> XTensorBlockManagerImpl::alloc_internal(size_t need_size) {
  std::lock_guard<std::mutex> lock(mtx_);

  size_t avail = available_size_internal();
  if (avail < need_size) {
    LOG(WARNING) << "available_size()=" << avail
                 << " < need_size=" << need_size;
  }

  std::vector<int32_t> ret_index;
  ret_index.reserve(need_size);

  size_t remaining_need = need_size;

  // Try to allocate from reserved blocks first
  if (!reserved_blocks_.empty()) {
    size_t num_from_reserved =
        std::min(reserved_blocks_.size(), remaining_need);
    ret_index.insert(ret_index.end(),
                     reserved_blocks_.begin(),
                     reserved_blocks_.begin() + num_from_reserved);
    reserved_blocks_.erase(reserved_blocks_.begin(),
                           reserved_blocks_.begin() + num_from_reserved);
    remaining_need -= num_from_reserved;
  }

  // Allocate the remaining blocks from pages
  while (remaining_need > 0) {
    VirtPage* page = nullptr;

    if (avail_pages_.empty()) {
      // Allocate a new page for this DP group
      auto new_page = PageAllocator::get_instance().alloc_kv_cache_page(
          model_id_, dp_rank_);
      if (new_page == nullptr) {
        LOG(ERROR) << "Failed to allocate new page for dp_rank=" << dp_rank_;
        // Return what we have allocated so far (caller should handle partial
        // allocation)
        return ret_index;
      }
      new_page->init(block_mem_size_);
      num_avail_blocks_ += new_page->num_free_blocks();
      int64_t page_id = new_page->page_id();
      avail_pages_[page_id] = std::move(new_page);
      page = avail_pages_[page_id].get();
    } else {
      // Get a page from avail_pages
      auto it = avail_pages_.begin();
      page = it->second.get();
    }

    size_t num_from_page = std::min(page->num_free_blocks(), remaining_need);
    auto allocated_indices = page->alloc(num_from_page);

    for (int64_t idx : allocated_indices) {
      ret_index.push_back(static_cast<int32_t>(idx));
    }

    if (page->full()) {
      // Move page from avail_pages to full_pages
      int64_t page_id = page->page_id();
      full_pages_[page_id] = std::move(avail_pages_[page_id]);
      avail_pages_.erase(page_id);
    }

    num_avail_blocks_ -= num_from_page;
    remaining_need -= num_from_page;
  }
  return ret_index;
}

std::vector<Block> XTensorBlockManagerImpl::allocate(size_t num_blocks) {
  auto indices = alloc_internal(num_blocks);
  std::vector<Block> blocks;
  blocks.reserve(indices.size());
  for (int32_t idx : indices) {
    blocks.emplace_back(idx, this);
  }
  return blocks;
}

std::optional<std::vector<Block>>
XTensorBlockManagerImpl::allocate_for_sequence(Sequence* seq,
                                               size_t num_tokens) {
  if (seq == nullptr) {
    return std::nullopt;
  }
  const size_t block_size = options_.block_size();
  if (block_size == 0) {
    return std::vector<Block>{};
  }
  const size_t held = seq->kv_state().num_blocks(block_type());
  const size_t num_blocks_needed = (num_tokens + block_size - 1) / block_size;
  if (num_blocks_needed <= held) {
    return std::vector<Block>{};
  }
  const size_t num_additional = num_blocks_needed - held;
  std::vector<Block> blocks = allocate(num_additional);  // VMM page alloc
  if (blocks.size() != num_additional) {
    return std::nullopt;
  }
  return blocks;
}

Block XTensorBlockManagerImpl::allocate() {
  auto blocks = allocate(1);
  if (blocks.empty()) {
    return Block();  // Invalid block
  }
  return std::move(blocks[0]);
}

void XTensorBlockManagerImpl::free_blocks(const std::vector<int32_t>& indices) {
  std::lock_guard<std::mutex> lock(mtx_);

  if (indices.empty()) {
    return;
  }

  auto& page_allocator = PageAllocator::get_instance();

  // Group indices by page_id
  std::unordered_map<int64_t, std::vector<int64_t>> idx_dict;
  for (int32_t idx : indices) {
    int64_t page_id = page_allocator.get_virt_page_id(idx, block_mem_size_);
    idx_dict[page_id].push_back(static_cast<int64_t>(idx));
  }
  std::vector<int64_t> pages_to_free;

  for (auto& [page_id, idxs] : idx_dict) {
    // Find the page - it must be in either full_pages or avail_pages
    VirtPage* page = nullptr;
    bool was_in_full = false;

    auto full_it = full_pages_.find(page_id);
    if (full_it != full_pages_.end()) {
      page = full_it->second.get();
      was_in_full = true;
    } else {
      auto avail_it = avail_pages_.find(page_id);
      if (avail_it != avail_pages_.end()) {
        page = avail_it->second.get();
      }
    }

    if (page == nullptr) {
      LOG(ERROR)
          << "Page " << page_id << " not found in avail_pages or full_pages. "
          << "Skipping to avoid crash, but this indicates a serious bug. "
          << "avail_pages size: " << avail_pages_.size() << ", "
          << "full_pages size: " << full_pages_.size() << ", "
          << "block_mem_size: " << block_mem_size_ << ", "
          << "page_size: " << page_allocator.page_size();
      continue;
    }

    num_avail_blocks_ += idxs.size();
    page->free_batch(idxs);

    if (page->empty()) {
      pages_to_free.push_back(page_id);
      num_avail_blocks_ -= page->num_free_blocks();
      if (was_in_full) {
        full_pages_.erase(page_id);
      } else {
        avail_pages_.erase(page_id);
      }
    } else if (was_in_full) {
      // Move from full_pages to avail_pages
      avail_pages_[page_id] = std::move(full_pages_[page_id]);
      full_pages_.erase(page_id);
    }
  }

  if (!pages_to_free.empty()) {
    page_allocator.free_kv_cache_pages(model_id_, dp_rank_, pages_to_free);
  }
}

void XTensorBlockManagerImpl::deallocate(const Slice<Block>& /*blocks*/) {
  // No-op: Block destructor will call free() when ref_count reaches 0
  // Prefix cache is not supported in XTensor mode
}

void XTensorBlockManagerImpl::free(int32_t block_id) {
  free_blocks({block_id});
}

std::vector<Block> XTensorBlockManagerImpl::allocate_shared(
    const Slice<int32_t>& /*token_ids*/,
    const Slice<Block>& /*existed_shared_blocks*/,
    const MMData& /*mm_data*/,
    const Slice<XXH3Key>& /*block_hashes*/) {
  // Prefix cache not supported
  VLOG(1) << "allocate_shared called but prefix cache is not supported";
  return {};
}

void XTensorBlockManagerImpl::cache(const Slice<int32_t>& /*token_ids*/,
                                    std::vector<Block>& /*blocks*/,
                                    size_t /*existed_shared_blocks_num*/,
                                    const MMData& /*mm_data*/,
                                    const Slice<XXH3Key>& /*block_hashes*/) {
  // Prefix cache not supported
  VLOG(1) << "cache called but prefix cache is not supported";
  return;
}

void XTensorBlockManagerImpl::cache(const std::vector<Block>& /*blocks*/) {
  // Prefix cache not supported
  VLOG(1) << "cache called but prefix cache is not supported";
  return;
}

size_t XTensorBlockManagerImpl::num_free_blocks() const {
  return available_size();
}

size_t XTensorBlockManagerImpl::num_used_blocks() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return get_num_allocated_blocks();
}

double XTensorBlockManagerImpl::kv_cache_utilization() const {
  std::lock_guard<std::mutex> lock(mtx_);
  size_t total = num_total_blocks();
  if (total == 0) return 0.0;
  return static_cast<double>(get_num_allocated_blocks()) / total;
}

size_t XTensorBlockManagerImpl::num_total_blocks() const {
  return options_.num_blocks();
}

size_t XTensorBlockManagerImpl::available_size_internal() const {
  // Note: Caller must hold mtx_
  size_t avail_blocks = num_avail_blocks_.load() + reserved_blocks_.size();

  // Conservative approach: only count reserved pages (already mapped to
  // physical memory). free_page_list_ pages are not counted because they
  // require physical memory mapping which may fail if GPU memory is
  // insufficient.
  auto& page_allocator = PageAllocator::get_instance();
  size_t reserved_pages =
      page_allocator.get_num_reserved_virt_pages(model_id_, dp_rank_);
  size_t blocks_from_reserved_pages =
      reserved_pages * VirtPage::get_num_blocks(page_size_, block_mem_size_);

  return avail_blocks + blocks_from_reserved_pages;
}

size_t XTensorBlockManagerImpl::available_size() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return available_size_internal();
}

bool XTensorBlockManagerImpl::try_to_reserve(size_t need_size) {
  // Check available size first (consistent with Python version)
  if (available_size() < need_size) {
    return false;
  }

  // alloc_internal will acquire mtx_ internally
  auto reserved = alloc_internal(need_size);
  if (reserved.empty()) {
    LOG(WARNING) << "Failed to reserve blocks.";
    return false;
  }

  std::lock_guard<std::mutex> lock(mtx_);
  reserved_blocks_.insert(
      reserved_blocks_.end(), reserved.begin(), reserved.end());
  return true;
}

void XTensorBlockManagerImpl::free_reserved() {
  std::vector<int32_t> blocks_to_free;
  {
    std::lock_guard<std::mutex> lock(mtx_);
    if (reserved_blocks_.empty()) {
      return;
    }
    blocks_to_free = std::move(reserved_blocks_);
    reserved_blocks_.clear();
  }
  // free_blocks will acquire mtx_ internally
  free_blocks(blocks_to_free);
}

void XTensorBlockManagerImpl::trim() {
  std::lock_guard<std::mutex> lock(mtx_);
  PageAllocator::get_instance().trim_kv_cache(model_id_, dp_rank_);
}

size_t XTensorBlockManagerImpl::get_mapped_memory_size() const {
  auto& page_allocator = PageAllocator::get_instance();
  // Each virtual page uses phy_pages_per_virt_page physical pages
  // Each physical page is phy_page_size bytes
  return page_allocator.get_num_inuse_virt_pages(model_id_, dp_rank_) *
         page_allocator.phy_pages_per_virt_page(model_id_) *
         page_allocator.page_size();
}

size_t XTensorBlockManagerImpl::get_num_allocated_blocks() const {
  // Blocks from fully allocated pages
  size_t blocks_per_page =
      VirtPage::get_num_blocks(page_size_, block_mem_size_);
  size_t blocks_from_full_pages = full_pages_.size() * blocks_per_page;

  // Blocks from partially allocated pages
  // num_avail_blocks is the number of free blocks in the partially allocated
  // pages so the number of allocated blocks is the total number of blocks in
  // the partially allocated pages minus the number of free blocks.
  size_t blocks_from_avail_pages =
      avail_pages_.size() * blocks_per_page - num_avail_blocks_.load();

  // Blocks from reserved blocks
  size_t blocks_from_reserved = reserved_blocks_.size();

  return blocks_from_full_pages + blocks_from_avail_pages +
         blocks_from_reserved;
}

}  // namespace xllm

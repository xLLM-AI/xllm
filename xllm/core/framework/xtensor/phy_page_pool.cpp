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

#include "phy_page_pool.h"

#include <glog/logging.h>

#include <algorithm>
#include <unordered_set>

namespace xllm {

void PhyPagePool::init(const torch::Device& device, size_t num_pages) {
  std::lock_guard<std::mutex> lock(mtx_);

  if (initialized_) {
    LOG(WARNING) << "PhyPagePool already initialized, ignoring re-init";
    return;
  }

  device_ = device;
  num_total_pages_ = num_pages;

  LOG(INFO) << "PhyPagePool: pre-allocating " << num_pages
            << " physical pages on device " << device;

  // Pre-allocate all physical pages for data with unique page_ids
  all_pages_.reserve(num_pages);
  page_allocated_.resize(num_pages, false);

  all_page_ptrs_.reserve(num_pages);
  for (size_t i = 0; i < num_pages; ++i) {
    page_id_t page_id = static_cast<page_id_t>(i);
    all_pages_.push_back(std::make_unique<PhyPage>(device_, page_id));
    all_page_ptrs_.push_back(all_pages_.back().get());
    free_page_ids_.push_back(page_id);
  }

  initialized_ = true;

  LOG(INFO) << "PhyPagePool: successfully pre-allocated " << num_pages
            << " physical pages (page_id 0-" << (num_pages - 1) << ")";
}

std::unique_ptr<PhyPage> PhyPagePool::get() {
  std::lock_guard<std::mutex> lock(mtx_);

  CHECK(initialized_) << "PhyPagePool not initialized";

  if (free_page_ids_.empty()) {
    LOG(WARNING) << "PhyPagePool: no free pages available";
    return nullptr;
  }

  // FIFO: pop from front to allocate left-to-right
  page_id_t page_id = free_page_ids_.front();
  free_page_ids_.pop_front();
  page_allocated_[page_id] = true;

  // Move ownership to caller
  return std::move(all_pages_[page_id]);
}

std::vector<std::unique_ptr<PhyPage>> PhyPagePool::batch_get(size_t count) {
  std::lock_guard<std::mutex> lock(mtx_);

  CHECK(initialized_) << "PhyPagePool not initialized";

  if (count == 0) {
    return {};
  }

  if (free_page_ids_.size() < count) {
    LOG(WARNING) << "PhyPagePool: not enough free pages, requested " << count
                 << ", available " << free_page_ids_.size();
    return {};
  }

  std::vector<std::unique_ptr<PhyPage>> result;
  result.reserve(count);

  // FIFO: pop from front to allocate left-to-right
  for (size_t i = 0; i < count; ++i) {
    page_id_t page_id = free_page_ids_.front();
    free_page_ids_.pop_front();
    page_allocated_[page_id] = true;
    result.push_back(std::move(all_pages_[page_id]));
  }

  return result;
}

void PhyPagePool::put(std::unique_ptr<PhyPage> page) {
  if (page == nullptr) {
    return;
  }

  std::lock_guard<std::mutex> lock(mtx_);

  CHECK(initialized_) << "PhyPagePool not initialized";

  // Verify the page belongs to this pool (same device)
  CHECK(page->device() == device_) << "Page device mismatch: expected "
                                   << device_ << ", got " << page->device();

  page_id_t page_id = page->page_id();
  CHECK(page_id >= 0 && page_id < static_cast<page_id_t>(num_total_pages_))
      << "Invalid page_id: " << page_id;

  // Return ownership to pool
  all_pages_[page_id] = std::move(page);
  page_allocated_[page_id] = false;
  // Use push_front to keep smaller page_ids at front for KV cache allocation
  free_page_ids_.push_front(page_id);
}

void PhyPagePool::batch_put(std::vector<std::unique_ptr<PhyPage>>& pages) {
  if (pages.empty()) {
    return;
  }

  std::lock_guard<std::mutex> lock(mtx_);

  CHECK(initialized_) << "PhyPagePool not initialized";

  for (auto& page : pages) {
    if (page == nullptr) {
      continue;
    }
    // Verify the page belongs to this pool (same device)
    CHECK(page->device() == device_) << "Page device mismatch: expected "
                                     << device_ << ", got " << page->device();

    page_id_t page_id = page->page_id();
    CHECK(page_id >= 0 && page_id < static_cast<page_id_t>(num_total_pages_))
        << "Invalid page_id: " << page_id;

    // Return ownership to pool
    all_pages_[page_id] = std::move(page);
    page_allocated_[page_id] = false;
    // Use push_front to keep smaller page_ids at front for KV cache allocation
    free_page_ids_.push_front(page_id);
  }
  pages.clear();
}

page_id_t PhyPagePool::allocate_contiguous_from_right(size_t count) {
  std::lock_guard<std::mutex> lock(mtx_);

  CHECK(initialized_) << "PhyPagePool not initialized";

  if (count == 0 || count > free_page_ids_.size()) {
    return -1;
  }

  // Scan from right to left in page_allocated_ to find contiguous free segment
  size_t run = 0;
  page_id_t start_page = -1;

  for (int64_t i = static_cast<int64_t>(num_total_pages_) - 1; i >= 0; --i) {
    if (!page_allocated_[i]) {
      run++;
      if (run == count) {
        start_page = static_cast<page_id_t>(i);
        break;
      }
    } else {
      run = 0;
    }
  }

  if (start_page < 0) {
    LOG(WARNING) << "PhyPagePool: cannot find " << count
                 << " contiguous free pages from right";
    return -1;
  }

  // Mark these pages as allocated
  page_id_t end_page = start_page + static_cast<page_id_t>(count);
  for (page_id_t page_id = start_page; page_id < end_page; ++page_id) {
    page_allocated_[page_id] = true;
  }

  // Remove from free_page_ids_ in one pass - O(n)
  auto new_end = std::remove_if(free_page_ids_.begin(),
                                free_page_ids_.end(),
                                [start_page, end_page](page_id_t id) {
                                  return id >= start_page && id < end_page;
                                });
  free_page_ids_.erase(new_end, free_page_ids_.end());

  LOG(INFO) << "PhyPagePool: allocated " << count
            << " contiguous pages from right, start_page=" << start_page;

  return start_page;
}

std::vector<page_id_t> PhyPagePool::allocate_pages_from_right(size_t count) {
  std::lock_guard<std::mutex> lock(mtx_);

  CHECK(initialized_) << "PhyPagePool not initialized";

  if (count == 0 || count > free_page_ids_.size()) {
    LOG(WARNING) << "PhyPagePool: not enough free pages for non-contiguous "
                    "allocation, requested "
                 << count << ", available " << free_page_ids_.size();
    return {};
  }

  std::vector<page_id_t> result;
  result.reserve(count);

  // Scan from right to left to collect free pages (non-contiguous ok)
  for (int64_t i = static_cast<int64_t>(num_total_pages_) - 1;
       i >= 0 && result.size() < count;
       --i) {
    if (!page_allocated_[i]) {
      result.push_back(static_cast<page_id_t>(i));
    }
  }

  if (result.size() < count) {
    LOG(WARNING) << "PhyPagePool: cannot find enough free pages from right, "
                    "requested "
                 << count << ", found " << result.size();
    return {};
  }

  // Mark these pages as allocated
  for (page_id_t page_id : result) {
    page_allocated_[page_id] = true;
  }

  // Remove from free_page_ids_ in one pass - O(n)
  std::unordered_set<page_id_t> allocated_set(result.begin(), result.end());
  auto new_end =
      std::remove_if(free_page_ids_.begin(),
                     free_page_ids_.end(),
                     [&allocated_set](page_id_t id) {
                       return allocated_set.find(id) != allocated_set.end();
                     });
  free_page_ids_.erase(new_end, free_page_ids_.end());

  LOG(INFO) << "PhyPagePool: allocated " << count
            << " non-contiguous pages from right";

  return result;
}

void PhyPagePool::free_weight_pages(const std::vector<page_id_t>& page_ids) {
  if (page_ids.empty()) {
    return;
  }

  std::lock_guard<std::mutex> lock(mtx_);

  CHECK(initialized_) << "PhyPagePool not initialized";

  for (page_id_t page_id : page_ids) {
    CHECK(page_id >= 0 && page_id < static_cast<page_id_t>(num_total_pages_))
        << "Invalid page_id: " << page_id;

    if (!page_allocated_[page_id]) {
      LOG(WARNING) << "PhyPagePool: page " << page_id
                   << " is not allocated, skipping";
      continue;
    }

    page_allocated_[page_id] = false;
    // Push to back so large page_ids stay towards the right
    free_page_ids_.push_back(page_id);
  }

  LOG(INFO) << "PhyPagePool: freed " << page_ids.size() << " weight pages";
}

size_t PhyPagePool::num_available() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return free_page_ids_.size();
}

// ============== Global XTensor Support ==============

const std::vector<PhyPage*>& PhyPagePool::get_all_pages() const {
  CHECK(initialized_) << "PhyPagePool not initialized";
  return all_page_ptrs_;
}

}  // namespace xllm

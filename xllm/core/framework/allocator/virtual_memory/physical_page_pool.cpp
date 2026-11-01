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

#include "core/framework/allocator/virtual_memory/physical_page_pool.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <unordered_set>

#include "core/framework/allocator/virtual_memory/shared_physical_page_mapping.h"
#include "core/platform/device.h"

namespace xllm {
namespace {

size_t saturating_subtract(size_t value, size_t subtrahend) {
  return value >= subtrahend ? value - subtrahend : 0;
}

}  // namespace

void PhysicalPagePool::init(const torch::Device& device,
                            size_t num_pages,
                            size_t page_size,
                            bool lazy) {
  init(device, num_pages, num_pages, page_size, lazy);
}

void PhysicalPagePool::init(const torch::Device& device,
                            size_t num_pages,
                            size_t initial_num_pages,
                            size_t page_size,
                            bool lazy) {
  std::lock_guard<std::mutex> lock(mtx_);
  if (initialized_) {
    LOG(WARNING) << "PhysicalPagePool already initialized, ignoring re-init";
    return;
  }
  CHECK_GT(page_size, 0);
  CHECK_LE(num_pages,
           static_cast<size_t>(std::numeric_limits<page_id_t>::max()));
  CHECK_LE(num_pages, std::numeric_limits<size_t>::max() / page_size);
  CHECK_LE(initial_num_pages, num_pages);
  CHECK(lazy || initial_num_pages == num_pages)
      << "Fixed pools require every logical slot to be active";
  device_ = device;
  lazy_ = lazy;
  page_size_ = page_size;
  num_total_pages_ = num_pages;
  num_active_pages_ = initial_num_pages;
  all_pages_.resize(num_pages);
  all_page_ptrs_.resize(num_pages, nullptr);
  page_allocated_.resize(num_pages, false);
  page_active_.resize(num_pages, false);
  free_memory_query_ = [this]() { return Device(device_).free_memory(); };
  allocator_memory_query_ = [this]() {
    return Device(device_).allocator_memory_stats();
  };
  page_factory_ = [this](page_id_t page_id) {
    return PhysicalPage::try_create(device_, page_size_, page_id);
  };
  // Slot geometry remains stable independently of resident physical backing.
  for (size_t i = 0; i < initial_num_pages; ++i) {
    const page_id_t page_id = static_cast<page_id_t>(i);
    page_active_[i] = true;
    if (!lazy) {
      all_pages_[i] =
          std::make_unique<PhysicalPage>(device_, page_size_, page_id);
      all_page_ptrs_[i] = all_pages_[i].get();
      ++num_committed_pages_;
    }
    free_page_ids_.emplace_back(page_id);
  }
  initialized_ = true;
  LOG(INFO) << "PhysicalPagePool: initialized " << num_pages
            << " stable slots with " << num_active_pages_
            << " active slots and " << num_committed_pages_
            << " resident pages on device " << device_;
}

bool PhysicalPagePool::reset() {
  std::lock_guard<std::mutex> lock(mtx_);
  if (!initialized_) {
    return true;
  }
  if (num_allocated_pages_ != 0 || !shared_mapping_.expired()) {
    LOG(ERROR) << "PhysicalPagePool reset requested with live pages or aliases";
    return false;
  }
  free_page_ids_.clear();
  all_page_ptrs_.clear();
  all_pages_.clear();
  page_allocated_.clear();
  page_active_.clear();
  page_factory_ = {};
  free_memory_query_ = {};
  allocator_memory_query_ = {};
  num_total_pages_ = 0;
  num_active_pages_ = 0;
  num_allocated_pages_ = 0;
  num_committed_pages_ = 0;
  allocation_headroom_bytes_ = 0;
  growth_reserve_bytes_ = 0;
  execution_memory_budget_.reset();
  lazy_ = false;
  page_size_ = 0;
  device_ = torch::Device(torch::kCPU);
  initialized_ = false;
  return true;
}

bool PhysicalPagePool::materialize_pages_locked(
    const std::vector<page_id_t>& page_ids) {
  std::vector<page_id_t> missing;
  missing.reserve(page_ids.size());
  for (page_id_t page_id : page_ids) {
    CHECK_GE(page_id, 0);
    CHECK_LT(static_cast<size_t>(page_id), num_total_pages_);
    if (all_page_ptrs_[page_id] == nullptr) {
      missing.emplace_back(page_id);
    }
  }
  if (missing.empty()) {
    return true;
  }
  // A scheduler snapshot is not a reservation: recheck under the growth lock.
  const int64_t free_bytes = free_memory_query_();
  const auto allocator_stats = allocator_memory_query_();
  if (free_bytes < 0) {
    execution_memory_budget_.observe_allocator_memory(allocator_stats);
    LOG(WARNING) << "PhysicalPagePool growth failed: "
                    "reason=free_memory_query_failed"
                 << " device_index=" << static_cast<int32_t>(device_.index())
                 << " requested_pages=" << missing.size()
                 << " committed_pages=" << num_committed_pages_
                 << " allocated_pages=" << num_allocated_pages_
                 << " active_pages=" << num_active_pages_
                 << " total_pages=" << num_total_pages_
                 << " page_size=" << page_size_ << " free_bytes=" << free_bytes;
    return false;
  }
  const size_t headroom = allocation_headroom_locked(
      static_cast<size_t>(free_bytes), allocator_stats);
  if (missing.size() >
      (std::numeric_limits<size_t>::max() - headroom) / page_size_) {
    LOG(WARNING) << "PhysicalPagePool growth failed: "
                    "reason=request_size_overflow"
                 << " device_index=" << static_cast<int32_t>(device_.index())
                 << " requested_pages=" << missing.size()
                 << " page_size=" << page_size_
                 << " headroom_bytes=" << headroom
                 << " growth_reserve_bytes=" << growth_reserve_bytes_
                 << " allocation_headroom_bytes=" << allocation_headroom_bytes_;
    return false;
  }
  const size_t required_bytes = missing.size() * page_size_ + headroom;
  if (static_cast<uint64_t>(free_bytes) < required_bytes) {
    LOG(WARNING)
        << "PhysicalPagePool growth failed: "
           "reason=insufficient_free_memory"
        << " device_index=" << static_cast<int32_t>(device_.index())
        << " requested_pages=" << missing.size() << " page_size=" << page_size_
        << " required_bytes=" << required_bytes << " free_bytes=" << free_bytes
        << " headroom_bytes=" << headroom
        << " growth_reserve_bytes=" << growth_reserve_bytes_
        << " allocation_headroom_bytes=" << allocation_headroom_bytes_
        << " allocator_stats_available=" << allocator_stats.has_value()
        << " allocator_reserved_bytes="
        << (allocator_stats.has_value() ? allocator_stats->reserved_bytes : 0)
        << " allocator_reported_peak_reserved_bytes="
        << (allocator_stats.has_value() ? allocator_stats->peak_reserved_bytes
                                        : 0)
        << " committed_pages=" << num_committed_pages_
        << " allocated_pages=" << num_allocated_pages_
        << " active_pages=" << num_active_pages_
        << " total_pages=" << num_total_pages_;
    return false;
  }
  std::vector<std::unique_ptr<PhysicalPage>> pending;
  std::vector<PhysicalPage*> pending_ptrs;
  pending.reserve(missing.size());
  pending_ptrs.reserve(missing.size());
  for (page_id_t page_id : missing) {
    auto page = page_factory_(page_id);
    if (page == nullptr) {
      // Private handles are destroyed without publishing slots or accounting.
      LOG(WARNING)
          << "PhysicalPagePool growth failed: "
             "reason=physical_page_creation"
          << " device_index=" << static_cast<int32_t>(device_.index())
          << " page_id=" << page_id << " creation_index=" << pending.size()
          << " requested_pages=" << missing.size()
          << " available_free_slots=" << free_page_ids_.size()
          << " page_size=" << page_size_
          << " free_bytes_before_creation=" << free_bytes
          << " required_bytes_before_creation=" << required_bytes
          << " headroom_bytes_before_creation=" << headroom
          << " allocator_stats_available=" << allocator_stats.has_value()
          << " allocator_reserved_bytes="
          << (allocator_stats.has_value() ? allocator_stats->reserved_bytes : 0)
          << " allocator_reported_peak_reserved_bytes="
          << (allocator_stats.has_value() ? allocator_stats->peak_reserved_bytes
                                          : 0)
          << " committed_pages=" << num_committed_pages_
          << " pending_pages=" << pending.size()
          << " allocated_pages=" << num_allocated_pages_;
      return false;
    }
    pending_ptrs.emplace_back(page.get());
    pending.emplace_back(std::move(page));
  }
  // Device allocators run outside this pool lock. Recheck their remaining
  // budget before publishing backing, keeping the more conservative recovery
  // reserve from allocator snapshots on both sides of the driver query.
  const auto allocator_stats_before = allocator_memory_query_();
  const int64_t remaining_bytes = free_memory_query_();
  const auto allocator_stats_after = allocator_memory_query_();
  execution_memory_budget_.observe_allocator_memory(allocator_stats_before);
  execution_memory_budget_.observe_allocator_memory(allocator_stats_after);
  if (remaining_bytes < 0) {
    LOG(WARNING) << "PhysicalPagePool growth failed: "
                    "reason=post_allocation_memory_query_failed"
                 << " device_index=" << static_cast<int32_t>(device_.index())
                 << " requested_pages=" << missing.size()
                 << " page_size=" << page_size_
                 << " committed_pages=" << num_committed_pages_
                 << " allocated_pages=" << num_allocated_pages_
                 << " remaining_bytes=" << remaining_bytes;
    return false;
  }
  // Pending handles are already resident, although pool ownership has not
  // been published. Exclude their bytes from fallback execution accounting.
  const size_t committed_bytes =
      (num_committed_pages_ + pending.size()) * page_size_;
  const size_t remaining_headroom =
      std::max(growth_reserve_bytes_,
               std::max(execution_memory_budget_.headroom(
                            static_cast<size_t>(remaining_bytes),
                            committed_bytes,
                            allocation_headroom_bytes_,
                            allocator_stats_before),
                        execution_memory_budget_.headroom(
                            static_cast<size_t>(remaining_bytes),
                            committed_bytes,
                            allocation_headroom_bytes_,
                            allocator_stats_after)));
  if (static_cast<uint64_t>(remaining_bytes) < remaining_headroom) {
    LOG(WARNING) << "PhysicalPagePool growth failed: "
                    "reason=post_allocation_headroom"
                 << " device_index=" << static_cast<int32_t>(device_.index())
                 << " requested_pages=" << missing.size()
                 << " page_size=" << page_size_
                 << " remaining_bytes=" << remaining_bytes
                 << " required_headroom_bytes=" << remaining_headroom
                 << " growth_reserve_bytes=" << growth_reserve_bytes_
                 << " allocation_headroom_bytes=" << allocation_headroom_bytes_
                 << " allocator_stats_before_available="
                 << allocator_stats_before.has_value()
                 << " allocator_reserved_before_bytes="
                 << (allocator_stats_before.has_value()
                         ? allocator_stats_before->reserved_bytes
                         : 0)
                 << " allocator_peak_before_bytes="
                 << (allocator_stats_before.has_value()
                         ? allocator_stats_before->peak_reserved_bytes
                         : 0)
                 << " allocator_stats_after_available="
                 << allocator_stats_after.has_value()
                 << " allocator_reserved_after_bytes="
                 << (allocator_stats_after.has_value()
                         ? allocator_stats_after->reserved_bytes
                         : 0)
                 << " allocator_peak_after_bytes="
                 << (allocator_stats_after.has_value()
                         ? allocator_stats_after->peak_reserved_bytes
                         : 0)
                 << " committed_pages=" << num_committed_pages_
                 << " pending_pages=" << pending.size()
                 << " allocated_pages=" << num_allocated_pages_;
    return false;
  }
  auto mapping = shared_mapping_.lock();
  if (mapping != nullptr && !mapping->map_pages(pending_ptrs)) {
    LOG(WARNING) << "PhysicalPagePool growth failed: "
                    "reason=shared_page_mapping"
                 << " device_index=" << static_cast<int32_t>(device_.index())
                 << " requested_pages=" << missing.size()
                 << " page_size=" << page_size_
                 << " committed_pages=" << num_committed_pages_
                 << " allocated_pages=" << num_allocated_pages_;
    return false;
  }
  for (size_t i = 0; i < missing.size(); ++i) {
    const page_id_t page_id = missing[i];
    all_page_ptrs_[page_id] = pending[i].get();
    all_pages_[page_id] = std::move(pending[i]);
  }
  num_committed_pages_ += missing.size();
  return true;
}

std::unique_ptr<PhysicalPage> PhysicalPagePool::get() {
  auto pages = batch_get(/*count=*/1);
  return pages.empty() ? nullptr : std::move(pages.front());
}

std::vector<std::unique_ptr<PhysicalPage>> PhysicalPagePool::batch_get(
    size_t count) {
  std::lock_guard<std::mutex> lock(mtx_);
  CHECK(initialized_) << "PhysicalPagePool not initialized";
  if (count == 0) {
    return {};
  }
  const size_t available_logical_slots =
      saturating_subtract(num_total_pages_, num_allocated_pages_);
  if (count > available_logical_slots) {
    LOG(WARNING) << "PhysicalPagePool allocation failed: "
                    "reason=logical_capacity"
                 << " device_index=" << static_cast<int32_t>(device_.index())
                 << " requested_pages=" << count
                 << " available_logical_slots=" << available_logical_slots
                 << " available_free_slots=" << free_page_ids_.size()
                 << " committed_pages=" << num_committed_pages_
                 << " allocated_pages=" << num_allocated_pages_
                 << " active_pages=" << num_active_pages_
                 << " total_pages=" << num_total_pages_;
    return {};
  }
  std::vector<page_id_t> page_ids;
  page_ids.reserve(count);
  auto it = free_page_ids_.begin();
  const size_t active_free_count = std::min(count, free_page_ids_.size());
  for (size_t i = 0; i < active_free_count; ++i, ++it) {
    page_ids.emplace_back(*it);
  }
  for (size_t i = 0; i < num_total_pages_ && page_ids.size() < count; ++i) {
    if (!page_active_[i]) {
      page_ids.emplace_back(static_cast<page_id_t>(i));
    }
  }
  if (page_ids.size() < count) {
    LOG(WARNING) << "PhysicalPagePool allocation failed: "
                    "reason=logical_accounting_mismatch"
                 << " device_index=" << static_cast<int32_t>(device_.index())
                 << " requested_pages=" << count
                 << " selected_pages=" << page_ids.size()
                 << " available_logical_slots=" << available_logical_slots
                 << " available_free_slots=" << free_page_ids_.size()
                 << " committed_pages=" << num_committed_pages_
                 << " allocated_pages=" << num_allocated_pages_
                 << " active_pages=" << num_active_pages_
                 << " total_pages=" << num_total_pages_;
    return {};
  }
  std::vector<std::unique_ptr<PhysicalPage>> result;
  result.reserve(count);
  if (!materialize_pages_locked(page_ids)) {
    return {};
  }
  // Activation is published only after the complete backing operation succeeds.
  activate_pages_locked(page_ids);
  for (size_t i = 0; i < active_free_count; ++i) {
    free_page_ids_.pop_front();
  }
  for (page_id_t page_id : page_ids) {
    page_allocated_[page_id] = true;
    result.emplace_back(std::move(all_pages_[page_id]));
  }
  num_allocated_pages_ += count;
  return result;
}

void PhysicalPagePool::put(std::unique_ptr<PhysicalPage> page) {
  if (page == nullptr) {
    return;
  }
  std::vector<std::unique_ptr<PhysicalPage>> pages;
  pages.reserve(1);
  pages.emplace_back(std::move(page));
  batch_put(pages);
}

void PhysicalPagePool::batch_put(
    std::vector<std::unique_ptr<PhysicalPage>>& pages) {
  if (pages.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(mtx_);
  CHECK(initialized_) << "PhysicalPagePool not initialized";
  std::unordered_set<page_id_t> seen;
  seen.reserve(pages.size());
  for (const auto& page : pages) {
    if (page == nullptr) {
      continue;
    }
    CHECK_EQ(page->device(), device_);
    CHECK_EQ(page->page_size(), page_size_);
    const page_id_t page_id = page->page_id();
    CHECK_GE(page_id, 0);
    CHECK_LT(static_cast<size_t>(page_id), num_total_pages_);
    CHECK(seen.emplace(page_id).second) << "Duplicate returned physical page";
    CHECK(page_allocated_[page_id]);
    CHECK(all_pages_[page_id] == nullptr);
    CHECK_EQ(all_page_ptrs_[page_id], page.get());
  }
  for (auto& page : pages) {
    if (page == nullptr) {
      continue;
    }
    const page_id_t page_id = page->page_id();
    all_pages_[page_id] = std::move(page);
    page_allocated_[page_id] = false;
    --num_allocated_pages_;
    free_page_ids_.emplace_front(page_id);
  }
  pages.clear();
}

void PhysicalPagePool::reserve_pages_locked(
    const std::vector<page_id_t>& page_ids) {
  std::unordered_set<page_id_t> reserved(page_ids.begin(), page_ids.end());
  activate_pages_locked(page_ids);
  for (page_id_t page_id : page_ids) {
    CHECK(!page_allocated_[page_id]);
    CHECK(all_pages_[page_id] != nullptr);
    page_allocated_[page_id] = true;
  }
  num_allocated_pages_ += page_ids.size();
  const auto end = std::remove_if(
      free_page_ids_.begin(),
      free_page_ids_.end(),
      [&reserved](page_id_t page_id) { return reserved.count(page_id) != 0; });
  free_page_ids_.erase(end, free_page_ids_.end());
}

page_id_t PhysicalPagePool::allocate_contiguous_from_right(size_t count) {
  std::lock_guard<std::mutex> lock(mtx_);
  CHECK(initialized_) << "PhysicalPagePool not initialized";
  const size_t available_logical_slots =
      saturating_subtract(num_total_pages_, num_allocated_pages_);
  if (count == 0) {
    return -1;
  }
  if (count > available_logical_slots) {
    LOG(WARNING) << "PhysicalPagePool contiguous allocation failed: "
                    "reason=logical_capacity"
                 << " device_index=" << static_cast<int32_t>(device_.index())
                 << " requested_pages=" << count
                 << " available_logical_slots=" << available_logical_slots
                 << " available_free_slots=" << free_page_ids_.size()
                 << " allocated_pages=" << num_allocated_pages_
                 << " total_pages=" << num_total_pages_;
    return -1;
  }
  size_t run = 0;
  page_id_t start_page = -1;
  for (int64_t i = static_cast<int64_t>(num_total_pages_) - 1; i >= 0; --i) {
    if (page_allocated_[i]) {
      run = 0;
      continue;
    }
    if (++run == count) {
      start_page = i;
      break;
    }
  }
  if (start_page < 0) {
    LOG(WARNING) << "PhysicalPagePool contiguous allocation failed: "
                    "reason=contiguous_capacity"
                 << " device_index=" << static_cast<int32_t>(device_.index())
                 << " requested_pages=" << count
                 << " available_logical_slots=" << available_logical_slots
                 << " available_free_slots=" << free_page_ids_.size()
                 << " allocated_pages=" << num_allocated_pages_
                 << " total_pages=" << num_total_pages_;
    return -1;
  }
  std::vector<page_id_t> page_ids;
  page_ids.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    page_ids.emplace_back(start_page + static_cast<page_id_t>(i));
  }
  // Weight reservations retain pool ownership but pin backing as active KV
  // ownership does. No reservation is visible on partial allocation failure.
  if (!materialize_pages_locked(page_ids)) {
    return -1;
  }
  reserve_pages_locked(page_ids);
  return start_page;
}

std::vector<page_id_t> PhysicalPagePool::allocate_pages_from_right(
    size_t count) {
  std::lock_guard<std::mutex> lock(mtx_);
  CHECK(initialized_) << "PhysicalPagePool not initialized";
  const size_t available_logical_slots =
      saturating_subtract(num_total_pages_, num_allocated_pages_);
  if (count == 0) {
    return {};
  }
  if (count > available_logical_slots) {
    LOG(WARNING) << "PhysicalPagePool right-side allocation failed: "
                    "reason=logical_capacity"
                 << " device_index=" << static_cast<int32_t>(device_.index())
                 << " requested_pages=" << count
                 << " available_logical_slots=" << available_logical_slots
                 << " available_free_slots=" << free_page_ids_.size()
                 << " allocated_pages=" << num_allocated_pages_
                 << " total_pages=" << num_total_pages_;
    return {};
  }
  std::vector<page_id_t> page_ids;
  page_ids.reserve(count);
  for (int64_t i = static_cast<int64_t>(num_total_pages_) - 1;
       i >= 0 && page_ids.size() < count;
       --i) {
    if (!page_allocated_[i]) {
      page_ids.emplace_back(static_cast<page_id_t>(i));
    }
  }
  if (page_ids.size() < count) {
    LOG(WARNING) << "PhysicalPagePool right-side allocation failed: "
                    "reason=logical_accounting_mismatch"
                 << " device_index=" << static_cast<int32_t>(device_.index())
                 << " requested_pages=" << count
                 << " bitmap_free_pages=" << page_ids.size()
                 << " available_logical_slots=" << available_logical_slots
                 << " available_free_slots=" << free_page_ids_.size()
                 << " allocated_pages=" << num_allocated_pages_
                 << " total_pages=" << num_total_pages_;
    return {};
  }
  if (!materialize_pages_locked(page_ids)) {
    return {};
  }
  reserve_pages_locked(page_ids);
  return page_ids;
}

void PhysicalPagePool::release_reserved_pages(
    const std::vector<page_id_t>& page_ids) {
  std::lock_guard<std::mutex> lock(mtx_);
  CHECK(initialized_) << "PhysicalPagePool not initialized";
  for (page_id_t page_id : page_ids) {
    CHECK_GE(page_id, 0);
    CHECK_LT(static_cast<size_t>(page_id), num_total_pages_);
    if (!page_allocated_[page_id]) {
      continue;
    }
    CHECK(all_pages_[page_id] != nullptr)
        << "Cannot release transferred KV ownership as a weight reservation";
    page_allocated_[page_id] = false;
    --num_allocated_pages_;
    free_page_ids_.emplace_back(page_id);
  }
  // Reuse resident backing before requiring a new device allocation. This
  // preserves admission snapshots when weight pages return behind empty slots.
  std::stable_partition(
      free_page_ids_.begin(), free_page_ids_.end(), [this](page_id_t page_id) {
        return all_page_ptrs_[page_id] != nullptr;
      });
}

size_t PhysicalPagePool::num_available() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return free_page_ids_.size();
}

size_t PhysicalPagePool::num_committed() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return num_committed_pages_;
}

size_t PhysicalPagePool::num_active() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return num_active_pages_;
}

size_t PhysicalPagePool::num_allocatable() const {
  std::lock_guard<std::mutex> lock(mtx_);
  if (!initialized_) {
    return 0;
  }
  const size_t idle_pages =
      saturating_subtract(num_committed_pages_, num_allocated_pages_);
  const size_t uncommitted_pages =
      saturating_subtract(num_total_pages_, num_committed_pages_);
  const auto allocator_stats = allocator_memory_query_();
  if (uncommitted_pages == 0) {
    execution_memory_budget_.observe_allocator_memory(allocator_stats);
    return idle_pages;
  }
  const int64_t free_bytes = free_memory_query_();
  if (free_bytes <= 0) {
    execution_memory_budget_.observe_allocator_memory(allocator_stats);
    return idle_pages;
  }
  const size_t headroom = allocation_headroom_locked(
      static_cast<size_t>(free_bytes), allocator_stats);
  if (static_cast<uint64_t>(free_bytes) <= headroom) {
    return idle_pages;
  }
  const size_t growth_pages =
      (static_cast<size_t>(free_bytes) - headroom) / page_size_;
  return idle_pages + std::min(uncommitted_pages, growth_pages);
}

void PhysicalPagePool::set_allocation_headroom(size_t bytes) {
  std::lock_guard<std::mutex> lock(mtx_);
  allocation_headroom_bytes_ = bytes;
}

void PhysicalPagePool::set_growth_reserve(size_t bytes) {
  std::lock_guard<std::mutex> lock(mtx_);
  growth_reserve_bytes_ = bytes;
}

size_t PhysicalPagePool::allocation_headroom_locked(
    size_t free_bytes,
    std::optional<AllocatorMemoryStats> allocator_stats) const {
  return std::max(
      growth_reserve_bytes_,
      execution_memory_budget_.headroom(free_bytes,
                                        num_committed_pages_ * page_size_,
                                        allocation_headroom_bytes_,
                                        allocator_stats));
}

void PhysicalPagePool::activate_pages_locked(
    const std::vector<page_id_t>& page_ids) {
  for (page_id_t page_id : page_ids) {
    if (!page_active_[page_id]) {
      page_active_[page_id] = true;
      ++num_active_pages_;
    }
  }
}

void PhysicalPagePool::observe_execution_memory() {
  std::lock_guard<std::mutex> lock(mtx_);
  if (!initialized_) {
    return;
  }
  const int64_t free_bytes = free_memory_query_();
  const auto allocator_stats = allocator_memory_query_();
  if (free_bytes >= 0) {
    execution_memory_budget_.observe(static_cast<size_t>(free_bytes),
                                     num_committed_pages_ * page_size_,
                                     allocator_stats);
  } else {
    execution_memory_budget_.observe_allocator_memory(allocator_stats);
  }
}

size_t PhysicalPagePool::trim_idle(size_t keep_pages) {
  std::lock_guard<std::mutex> lock(mtx_);
  if (!initialized_ || !lazy_) {
    return 0;
  }
  auto mapping = shared_mapping_.lock();
  // Registrations keep their lease until all device/remote users finish. A
  // lease pins resident backing as well as the shared virtual address range.
  if (mapping != nullptr && mapping.use_count() > 2) {
    return 0;
  }
  const size_t idle_pages =
      saturating_subtract(num_committed_pages_, num_allocated_pages_);
  if (idle_pages <= keep_pages) {
    return 0;
  }
  const size_t release_count = idle_pages - keep_pages;
  std::vector<page_id_t> page_ids;
  page_ids.reserve(release_count);
  for (int64_t i = static_cast<int64_t>(num_total_pages_) - 1;
       i >= 0 && page_ids.size() < release_count;
       --i) {
    if (!page_allocated_[i] && all_pages_[i] != nullptr) {
      page_ids.emplace_back(i);
    }
  }
  // Model-local aliases are gone before pages return here. Remove every shared
  // alias before destroying handles, without relocating surviving slots.
  if (mapping != nullptr) {
    mapping->unmap_pages(page_ids);
  }
  for (page_id_t page_id : page_ids) {
    all_page_ptrs_[page_id] = nullptr;
    all_pages_[page_id].reset();
  }
  num_committed_pages_ -= page_ids.size();
  std::stable_partition(
      free_page_ids_.begin(), free_page_ids_.end(), [this](page_id_t page_id) {
        return all_page_ptrs_[page_id] != nullptr;
      });
  return page_ids.size();
}

std::vector<PhysicalPage*> PhysicalPagePool::get_all_pages() const {
  std::lock_guard<std::mutex> lock(mtx_);
  CHECK(initialized_) << "PhysicalPagePool not initialized";
  return all_page_ptrs_;
}

void PhysicalPagePool::attach_shared_mapping(
    const std::shared_ptr<SharedPhysicalPageMapping>& mapping) {
  std::lock_guard<std::mutex> lock(mtx_);
  CHECK(initialized_);
  CHECK(shared_mapping_.expired());
  mapping->init(device_, all_page_ptrs_, page_size_);
  CHECK(mapping->is_initialized());
  shared_mapping_ = mapping;
}

bool PhysicalPagePool::detach_shared_mapping(
    const SharedPhysicalPageMapping* mapping) {
  std::lock_guard<std::mutex> lock(mtx_);
  auto current = shared_mapping_.lock();
  CHECK(current.get() == mapping);
  if (current.use_count() != 2) {
    return false;
  }
  shared_mapping_.reset();
  return true;
}

std::shared_ptr<void> PhysicalPagePool::acquire_shared_mapping_lease(
    bool require_contiguous) {
  std::lock_guard<std::mutex> lock(mtx_);
  auto mapping = shared_mapping_.lock();
  if (mapping == nullptr ||
      (require_contiguous &&
       (lazy_ || num_committed_pages_ != num_total_pages_))) {
    return nullptr;
  }
  return std::shared_ptr<void>(mapping, mapping->base_vaddr());
}

}  // namespace xllm

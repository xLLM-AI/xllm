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

#include "core/framework/model_loader/weight/weight_memory_manager.h"

#include <glog/logging.h>

#include "core/framework/allocator/global_memory_region.h"
#include "core/framework/allocator/virtual_memory/mapped_memory_region.h"
#include "core/framework/allocator/virtual_memory/physical_page_pool.h"

namespace xllm {

WeightMemoryManager::WeightMemoryManager(PhysicalPagePool& pool,
                                         GlobalMemoryRegion& global_region)
    : pool_(pool), global_region_(global_region), page_coordinator_(pool) {}

WeightMemoryManager::~WeightMemoryManager() { clear(); }

void WeightMemoryManager::init(const torch::Device& device) {
  std::lock_guard<std::mutex> lock(mutex_);
  device_ = device;
}

bool WeightMemoryManager::alloc_weight_pages(const std::string& model_id,
                                             size_t num_pages) {
  std::lock_guard<std::mutex> lock(mutex_);
  const WeightAllocation* existing = store_.find(model_id);
  if (existing != nullptr && existing->num_pages() != 0) {
    LOG(ERROR) << "Model already has weight pages: " << model_id;
    return false;
  }
  WeightPageReservation reservation =
      page_coordinator_.reserve_weight_pages(num_pages);
  if (reservation.page_ids.empty()) {
    return false;
  }
  if (reservation.contiguous_start >= 0) {
    return record_weight_allocation_locked(
        model_id, reservation.contiguous_start, num_pages);
  }
  return record_weight_fallback_allocation_locked(model_id,
                                                  reservation.page_ids);
}

bool WeightMemoryManager::record_weight_allocation_locked(
    const std::string& model_id,
    page_id_t start_page_id,
    size_t num_pages) {
  void* base_ptr = global_region_.get_vaddr_by_page_id(start_page_id);
  CHECK_GE(start_page_id, 0);
  CHECK_LT(static_cast<size_t>(start_page_id), pool_.num_total());
  CHECK_LE(num_pages, pool_.num_total() - static_cast<size_t>(start_page_id));
  if (base_ptr == nullptr || num_pages == 0) {
    LOG(ERROR)
        << "WeightMemoryManager: invalid GlobalMemoryRegion region for model "
        << model_id;
    std::vector<page_id_t> page_ids;
    page_ids.reserve(num_pages);
    for (size_t i = 0; i < num_pages; ++i) {
      page_ids.emplace_back(start_page_id + static_cast<page_id_t>(i));
    }
    pool_.release_reserved_pages(page_ids);
    return false;
  }

  auto& weight = store_.get_or_create(model_id);
  size_t page_size = global_region_.page_size();
  std::vector<WeightSegment> segments = {
      {static_cast<uint64_t>(start_page_id) * page_size,
       static_cast<uint64_t>(num_pages) * page_size}};
  weight.set_contiguous(
      start_page_id, num_pages, base_ptr, std::move(segments));

  LOG(INFO) << "WeightMemoryManager: recorded weight allocation for model "
            << model_id << ", start_page=" << start_page_id
            << ", num_pages=" << num_pages << ", base_ptr=" << base_ptr;
  return true;
}

bool WeightMemoryManager::record_weight_fallback_allocation_locked(
    const std::string& model_id,
    const std::vector<page_id_t>& page_ids) {
  // Create MappedMemoryRegion with the non-contiguous preallocated pages
  auto weight_region = std::make_unique<MappedMemoryRegion>(
      page_ids, device_, pool_.page_size());
  if (is_null_vir_ptr(weight_region->vaddr()) ||
      !global_region_.is_initialized()) {
    LOG(ERROR)
        << "WeightMemoryManager: failed to create MappedMemoryRegion for model "
        << model_id;
    // The MappedMemoryRegion destructor returns its reserved pages to the pool.
    return false;
  }

  auto& weight = store_.get_or_create(model_id);
  // Preserve logical mapping order; merge only forward-adjacent pages.
  size_t page_size = global_region_.page_size();
  const auto raw_segments = make_weight_transfer_segments(page_ids, page_size);
  std::vector<WeightSegment> segments;
  segments.reserve(raw_segments.size());
  for (const auto& [offset, size] : raw_segments) {
    segments.emplace_back(offset, size);
  }
  weight.set_fragmented(
      std::move(weight_region), page_ids.size(), std::move(segments));

  LOG(INFO)
      << "WeightMemoryManager: recorded mapped memory allocation for model "
      << model_id << ", num_pages=" << page_ids.size()
      << ", base_ptr=" << weight.base_ptr()
      << ", weight_segments=" << weight.segments().size() << " (fallback mode)";
  return true;
}

bool WeightMemoryManager::allocate_weight(const std::string& model_id,
                                          void*& ptr,
                                          size_t size) {
  std::lock_guard<std::mutex> lock(mutex_);

  WeightAllocation* weight = store_.find(model_id);
  if (weight == nullptr || weight->base_ptr() == nullptr) {
    LOG(ERROR) << "No pre-allocated weight region for model " << model_id;
    return false;
  }

  if (!weight->allocate(ptr, size, global_region_.page_size())) {
    LOG(ERROR) << "Not enough space in weight region for model " << model_id
               << ": requested " << size;
    return false;
  }

  VLOG(1) << "WeightMemoryManager: allocated " << size << " bytes for model "
          << model_id << ", ptr=" << ptr;

  return true;
}

size_t WeightMemoryManager::free_weight(const std::string& model_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  WeightAllocation* weight = store_.find(model_id);
  if (weight == nullptr || weight->num_pages() == 0) {
    LOG(WARNING) << "No weight allocation found for model " << model_id;
    return 0;
  }

  size_t num_pages = weight->num_pages();

  // Handle MappedMemoryRegion fallback case
  if (weight->is_fragmented()) {
    // MappedMemoryRegion's destructor will unmap and free pages
    weight->reset();
    LOG(INFO) << "Freed " << num_pages
              << " weight pages (MappedMemoryRegion fallback) for model "
              << model_id;
  } else {
    // Normal path: free contiguous pages from GlobalMemoryRegion
    page_id_t start_page = weight->start_page_id();

    // Build page_ids vector and free via PhysicalPagePool
    std::vector<page_id_t> page_ids;
    page_ids.reserve(num_pages);
    for (size_t i = 0; i < num_pages; ++i) {
      page_ids.emplace_back(start_page + static_cast<page_id_t>(i));
    }

    WeightPageReservation reservation;
    reservation.contiguous_start = start_page;
    reservation.page_ids = std::move(page_ids);
    page_coordinator_.release_weight_pages(reservation);

    LOG(INFO) << "Freed " << num_pages << " weight pages for model "
              << model_id;
  }

  // Clear weight allocation record
  weight->reset();

  return num_pages;
}

std::optional<WeightAllocationInfo>
WeightMemoryManager::get_weight_allocation_info(
    const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const WeightAllocation* weight = store_.find(model_id);
  if (weight == nullptr) {
    return std::nullopt;
  }
  return WeightAllocationInfo{weight->base_ptr(), weight->num_pages()};
}

std::vector<WeightSegment> WeightMemoryManager::get_model_weight_segments(
    const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const WeightAllocation* weight = store_.find(model_id);
  if (weight == nullptr) {
    return {};
  }
  return weight->segments();
}

std::unordered_map<std::string, std::vector<WeightSegment>>
WeightMemoryManager::get_all_model_weight_segments() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::unordered_map<std::string, std::vector<WeightSegment>> result;

  for (const auto& [model_id, weight] : store_.models()) {
    if (!weight.segments().empty()) {
      result[model_id] = weight.segments();
    }
  }

  return result;
}

void WeightMemoryManager::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  // Release model-owned mappings before tearing down the global address space.
  for (const auto& [model_id, weight] : store_.models()) {
    if (weight.num_pages() == 0) {
      continue;
    }
    if (weight.is_fragmented()) {
      continue;
    }

    std::vector<page_id_t> page_ids;
    page_ids.reserve(weight.num_pages());
    for (size_t i = 0; i < weight.num_pages(); ++i) {
      page_ids.emplace_back(weight.start_page_id() + static_cast<page_id_t>(i));
    }
    pool_.release_reserved_pages(page_ids);
    VLOG(1) << "Released weight pages for model " << model_id;
  }
  store_.clear();
}

}  // namespace xllm

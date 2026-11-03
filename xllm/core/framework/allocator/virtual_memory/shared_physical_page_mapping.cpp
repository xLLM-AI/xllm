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

#include "core/framework/allocator/virtual_memory/shared_physical_page_mapping.h"

#include <glog/logging.h>

#include <limits>
#include <unordered_set>

namespace xllm {

SharedPhysicalPageMapping::~SharedPhysicalPageMapping() { reset(); }

void SharedPhysicalPageMapping::init(const torch::Device& device,
                                     const std::vector<PhysicalPage*>& pages,
                                     size_t page_size) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (initialized_) {
    LOG(WARNING) << "SharedPhysicalPageMapping already initialized";
    return;
  }
  if (pages.empty()) {
    LOG(ERROR) << "SharedPhysicalPageMapping: no logical page slots";
    return;
  }
  CHECK_GT(page_size, 0);
  CHECK_LE(pages.size(), std::numeric_limits<size_t>::max() / page_size);
  device_ = device;
  page_size_ = page_size;
  num_total_pages_ = pages.size();
  total_size_ = num_total_pages_ * page_size_;
  mapped_pages_.resize(num_total_pages_, nullptr);
  // Reserving virtual addresses consumes no physical KV backing. The full
  // arena remains at this address while individual slots become resident.
  vmm::create_vir_ptr(vaddr_, total_size_);
  initialized_ = true;
  std::vector<PhysicalPage*> resident;
  resident.reserve(pages.size());
  for (size_t i = 0; i < pages.size(); ++i) {
    if (pages[i] == nullptr) {
      continue;
    }
    CHECK_EQ(pages[i]->page_id(), static_cast<page_id_t>(i));
    resident.emplace_back(pages[i]);
  }
  if (!map_pages_locked(resident)) {
    vmm::release_vir_ptr(vaddr_, total_size_);
    vaddr_ = {};
    total_size_ = 0;
    page_size_ = 0;
    num_total_pages_ = 0;
    mapped_pages_.clear();
    initialized_ = false;
  }
}

void SharedPhysicalPageMapping::reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_) {
    return;
  }
  std::vector<page_id_t> resident;
  resident.reserve(mapped_pages_.size());
  for (size_t i = 0; i < mapped_pages_.size(); ++i) {
    if (mapped_pages_[i] != nullptr) {
      resident.emplace_back(static_cast<page_id_t>(i));
    }
  }
  unmap_pages_locked(resident);
  vmm::release_vir_ptr(vaddr_, total_size_);
  vaddr_ = {};
  total_size_ = 0;
  page_size_ = 0;
  num_total_pages_ = 0;
  mapped_pages_.clear();
  initialized_ = false;
}

bool SharedPhysicalPageMapping::map_pages_locked(
    const std::vector<PhysicalPage*>& pages) {
  CHECK(initialized_);
  std::unordered_set<page_id_t> seen;
  seen.reserve(pages.size());
  for (size_t i = 0; i < pages.size(); ++i) {
    const PhysicalPage* page = pages[i];
    if (page == nullptr) {
      LOG(ERROR) << "SharedPhysicalPageMapping rejected page: reason=null_page"
                 << " device_index=" << static_cast<int32_t>(device_.index())
                 << " input_index=" << i;
      return false;
    }

    const page_id_t page_id = page->page_id();
    if (page_id < 0 || static_cast<size_t>(page_id) >= num_total_pages_) {
      LOG(ERROR)
          << "SharedPhysicalPageMapping rejected page: reason=invalid_page_id"
          << " device_index=" << static_cast<int32_t>(device_.index())
          << " page_id=" << page_id << " total_pages=" << num_total_pages_;
      return false;
    }
    if (page->device() != device_) {
      LOG(ERROR)
          << "SharedPhysicalPageMapping rejected page: reason=device_mismatch"
          << " expected_device_index=" << static_cast<int32_t>(device_.index())
          << " actual_device_index="
          << static_cast<int32_t>(page->device().index())
          << " page_id=" << page_id;
      return false;
    }
    if (page->page_size() != page_size_) {
      LOG(ERROR) << "SharedPhysicalPageMapping rejected page: "
                    "reason=page_size_mismatch"
                 << " device_index=" << static_cast<int32_t>(device_.index())
                 << " page_id=" << page_id
                 << " expected_page_size=" << page_size_
                 << " actual_page_size=" << page->page_size();
      return false;
    }
    if (!seen.emplace(page_id).second) {
      LOG(ERROR)
          << "SharedPhysicalPageMapping rejected page: reason=duplicate_page_id"
          << " device_index=" << static_cast<int32_t>(device_.index())
          << " page_id=" << page_id;
      return false;
    }

    PhysicalPage* current = mapped_pages_[page_id];
    if (current != nullptr && current != page) {
      LOG(ERROR) << "SharedPhysicalPageMapping rejected page: "
                    "reason=conflicting_existing_mapping"
                 << " device_index=" << static_cast<int32_t>(device_.index())
                 << " page_id=" << page_id;
      return false;
    }
  }
  std::vector<page_id_t> added;
  added.reserve(pages.size());
  for (PhysicalPage* page : pages) {
    const page_id_t page_id = page->page_id();
    if (mapped_pages_[page_id] != nullptr) {
      continue;
    }
    VirPtr address = add_vir_ptr_offset(vaddr_, page_id * page_size_);
    PhyMemHandle handle = page->get_phy_handle();
    if (!vmm::try_map(address, handle, page_size_, device_.index())) {
      LOG(WARNING) << "SharedPhysicalPageMapping failed to map page: "
                      "reason=backend_mapping_failure"
                   << " device_index=" << static_cast<int32_t>(device_.index())
                   << " page_id=" << page_id << " page_size=" << page_size_;
      unmap_pages_locked(added);
      return false;
    }
    mapped_pages_[page_id] = page;
    added.emplace_back(page_id);
  }
  return true;
}

bool SharedPhysicalPageMapping::map_pages(
    const std::vector<PhysicalPage*>& pages) {
  std::lock_guard<std::mutex> lock(mutex_);
  return map_pages_locked(pages);
}

void SharedPhysicalPageMapping::unmap_pages_locked(
    const std::vector<page_id_t>& page_ids) {
  for (page_id_t page_id : page_ids) {
    CHECK_GE(page_id, 0);
    CHECK_LT(static_cast<size_t>(page_id), mapped_pages_.size());
    if (mapped_pages_[page_id] == nullptr) {
      continue;
    }
    VirPtr address = add_vir_ptr_offset(vaddr_, page_id * page_size_);
    vmm::unmap(address, page_size_, page_size_);
    mapped_pages_[page_id] = nullptr;
  }
}

void SharedPhysicalPageMapping::unmap_pages(
    const std::vector<page_id_t>& page_ids) {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK(initialized_);
  unmap_pages_locked(page_ids);
}

void* SharedPhysicalPageMapping::get_vaddr_by_page_id(page_id_t page_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_ || page_id < 0 ||
      static_cast<size_t>(page_id) >= num_total_pages_ ||
      mapped_pages_[page_id] == nullptr) {
    return nullptr;
  }
  return vir_ptr_to_void_ptr(add_vir_ptr_offset(vaddr_, page_id * page_size_));
}

}  // namespace xllm

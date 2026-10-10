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

#include "core/virtual_memory/shared_page_mapping.h"

#include <glog/logging.h>

#include <limits>

namespace xllm {

void SharedPageMapping::init(const torch::Device& device,
                             const std::vector<PhysicalPage*>& pages,
                             size_t page_size) {
  if (initialized_) {
    LOG(WARNING) << "SharedPageMapping already initialized";
    return;
  }

  num_total_pages_ = pages.size();
  if (num_total_pages_ == 0) {
    LOG(ERROR) << "SharedPageMapping: no physical pages";
    return;
  }

  CHECK_GT(page_size, 0);
  page_size_ = page_size;
  CHECK_LE(num_total_pages_, std::numeric_limits<size_t>::max() / page_size_);
  for (size_t page_id = 0; page_id < pages.size(); ++page_id) {
    const PhysicalPage* page = pages[page_id];
    CHECK(page != nullptr);
    CHECK_EQ(page->page_id(), static_cast<page_id_t>(page_id));
    CHECK_EQ(page->device(), device);
    CHECK_EQ(page->page_size(), page_size_);
  }
  total_size_ = num_total_pages_ * page_size_;

  vmm::create_vir_ptr(vaddr_, total_size_);
  if (is_null_vir_ptr(vaddr_)) {
    LOG(ERROR) << "SharedPageMapping: failed to allocate virtual memory";
    return;
  }

  if (!map_all_pages(pages)) {
    LOG(ERROR) << "Failed to map all pages for SharedPageMapping";
    vmm::release_vir_ptr(vaddr_, total_size_);
    vaddr_ = {};
    total_size_ = 0;
    page_size_ = 0;
    num_total_pages_ = 0;
    return;
  }

  initialized_ = true;
  LOG(INFO) << "SharedPageMapping initialized: " << num_total_pages_
            << " pages, " << total_size_ << " bytes";
}

void SharedPageMapping::reset() {
  if (!initialized_) {
    return;
  }

  for (size_t page_id = 0; page_id < num_total_pages_; ++page_id) {
    VirPtr address = add_vir_ptr_offset(vaddr_, page_id * page_size_);
    vmm::unmap(address, page_size_, page_size_);
  }
  vmm::release_vir_ptr(vaddr_, total_size_);

  vaddr_ = {};
  total_size_ = 0;
  page_size_ = 0;
  num_total_pages_ = 0;
  initialized_ = false;
}

bool SharedPageMapping::map_page(PhysicalPage* page, size_t offset) {
  CHECK(page) << "Page is null";
  CHECK(offset % page_size_ == 0) << "Offset not aligned to page size";
  CHECK(offset < total_size_) << "Offset out of bounds";

  VirPtr vaddr = add_vir_ptr_offset(vaddr_, offset);
  PhyMemHandle phy_handle = page->get_phy_handle();
  vmm::map(vaddr, phy_handle, page_size_, page->device().index());
  return true;
}

bool SharedPageMapping::map_all_pages(const std::vector<PhysicalPage*>& pages) {
  if (pages.size() != num_total_pages_) {
    LOG(ERROR) << "Page count mismatch: expected " << num_total_pages_
               << ", got " << pages.size();
    return false;
  }

  for (size_t i = 0; i < num_total_pages_; ++i) {
    size_t offset = i * page_size_;
    if (!map_page(pages[i], offset)) {
      LOG(ERROR) << "Failed to map page " << i << " at offset " << offset;
      for (size_t mapped = 0; mapped < i; ++mapped) {
        VirPtr address = add_vir_ptr_offset(vaddr_, mapped * page_size_);
        vmm::unmap(address, page_size_, page_size_);
      }
      return false;
    }
  }
  return true;
}

void* SharedPageMapping::get_vaddr_by_page_id(page_id_t page_id) const {
  if (!initialized_) {
    return nullptr;
  }

  if (page_id < 0 || static_cast<size_t>(page_id) >= num_total_pages_) {
    return nullptr;
  }

  return vir_ptr_to_void_ptr(add_vir_ptr_offset(vaddr_, page_id * page_size_));
}

}  // namespace xllm

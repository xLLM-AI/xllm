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

#include "global_xtensor.h"

#include <glog/logging.h>

#include <algorithm>

#include "common/global_flags.h"
#include "core/framework/config/kv_cache_config.h"
#include "phy_page_pool.h"

namespace xllm {

void GlobalXTensor::init(const torch::Device& device) {
  if (initialized_) {
    LOG(WARNING) << "GlobalXTensor already initialized";
    return;
  }

  auto& pool = PhyPagePool::get_instance();
  CHECK(pool.is_initialized()) << "PhyPagePool must be initialized first";

  num_total_pages_ = pool.num_total();
  if (num_total_pages_ == 0) {
    LOG(ERROR) << "GlobalXTensor: PhyPagePool has no pages";
    return;
  }

  page_size_ =
      ::xllm::KVCacheConfig::get_instance().phy_page_granularity_size();
  total_size_ = num_total_pages_ * page_size_;

  vmm::create_vir_ptr(vaddr_, total_size_);
  if (is_null_vir_ptr(vaddr_)) {
    LOG(ERROR) << "GlobalXTensor: failed to allocate virtual memory";
    return;
  }

  auto pages = pool.get_all_pages();
  if (!map_all_pages(pages)) {
    LOG(ERROR) << "Failed to map all pages for GlobalXTensor";
    return;
  }

  initialized_ = true;
  LOG(INFO) << "GlobalXTensor initialized: " << num_total_pages_ << " pages, "
            << total_size_ << " bytes";
}

bool GlobalXTensor::map_page(PhyPage* page, size_t offset) {
  CHECK(page) << "Page is null";
  CHECK(offset % page_size_ == 0) << "Offset not aligned to page size";
  CHECK(offset < total_size_) << "Offset out of bounds";

  VirPtr vaddr = add_vir_ptr_offset(vaddr_, offset);
  PhyMemHandle phy_handle = page->get_phy_handle();
  vmm::map(vaddr, phy_handle);
  return true;
}

bool GlobalXTensor::map_all_pages(const std::vector<PhyPage*>& pages) {
  if (pages.size() != num_total_pages_) {
    LOG(ERROR) << "Page count mismatch: expected " << num_total_pages_
               << ", got " << pages.size();
    return false;
  }

  for (size_t i = 0; i < num_total_pages_; ++i) {
    size_t offset = i * page_size_;
    if (!map_page(pages[i], offset)) {
      LOG(ERROR) << "Failed to map page " << i << " at offset " << offset;
      return false;
    }
  }
  return true;
}

void* GlobalXTensor::get_vaddr_by_page_id(page_id_t page_id) const {
  if (!initialized_) {
    return nullptr;
  }

  if (page_id < 0 || static_cast<size_t>(page_id) >= num_total_pages_) {
    return nullptr;
  }

  return vir_ptr_to_void_ptr(add_vir_ptr_offset(vaddr_, page_id * page_size_));
}

}  // namespace xllm

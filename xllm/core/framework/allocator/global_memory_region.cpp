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

#include "core/framework/allocator/global_memory_region.h"

#include <glog/logging.h>

#include "core/framework/allocator/virtual_memory/physical_page_pool.h"

namespace xllm {

GlobalMemoryRegion::GlobalMemoryRegion() {
  // The page owner must outlive this shared address space.
  PhysicalPagePool::get_instance();
}

void GlobalMemoryRegion::init(const torch::Device& device) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (mapping_->is_initialized()) {
    LOG(WARNING) << "GlobalMemoryRegion already initialized";
    return;
  }

  auto& pool = PhysicalPagePool::get_instance();
  CHECK(pool.is_initialized()) << "PhysicalPagePool must be initialized first";
  CHECK_EQ(device, pool.device()) << "GlobalMemoryRegion device mismatch";
  mapping_->init(device, pool.get_all_pages(), pool.page_size());
}

void GlobalMemoryRegion::reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!mapping_->is_initialized()) {
    return;
  }

  CHECK_EQ(mapping_.use_count(), 1)
      << "GlobalMemoryRegion has active mapping leases";
  mapping_->reset();
}

std::shared_ptr<void> GlobalMemoryRegion::acquire_mapping_lease() {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK(mapping_->is_initialized()) << "GlobalMemoryRegion is not initialized";
  return std::shared_ptr<void>(mapping_, mapping_->base_vaddr());
}

}  // namespace xllm

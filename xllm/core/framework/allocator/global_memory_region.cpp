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

void GlobalMemoryRegion::init(const torch::Device& device) {
  if (mapping_.is_initialized()) {
    LOG(WARNING) << "GlobalMemoryRegion already initialized";
    return;
  }

  auto& pool = PhysicalPagePool::get_instance();
  CHECK(pool.is_initialized()) << "PhysicalPagePool must be initialized first";
  CHECK_EQ(device, pool.device()) << "GlobalMemoryRegion device mismatch";
  mapping_.init(device, pool.get_all_pages(), pool.page_size());
}

void GlobalMemoryRegion::reset() {
  if (!mapping_.is_initialized()) {
    return;
  }

  CHECK(!mooncake_registered_)
      << "GlobalMemoryRegion must be unregistered before reset";
  mapping_.reset();
  mooncake_registered_ = false;
}

}  // namespace xllm

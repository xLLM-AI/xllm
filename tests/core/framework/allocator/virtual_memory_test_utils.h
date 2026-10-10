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

#pragma once

#include <glog/logging.h>

#include "core/framework/allocator/global_memory_region.h"
#include "core/framework/allocator/virtual_memory/physical_page_pool.h"

namespace xllm {

class VirtualMemoryTestPeer final {
 public:
  // Call after all tensor owners return their pages and before either NPU
  // Environment shuts down the runtime. Singleton destruction runs too late.
  static void release_resources() {
    auto& global_tensor = GlobalMemoryRegion::get_instance();
    CHECK(!global_tensor.is_mooncake_registered());
    global_tensor.reset();
    CHECK(PhysicalPagePool::get_instance().reset());
  }
};

}  // namespace xllm

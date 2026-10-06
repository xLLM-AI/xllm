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

#include <algorithm>
#include <mutex>

#include "core/framework/xtensor/global_xtensor.h"
#include "core/framework/xtensor/phy_page_pool.h"
#include "core/platform/vmm_api.h"

namespace xllm {

class XTensorTestPeer final {
 public:
  // Call after all tensor owners return their pages and before either NPU
  // Environment shuts down the runtime. Singleton destruction runs too late.
  static void release_resources() {
    auto& global_tensor = GlobalXTensor::get_instance();
    CHECK(!global_tensor.mooncake_registered_);
    if (!is_null_vir_ptr(global_tensor.vaddr_)) {
      CHECK(global_tensor.initialized_);
      vmm::unmap(global_tensor.vaddr_, global_tensor.total_size_);
      vmm::release_vir_ptr(global_tensor.vaddr_, global_tensor.total_size_);
    }
    global_tensor.vaddr_ = {};
    global_tensor.total_size_ = 0;
    global_tensor.page_size_ = 0;
    global_tensor.num_total_pages_ = 0;
    global_tensor.initialized_ = false;

    auto& pool = PhyPagePool::get_instance();
    std::lock_guard<std::mutex> lock(pool.mtx_);
    CHECK_EQ(pool.free_page_ids_.size(), pool.num_total_pages_);
    CHECK(std::all_of(pool.all_pages_.begin(),
                      pool.all_pages_.end(),
                      [](const auto& page) { return page != nullptr; }));
    pool.all_page_ptrs_.clear();
    pool.all_pages_.clear();
    pool.free_page_ids_.clear();
    pool.page_allocated_.clear();
    pool.num_total_pages_ = 0;
    pool.device_ = torch::Device(torch::kCPU);
    pool.initialized_ = false;
  }
};

}  // namespace xllm

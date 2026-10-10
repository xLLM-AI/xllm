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

#pragma once

#include <torch/types.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

#include "core/framework/allocator/virtual_memory/shared_page_mapping.h"

namespace xllm {

// Worker-local shared mapping facade. Users retain a lease that keeps
// the mapping alive and prevents reset while its address range is in use.
class GlobalMemoryRegion final {
 public:
  static GlobalMemoryRegion& get_instance() {
    static GlobalMemoryRegion instance;
    return instance;
  }

  void init(const torch::Device& device);
  void reset();

  bool is_initialized() const { return mapping_->is_initialized(); }
  void* get_vaddr_by_page_id(page_id_t page_id) const {
    return mapping_->get_vaddr_by_page_id(page_id);
  }
  void* base_vaddr() const { return mapping_->base_vaddr(); }
  size_t total_size() const { return mapping_->total_size(); }
  size_t num_total_pages() const { return mapping_->num_total_pages(); }
  size_t page_size() const { return mapping_->page_size(); }

  std::shared_ptr<void> acquire_mapping_lease();

 private:
  GlobalMemoryRegion();
  ~GlobalMemoryRegion() = default;
  GlobalMemoryRegion(const GlobalMemoryRegion&) = delete;
  GlobalMemoryRegion& operator=(const GlobalMemoryRegion&) = delete;

  std::shared_ptr<SharedPageMapping> mapping_ =
      std::make_shared<SharedPageMapping>();
  std::mutex mutex_;
};

}  // namespace xllm

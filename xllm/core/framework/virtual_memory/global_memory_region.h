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

#include "core/virtual_memory/shared_page_mapping.h"

namespace xllm {

// Worker-local shared mapping facade. KV and weight transfer share one
// registration state, and registered memory cannot be reset.
class GlobalMemoryRegion final {
 public:
  static GlobalMemoryRegion& get_instance() {
    static GlobalMemoryRegion instance;
    return instance;
  }

  void init(const torch::Device& device);
  void reset();

  bool is_initialized() const { return mapping_.is_initialized(); }
  void* get_vaddr_by_page_id(page_id_t page_id) const {
    return mapping_.get_vaddr_by_page_id(page_id);
  }
  void* base_vaddr() const { return mapping_.base_vaddr(); }
  size_t total_size() const { return mapping_.total_size(); }
  size_t num_total_pages() const { return mapping_.num_total_pages(); }
  size_t page_size() const { return mapping_.page_size(); }

  bool is_mooncake_registered() const { return mooncake_registered_; }
  void set_mooncake_registered(bool registered) {
    mooncake_registered_ = registered;
  }

 private:
  GlobalMemoryRegion() = default;
  ~GlobalMemoryRegion() = default;
  GlobalMemoryRegion(const GlobalMemoryRegion&) = delete;
  GlobalMemoryRegion& operator=(const GlobalMemoryRegion&) = delete;

  SharedPageMapping mapping_;
  bool mooncake_registered_ = false;
};

}  // namespace xllm

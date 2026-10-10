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

#include <torch/types.h>

#include <cstddef>
#include <vector>

#include "core/virtual_memory/physical_page.h"

namespace xllm {

// Maps physical pages into one shared virtual address range. Page ownership
// remains with the caller, which must keep every page alive until reset().
// Input pages must be in physical page_id order: pages[i]->page_id() == i.
// Address lookup uses this pool order without changing page ownership.
class SharedPageMapping final {
 public:
  SharedPageMapping() = default;
  ~SharedPageMapping() = default;

  SharedPageMapping(const SharedPageMapping&) = delete;
  SharedPageMapping& operator=(const SharedPageMapping&) = delete;

  void init(const torch::Device& device,
            const std::vector<PhysicalPage*>& pages,
            size_t page_size);
  void reset();

  bool is_initialized() const { return initialized_; }
  void* get_vaddr_by_page_id(page_id_t page_id) const;
  void* base_vaddr() const { return vir_ptr_to_void_ptr(vaddr_); }
  size_t total_size() const { return total_size_; }
  size_t num_total_pages() const { return num_total_pages_; }
  size_t page_size() const { return page_size_; }

 private:
  bool map_page(PhysicalPage* page, size_t offset);
  bool map_all_pages(const std::vector<PhysicalPage*>& pages);

  bool initialized_ = false;
  VirPtr vaddr_ = {};
  size_t total_size_ = 0;
  size_t page_size_ = 0;
  size_t num_total_pages_ = 0;
};

}  // namespace xllm

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

#include "core/framework/xtensor/weight_allocation.h"

#include <cstdint>
#include <limits>
#include <utility>

#include "core/virtual_memory/mapped_memory_region.h"

namespace xllm {

WeightAllocation::~WeightAllocation() = default;

WeightAllocation::WeightAllocation(WeightAllocation&& other) noexcept = default;

WeightAllocation& WeightAllocation::operator=(
    WeightAllocation&& other) noexcept = default;

void WeightAllocation::set_contiguous(page_id_t start_page_id,
                                      size_t num_pages,
                                      void* base_ptr,
                                      std::vector<WeightSegment> segments) {
  tensor_.reset();
  start_page_id_ = start_page_id;
  num_pages_ = num_pages;
  base_ptr_ = base_ptr;
  current_offset_ = 0;
  segments_ = std::move(segments);
}

void WeightAllocation::set_fragmented(
    std::unique_ptr<MappedMemoryRegion> tensor,
    size_t num_pages,
    std::vector<WeightSegment> segments) {
  start_page_id_ = -1;
  num_pages_ = num_pages;
  base_ptr_ =
      tensor == nullptr ? nullptr : vir_ptr_to_void_ptr(tensor->vaddr());
  current_offset_ = 0;
  tensor_ = std::move(tensor);
  segments_ = std::move(segments);
}

bool WeightAllocation::allocate(void*& ptr, size_t size, size_t page_size) {
  if (base_ptr_ == nullptr) {
    return false;
  }

  if (tensor_ == nullptr &&
      (page_size == 0 ||
       num_pages_ > std::numeric_limits<size_t>::max() / page_size)) {
    return false;
  }
  const size_t region_size =
      tensor_ == nullptr ? num_pages_ * page_size : tensor_->size();
  if (current_offset_ > region_size || size > region_size - current_offset_) {
    return false;
  }

  ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(base_ptr_) +
                                current_offset_);
  current_offset_ += size;
  return true;
}

void WeightAllocation::reset() {
  start_page_id_ = -1;
  num_pages_ = 0;
  base_ptr_ = nullptr;
  current_offset_ = 0;
  tensor_.reset();
  segments_.clear();
}

}  // namespace xllm

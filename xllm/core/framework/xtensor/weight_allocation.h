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

#include <cstddef>
#include <memory>
#include <vector>

#include "core/common/types.h"
#include "core/framework/xtensor/weight_transfer_segments.h"
#include "core/virtual_memory/physical_page.h"

namespace xllm {

class MappedMemoryRegion;

// Owns the model-local weight reservation and its logical transfer layout.
class WeightAllocation final {
 public:
  WeightAllocation() = default;
  ~WeightAllocation();

  WeightAllocation(const WeightAllocation&) = delete;
  WeightAllocation& operator=(const WeightAllocation&) = delete;
  WeightAllocation(WeightAllocation&&) noexcept;
  WeightAllocation& operator=(WeightAllocation&&) noexcept;

  void set_contiguous(page_id_t start_page_id,
                      size_t num_pages,
                      void* base_ptr,
                      std::vector<WeightSegment> segments);
  void set_fragmented(std::unique_ptr<MappedMemoryRegion> tensor,
                      size_t num_pages,
                      std::vector<WeightSegment> segments);

  bool allocate(void*& ptr, size_t size, size_t page_size);
  void reset();

  page_id_t start_page_id() const { return start_page_id_; }
  size_t num_pages() const { return num_pages_; }
  void* base_ptr() const { return base_ptr_; }
  size_t current_offset() const { return current_offset_; }
  bool is_fragmented() const { return tensor_ != nullptr; }
  MappedMemoryRegion* tensor() const { return tensor_.get(); }
  const std::vector<WeightSegment>& segments() const { return segments_; }

 private:
  page_id_t start_page_id_ = -1;
  size_t num_pages_ = 0;
  void* base_ptr_ = nullptr;
  size_t current_offset_ = 0;
  std::unique_ptr<MappedMemoryRegion> tensor_;
  std::vector<WeightSegment> segments_;
};

}  // namespace xllm

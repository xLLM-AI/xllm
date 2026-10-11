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
#include <vector>

#include "core/common/types.h"
#include "core/framework/allocator/virtual_memory/physical_page.h"

namespace xllm {

class PhysicalPagePool;

struct WeightPageReservation {
  page_id_t contiguous_start = -1;
  std::vector<page_id_t> page_ids;
};

class WeightPageCoordinator final {
 public:
  explicit WeightPageCoordinator(PhysicalPagePool& pool) : pool_(pool) {}
  ~WeightPageCoordinator() = default;

  WeightPageCoordinator(const WeightPageCoordinator&) = delete;
  WeightPageCoordinator& operator=(const WeightPageCoordinator&) = delete;

  WeightPageReservation reserve_weight_pages(size_t num_pages) const;
  void release_weight_pages(const WeightPageReservation& reservation) const;

 private:
  PhysicalPagePool& pool_;
};

}  // namespace xllm

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

#include "core/framework/virtual_memory/weight_page_coordinator.h"

#include <glog/logging.h>

#include "core/virtual_memory/physical_page_pool.h"

namespace xllm {

WeightPageReservation WeightPageCoordinator::reserve_weight_pages(
    size_t num_pages) const {
  WeightPageReservation reservation;
  if (num_pages == 0) {
    return reservation;
  }

  auto& pool = PhysicalPagePool::get_instance();
  reservation.contiguous_start = pool.allocate_contiguous_from_right(num_pages);
  if (reservation.contiguous_start >= 0) {
    reservation.page_ids.reserve(num_pages);
    for (size_t i = 0; i < num_pages; ++i) {
      reservation.page_ids.push_back(reservation.contiguous_start +
                                     static_cast<page_id_t>(i));
    }
    return reservation;
  }

  reservation.page_ids = pool.allocate_pages_from_right(num_pages);
  if (reservation.page_ids.empty()) {
    LOG(ERROR) << "Failed to reserve " << num_pages << " weight pages";
  }
  return reservation;
}

void WeightPageCoordinator::release_weight_pages(
    const WeightPageReservation& reservation) const {
  if (reservation.page_ids.empty()) {
    return;
  }
  PhysicalPagePool::get_instance().release_reserved_pages(reservation.page_ids);
}

}  // namespace xllm

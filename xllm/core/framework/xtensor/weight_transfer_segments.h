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

#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace xllm {

// Segments are consumed in logical weight order, matching the page mapping.
inline std::vector<std::pair<uint64_t, uint64_t>> make_weight_transfer_segments(
    const std::vector<int64_t>& page_ids,
    uint64_t page_size) {
  CHECK_GT(page_size, 0);
  std::vector<std::pair<uint64_t, uint64_t>> segments;
  segments.reserve(page_ids.size());
  for (int64_t page_id : page_ids) {
    CHECK_GE(page_id, 0);
    CHECK_LE(static_cast<uint64_t>(page_id),
             (std::numeric_limits<uint64_t>::max() - page_size) / page_size);
    uint64_t offset = static_cast<uint64_t>(page_id) * page_size;
    if (!segments.empty() &&
        segments.back().first + segments.back().second == offset) {
      segments.back().second += page_size;
      continue;
    }
    segments.emplace_back(offset, page_size);
  }
  return segments;
}

}  // namespace xllm

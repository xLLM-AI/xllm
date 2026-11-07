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
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/util/slice.h"

namespace xllm::seq_lens {

inline int32_t at(const Slice<int32_t>& lens, int32_t seq_idx) {
  CHECK_LT(static_cast<size_t>(seq_idx), lens.size())
      << "seq_idx out of range, seq_idx=" << seq_idx
      << ", seq_lens_size=" << lens.size();
  return lens[static_cast<size_t>(seq_idx)];
}

// Empty lens have no maximum; callers use 0 so the padded metadata stays
// valid instead of reading past the buffer.
inline int32_t max_or_zero(const Slice<int32_t>& lens) {
  return lens.empty() ? 0 : *std::max_element(lens.begin(), lens.end());
}

// `T` widens the running total for callers that index tokens beyond int32
// (e.g. `ParallelInput::query_start_loc`).
template <typename T = int32_t>
void cumulative_into(std::vector<T>& dst, const Slice<int32_t>& lens) {
  dst.clear();
  dst.reserve(lens.size() + 1);
  dst.emplace_back(0);
  for (const int32_t length : lens) {
    dst.emplace_back(dst.back() + static_cast<T>(length));
  }
}

inline std::vector<int32_t> cumulative(const Slice<int32_t>& lens) {
  std::vector<int32_t> cumulative_lens;
  cumulative_into(cumulative_lens, lens);
  return cumulative_lens;
}

// [l0, l0+l1, ...] — each sequence's exclusive end, with no leading zero. Not
// `DSAMetadata::query_start_offsets`, which is `[0, cumsum(...)]`.
inline std::vector<int32_t> query_end_offsets(const Slice<int32_t>& lens) {
  std::vector<int32_t> offsets;
  offsets.reserve(lens.size());
  int32_t running = 0;
  for (const int32_t length : lens) {
    running += length;
    offsets.emplace_back(running);
  }
  return offsets;
}

// Incremental form for buffers cumulative by construction (e.g.
// `paged_kv_indptr`), not for seq-lens buffers.
inline void append_cumulative(std::vector<int32_t>& cumulative_lens,
                              int32_t length) {
  if (cumulative_lens.empty()) {
    cumulative_lens.emplace_back(0);
  }
  cumulative_lens.emplace_back(cumulative_lens.back() + length);
}

}  // namespace xllm::seq_lens

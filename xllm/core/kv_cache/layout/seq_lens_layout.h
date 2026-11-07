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

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/util/slice.h"

namespace xllm {

// Every reader and writer of a seq-lens buffer goes through this type, so the
// backend layout policy lives in one place: NPU stores per-sequence lengths,
// other backends cumulative lengths with a leading zero.
class CuSeqLens final {
 public:
  // Reads one sequence's length out of a buffer in the backend layout.
  static int32_t at(const Slice<int32_t>& seq_lens, int32_t seq_idx) {
    const size_t index = static_cast<size_t>(seq_idx);
#if defined(USE_NPU)
    CHECK_LT(index, seq_lens.size());
    return seq_lens[index];
#else
    CHECK_LT(index + 1, seq_lens.size());
    return seq_lens[index + 1] - seq_lens[index];
#endif
  }

  // Appends one length in the backend layout.
  static void append(std::vector<int32_t>& seq_lens, int32_t length) {
#if defined(USE_NPU)
    seq_lens.emplace_back(length);
#else
    append_cumulative(seq_lens, length);
#endif
  }

  // Appends one length in the cumulative layout [0, l0, l0+l1, ...].
  static void append_cumulative(std::vector<int32_t>& seq_lens,
                                int32_t length) {
    if (seq_lens.empty()) {
      seq_lens.emplace_back(0);
    }
    seq_lens.emplace_back(seq_lens.back() + length);
  }

  // Appends one query-start offset, which carries no leading zero.
  static void append_query_start(std::vector<int32_t>& query_start_offsets,
                                 int32_t length) {
    query_start_offsets.emplace_back(
        (query_start_offsets.empty() ? 0 : query_start_offsets.back()) +
        length);
  }

  // `dst` must already hold entries on cumulative layouts.
  static void merge(std::vector<int32_t>& dst,
                    const std::vector<int32_t>& src) {
#if defined(USE_NPU)
    dst.insert(dst.end(), src.begin(), src.end());
#else
    CHECK(!dst.empty());
    const int32_t offset = dst.back();
    for (size_t i = 1; i < src.size(); ++i) {
      dst.emplace_back(src[i] + offset);
    }
#endif
  }

  // Per-sequence lengths converted into the backend layout.
  static std::vector<int32_t> from_lengths(
      const std::vector<int32_t>& lengths) {
    std::vector<int32_t> seq_lens;
    seq_lens.reserve(capacity_for(lengths.size()));
    for (const int32_t length : lengths) {
      append(seq_lens, length);
    }
    return seq_lens;
  }

  // Per-sequence lengths, inverting the cumulative layout when in use.
  static std::vector<int32_t> to_lengths(const std::vector<int32_t>& seq_lens) {
    if (seq_lens.size() < 2) {
      return {};
    }
    std::vector<int32_t> lengths;
    lengths.reserve(seq_lens.size() - 1);
    for (size_t i = 1; i < seq_lens.size(); ++i) {
      lengths.push_back(seq_lens[i] - seq_lens[i - 1]);
    }
    return lengths;
  }

  // Query-start offsets: a running total without a leading zero.
  static std::vector<int32_t> to_query_start_offsets(
      const std::vector<int32_t>& seq_lens) {
#if defined(USE_NPU)
    std::vector<int32_t> offsets;
    offsets.reserve(seq_lens.size());
    for (const int32_t length : seq_lens) {
      append_query_start(offsets, length);
    }
    return offsets;
#else
    // The cumulative layout already holds exactly these offsets.
    if (seq_lens.empty()) {
      return {};
    }
    return std::vector<int32_t>(seq_lens.begin() + 1, seq_lens.end());
#endif
  }

  static size_t capacity_for(size_t num_sequences) {
#if defined(USE_NPU)
    return num_sequences;
#else
    return num_sequences + 1;
#endif
  }

  static int32_t entry_count(const std::vector<int32_t>& seq_lens) {
#if defined(USE_NPU)
    return static_cast<int32_t>(seq_lens.size());
#else
    return seq_lens.empty() ? 0 : static_cast<int32_t>(seq_lens.size()) - 1;
#endif
  }

  // One query length plus its query-start offset. `leading_zero` selects the
  // ACL graph-bound ABI (idle-rank draft replay and graph-bound spec verify),
  // which binds rows + 1 entries instead of a bare qo_indptr.
  static void append_q_len_and_qo_indptr(std::vector<int32_t>& q_seq_lens,
                                         std::vector<int32_t>& q_cu_seq_lens,
                                         int32_t length,
                                         bool leading_zero) {
    append(q_seq_lens, length);
    if (leading_zero) {
      append_cumulative(q_cu_seq_lens, length);
    } else {
      append_query_start(q_cu_seq_lens, length);
    }
  }

  static void append_kv_len_and_max(std::vector<int32_t>& kv_seq_lens,
                                    int32_t kv_len,
                                    int32_t& kv_max_seq_len) {
    if (kv_len > kv_max_seq_len) {
      kv_max_seq_len = kv_len;
    }
    append(kv_seq_lens, kv_len);
  }
};

}  // namespace xllm

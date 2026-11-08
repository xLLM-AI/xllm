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

#include "core/util/utils.h"

namespace xllm::util {

// Find the range of decode sequence indices (q_seq_lens == 1) in q_seq_lens
// Returns {start_index, end_index} of decode sequences,
// or {-1, -1} if no decode sequences found
std::pair<int32_t, int32_t> find_ones_indices(
    std::vector<int32_t>& q_seq_lens) {
  int32_t left = 0;
  int32_t right = static_cast<int32_t>(q_seq_lens.size()) - 1;
  int32_t start_index = -1;
  int32_t end_index = -1;

  // Binary search for the start index of decode sequences (q_seq_lens == 1)
  while (left < right) {
    int32_t mid = (left + right) / 2;
    if (q_seq_lens[mid] < 1) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  if (q_seq_lens[left] == 1) {
    start_index = left;
  } else {
    return {start_index, end_index};  // No decode sequences found
  }

  left = 0;
  right = static_cast<int32_t>(q_seq_lens.size()) - 1;

  // Binary search for the end index of decode sequences (q_seq_lens == 1)
  while (left < right) {
    int32_t mid = (left + right + 1) / 2;
    if (q_seq_lens[mid] > 1) {
      right = mid - 1;
    } else {
      left = mid;
    }
  }
  if (q_seq_lens[right] == 1) {
    end_index = right;
  }
  return {start_index, end_index};
}

bool match_suffix(const Slice<int32_t>& data, const Slice<int32_t>& suffix) {
  if (suffix.empty()) {
    return true;
  }

  const size_t data_len = data.size();
  const size_t suf_len = suffix.size();
  if (data_len < suf_len) {
    return false;
  }

  const int32_t* data_start = data.data() + (data_len - suf_len);
  const int32_t* data_end = data.data() + data_len;
  return std::equal(data_start, data_end, suffix.data());
}

std::vector<uint32_t> cal_vec_split_index(uint32_t vec_size,
                                          uint32_t part_num) {
  std::vector<uint32_t> split_index;
  split_index.reserve(part_num + 1);
  split_index.emplace_back(/*value=*/0);

  if (part_num == 1) {
    split_index.emplace_back(vec_size);
  } else {
    const uint32_t base = vec_size / part_num;
    const uint32_t remainder = vec_size % part_num;
    for (uint32_t i = 0; i < part_num; ++i) {
      split_index.emplace_back(split_index[i] +
                               ((i < remainder) ? (base + 1) : base));
    }
  }
  return split_index;
}

// find the least power of 2 that is greater than or equal to x
int32_t ceil_pow2(int32_t n) {
  if (n <= 0) {
    return 1;
  }
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return n + 1;
}

}  // namespace xllm::util

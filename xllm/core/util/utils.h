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

#include <glog/logging.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/util/slice.h"

namespace xllm::util {

std::pair<int32_t, int32_t> find_ones_indices(std::vector<int32_t>& q_seq_lens);

template <typename T>
void pad_2d_vector(std::vector<std::vector<T>>& vec, T pad_value) {
  size_t max_col_size = 0;
  for (const auto& row : vec) {
    max_col_size = std::max(max_col_size, row.size());
  }

  for (auto& row : vec) {
    row.resize(max_col_size, pad_value);
  }
}

template <typename T>
T sum(const std::vector<T>& vec) {
  if (vec.empty()) {
    LOG(FATAL) << "vector is empty.";
  }
  return std::accumulate(vec.begin(), vec.end(), T{});
}

template <typename T>
const T& min(const std::vector<T>& vec) {
  if (vec.empty()) {
    LOG(FATAL) << "vector is empty.";
  }
  return *std::min_element(vec.begin(), vec.end());
}

template <typename T>
const T& max(const std::vector<T>& vec) {
  if (vec.empty()) {
    LOG(FATAL) << "vector is empty.";
  }
  return *std::max_element(vec.begin(), vec.end());
}

template <typename T>
inline std::enable_if_t<std::is_integral_v<T>, T> ceil_div(T value, T divisor) {
  CHECK_GT(divisor, 0) << "divisor must be positive.";
  return value / divisor + static_cast<T>(value % divisor != 0);
}

static inline int64_t align_up(int64_t value, int64_t alignment) {
  if (alignment == 0) {
    return value;
  }
  return ((value + alignment - 1) / alignment) * alignment;
}

bool match_suffix(const Slice<int32_t>& data, const Slice<int32_t>& suffix);

std::vector<uint32_t> cal_vec_split_index(uint32_t vec_size, uint32_t part_num);

int32_t ceil_pow2(int32_t n);

}  // namespace xllm::util

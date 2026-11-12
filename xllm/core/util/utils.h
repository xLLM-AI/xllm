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
#include <vector>

namespace xllm::util {

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

int32_t ceil_pow2(int32_t n);

}  // namespace xllm::util

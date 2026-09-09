/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ===========================================================================*/

// ref to:
// https://github.com/vllm-project/vllm/blob/main/csrc/core/math.hpp

#pragma once

#include <climits>
#include <iostream>

namespace xllm {
namespace kernel {
namespace cuda {

inline constexpr uint32_t next_pow_2(uint32_t const num) {
  if (num <= 1) return num;
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

template <typename A, typename B>
static inline constexpr auto div_ceil(A a, B b) {
  return (a + b - 1) / b;
}

// Round a down to the next multiple of b.
// Caller must ensure b is non-zero.
template <typename T>
inline constexpr T round_to_previous_multiple_of(T a, T b) {
  return a % b == 0 ? a : (a / b) * b;
}

// Round a up to the next multiple of b.
// Caller must ensure b is non-zero.
template <typename T>
inline constexpr T round_to_next_multiple_of(T a, T b) {
  return a % b == 0 ? a : ((a / b) + 1) * b;
}

}  // namespace cuda
}  // namespace kernel
}  // namespace xllm

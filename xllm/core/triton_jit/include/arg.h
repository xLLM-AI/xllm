/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <c10/core/ScalarType.h>
#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <type_traits>

namespace xllm::triton_jit {
inline constexpr c10::ScalarType kNullPtrDtype = c10::ScalarType::Float;

// Specialization hint derived from a pointer address or an integer value.
enum class SpecHint { NONE, DIV16, EQ1 };

inline SpecHint hint_from_ptr(void* p) {
  std::uintptr_t v = reinterpret_cast<std::uintptr_t>(p);
  if (v % 16 == 0) {
    return SpecHint::DIV16;
  }
  if (v == 1) {
    return SpecHint::EQ1;
  }
  return SpecHint::NONE;
}

template <typename T>
inline SpecHint hint_from_int(T v) {
  static_assert(std::is_integral_v<T>,
                "hint_from_int expects an integral value");
  if (v % 16 == 0) {
    return SpecHint::DIV16;
  }
  if (v == 1) {
    return SpecHint::EQ1;
  }
  return SpecHint::NONE;
}

template <typename T>
struct is_optional_tensor : std::false_type {};
template <>
struct is_optional_tensor<std::optional<torch::Tensor>> : std::true_type {};
template <typename T>
inline constexpr bool is_optional_tensor_v = is_optional_tensor<T>::value;

}  // namespace xllm::triton_jit

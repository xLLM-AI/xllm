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
#include <glog/logging.h>

#include <cstdint>
#include <type_traits>

namespace xllm::triton_jit {

inline const char* scalar_to_triton(c10::ScalarType t) {
  switch (t) {
    case c10::ScalarType::Float: {
      return "fp32";
    }
    case c10::ScalarType::Double: {
      return "fp64";
    }
    case c10::ScalarType::Half: {
      return "fp16";
    }
    case c10::ScalarType::BFloat16: {
      return "bf16";
    }
    case c10::ScalarType::Int: {
      return "i32";
    }
    case c10::ScalarType::Long: {
      return "i64";
    }
    case c10::ScalarType::Short: {
      return "i16";
    }
    case c10::ScalarType::UInt32: {
      return "u32";
    }
    case c10::ScalarType::UInt64: {
      return "u64";
    }
    case c10::ScalarType::UInt16: {
      return "u16";
    }
    case c10::ScalarType::Char: {
      return "i8";
    }
    case c10::ScalarType::Byte: {
      return "u8";
    }
    case c10::ScalarType::Bool: {
      return "u1";
    }
    default: {
      LOG(FATAL) << "triton_jit: unsupported tensor scalar type";
    }
  }
  return "";
}

template <typename T>
inline constexpr const char* cpp_to_triton() {
  using U = std::remove_cv_t<std::remove_reference_t<T>>;
  if constexpr (std::is_same_v<U, bool>) {
    return "i1";
  } else if constexpr (std::is_same_v<U, int32_t> || std::is_same_v<U, int>) {
    return "i32";
  } else if constexpr (std::is_same_v<U, uint32_t> ||
                       std::is_same_v<U, unsigned int>) {
    return "u32";
  } else if constexpr (std::is_same_v<U, int64_t>) {
    return "i64";
  } else if constexpr (std::is_same_v<U, uint64_t>) {
    return "u64";
  } else if constexpr (std::is_same_v<U, float>) {
    return "fp32";
  } else if constexpr (std::is_same_v<U, double>) {
    return "fp64";
  } else {
    static_assert(sizeof(T) == 0,
                  "triton_jit: unsupported C++ scalar type for cpp_to_triton");
    return "";
  }
}

}  // namespace xllm::triton_jit

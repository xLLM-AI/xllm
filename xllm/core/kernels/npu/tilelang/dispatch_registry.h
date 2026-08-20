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

#include <c10/core/ScalarType.h>
#include <glog/logging.h>

#include <array>
#include <cstddef>
#include <sstream>
#include <string>
#include <type_traits>

namespace xllm::kernel::npu::tilelang {

enum class TilelangDType {
  kBF16,
  kFP16,
  kFP32,
  kFloat16,
  kFloat32,
  kInt8,
  kInt32,
  kUInt8,
};

inline TilelangDType to_tilelang_dtype(c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::BFloat16:
      return TilelangDType::kBF16;
    case c10::ScalarType::Half:
      return TilelangDType::kFloat16;
    case c10::ScalarType::Float:
      return TilelangDType::kFloat32;
    case c10::ScalarType::Char:
      return TilelangDType::kInt8;
    case c10::ScalarType::Int:
      return TilelangDType::kInt32;
    case c10::ScalarType::Byte:
      return TilelangDType::kUInt8;
    default:
      LOG(FATAL) << "TileLang: unsupported dtype " << dtype;
  }
  return TilelangDType::kBF16;
}

template <typename Fn>
using function_type_t = std::remove_pointer_t<Fn>;

template <typename Specialization, typename Fn>
struct KernelEntry {
  Specialization spec;
  const char* variant_key;
  Fn fn;
};

template <typename Entry, std::size_t N, typename Specialization>
inline const Entry* find_kernel_entry(const std::array<Entry, N>& registry,
                                      const Specialization& specialization) {
  for (const auto& entry : registry) {
    if (entry.spec == specialization) {
      return &entry;
    }
  }
  return nullptr;
}

template <typename Entry, std::size_t N>
inline std::string available_variant_keys(
    const std::array<Entry, N>& registry) {
  std::ostringstream oss;
  bool first = true;
  for (const auto& entry : registry) {
    if (!first) {
      oss << ", ";
    }
    first = false;
    oss << entry.variant_key;
  }
  return oss.str();
}

}  // namespace xllm::kernel::npu::tilelang

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

#include <xxHash/xxhash.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <string_view>

namespace xllm {

constexpr uint32_t XXH3_128BITS_HASH_VALUE_LEN = sizeof(XXH128_hash_t);

class XXH3Key final {
 public:
  XXH3Key() = default;
  explicit XXH3Key(const uint8_t* input_data) {
    std::memcpy(data, input_data, sizeof(data));
  }

  bool operator==(const XXH3Key& other) const {
    return std::memcmp(data, other.data, sizeof(data)) == 0;
  }

  void set(const uint8_t* input_data) {
    std::memcpy(data, input_data, sizeof(data));
  }

  std::string debug_string() const {
    std::string result;
    result.reserve(sizeof(data) * 4);
    for (uint8_t byte : data) {
      result += std::to_string(static_cast<int64_t>(byte));
      result += ' ';
    }
    return result;
  }

  // Unset keys are deterministic all-zero values, rejectable downstream.
  uint8_t data[XXH3_128BITS_HASH_VALUE_LEN]{};
};

class FixedStringKeyHash final {
 public:
  size_t operator()(const XXH3Key& key) const {
    return std::hash<std::string_view>()(std::string_view(
        reinterpret_cast<const char*>(key.data), sizeof(key.data)));
  }
};

class FixedStringKeyEqual final {
 public:
  bool operator()(const XXH3Key& left, const XXH3Key& right) const {
    return left == right;
  }
};

XXH3Key hash_bytes(const void* data, size_t size, uint64_t seed);
XXH3Key hash_string(std::string_view value, uint64_t seed);

using PrefixHash = std::array<uint8_t, XXH3_128BITS_HASH_VALUE_LEN>;

inline bool is_zero_prefix_hash(const PrefixHash& prefix_hash) {
  return std::all_of(prefix_hash.begin(), prefix_hash.end(), [](uint8_t value) {
    return value == 0;
  });
}

}  // namespace xllm

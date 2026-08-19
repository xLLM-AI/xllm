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

#include <torch/torch.h>
#include <xxHash/xxhash.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace xllm {

constexpr uint32_t XXH3_128BITS_HASH_VALUE_LEN = sizeof(XXH128_hash_t);

struct XXH3Key {
  uint8_t data[XXH3_128BITS_HASH_VALUE_LEN];

  XXH3Key() {}
  XXH3Key(const uint8_t* const input_data) {
    std::memcpy(data, input_data, XXH3_128BITS_HASH_VALUE_LEN);
  }

  bool operator==(const XXH3Key& other) const {
    return std::memcmp(data, other.data, XXH3_128BITS_HASH_VALUE_LEN) == 0;
  }

  void set(const uint8_t* const input_data) {
    std::memcpy(data, input_data, XXH3_128BITS_HASH_VALUE_LEN);
  }

  std::string debug_string() const {
    std::string rt;
    for (int i = 0; i < XXH3_128BITS_HASH_VALUE_LEN; i++) {
      rt += std::to_string(int64_t(data[i])) + " ";
    }

    return rt;
  }
};

struct FixedStringKeyHash {
  size_t operator()(const XXH3Key& key) const {
    return std::hash<std::string_view>()(std::string_view(
        reinterpret_cast<const char*>(key.data), sizeof(key.data)));
  }
};

struct FixedStringKeyEqual {
  bool operator()(const XXH3Key& left, const XXH3Key& right) const {
    return std::memcmp(left.data, right.data, sizeof(left.data)) == 0;
  }
};

XXH3Key hash_tensor(const torch::Tensor& tensor);
XXH3Key hash_string(const std::string& str);

using PrefixHash = std::array<uint8_t, XXH3_128BITS_HASH_VALUE_LEN>;

inline bool is_zero_prefix_hash(const PrefixHash& prefix_hash) {
  return std::all_of(
      prefix_hash.begin(), prefix_hash.end(), [](uint8_t v) { return v == 0; });
}

}  // namespace xllm

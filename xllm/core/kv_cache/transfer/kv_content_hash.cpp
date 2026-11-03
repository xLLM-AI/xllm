/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/kv_cache/transfer/kv_content_hash.h"

#include <glog/logging.h>

#include <algorithm>
#include <array>
#include <cstring>

#include "util/hash_util.h"
#include "util/slice.h"

namespace xllm {
namespace {

constexpr char kHexDigits[] = "0123456789abcdef";

}  // namespace

std::vector<std::string> compute_kv_block_hashes(
    const std::vector<int32_t>& token_ids,
    uint32_t block_size,
    uint32_t seed) {
  CHECK_GT(block_size, 0);
  std::vector<std::string> hashes;
  if (token_ids.empty()) {
    return hashes;
  }

  const Slice<int32_t> tokens(token_ids);
  const size_t block_size_value = static_cast<size_t>(block_size);
  const size_t block_count =
      (token_ids.size() + block_size_value - 1) / block_size_value;
  hashes.reserve(block_count);

  std::array<uint8_t, XXH3_128BITS_HASH_VALUE_LEN> previous{};
  std::array<uint8_t, XXH3_128BITS_HASH_VALUE_LEN> current{};

  for (size_t block_idx = 0; block_idx < block_count; ++block_idx) {
    const size_t begin = block_idx * block_size_value;
    const size_t end = std::min(begin + block_size_value, token_ids.size());
    const Slice<int32_t> block_tokens = tokens.slice(begin, end);

    std::vector<uint8_t> input;
    input.reserve(previous.size() +
                  block_tokens.size() * sizeof(int32_t));
    if (block_idx > 0) {
      input.insert(input.end(), previous.begin(), previous.end());
    }
    const auto* token_bytes =
        reinterpret_cast<const uint8_t*>(block_tokens.data());
    input.insert(input.end(),
                 token_bytes,
                 token_bytes + block_tokens.size() * sizeof(int32_t));

    const XXH128_hash_t value = XXH3_128bits_withSeed(
        input.data(), input.size(), static_cast<XXH64_hash_t>(seed));
    std::memcpy(current.data(), &value, sizeof(value));

    std::string hex;
    hex.reserve(XXH3_128BITS_HASH_VALUE_LEN * 2);
    for (uint8_t byte : current) {
      hex.push_back(kHexDigits[byte >> 4]);
      hex.push_back(kHexDigits[byte & 0x0f]);
    }
    hashes.emplace_back(std::move(hex));
    previous = current;
  }
  return hashes;
}

}  // namespace xllm

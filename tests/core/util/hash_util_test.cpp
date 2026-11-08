/* Copyright 2026 The xLLM Authors.

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

#include "core/util/hash_util.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string_view>
#include <unordered_map>

namespace xllm {
namespace {

TEST(HashUtilTest, PreservesSeededXxh128NativeRepresentation) {
  struct HashCase {
    std::string_view input;
    uint64_t seed;
    XXH128_hash_t expected;
  };

  // The persisted key format contains the native XXH128 low/high words,
  // rather than the byte order used by XXH128_canonicalFromHash.
  const std::array<HashCase, 9> cases = {{
      {"", 0, {0x6001c324468d497fULL, 0x99aa06d3014798d8ULL}},
      {"", 42, {0x3c1d09e9fe249164ULL, 0x16c20acd33f7af2fULL}},
      {"",
       std::numeric_limits<uint64_t>::max(),
       {0x2d10110a247d19ddULL, 0x5334ec22748b5fcdULL}},
      {"content", 0, {0xb27a2d2388d9568cULL, 0x917e1274274fc195ULL}},
      {"content", 42, {0x90de721d07526530ULL, 0x59fc1c67ee7bcf9dULL}},
      {"content",
       std::numeric_limits<uint64_t>::max(),
       {0x355b5f5d143f34f2ULL, 0x1644c7c31ea1a271ULL}},
      {std::string_view("a\0b", 3),
       0,
       {0xd5a06cd078125351ULL, 0x39797789ed4c7ea0ULL}},
      {std::string_view("a\0b", 3),
       42,
       {0xded1de69c192a4b0ULL, 0xc4dfdbc5f61e3debULL}},
      {std::string_view("a\0b", 3),
       std::numeric_limits<uint64_t>::max(),
       {0x657ae23ae06768e6ULL, 0x166fb51e8b43cca4ULL}},
  }};

  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.seed);
    SCOPED_TRACE(test_case.input.size());
    const XXH3Key string_key = hash_string(test_case.input, test_case.seed);
    const XXH3Key byte_key = hash_bytes(
        test_case.input.data(), test_case.input.size(), test_case.seed);
    EXPECT_EQ(
        0,
        std::memcmp(
            string_key.data, &test_case.expected, sizeof(test_case.expected)));
    EXPECT_EQ(string_key, byte_key);
  }
}

TEST(HashUtilTest, HashesBorrowedViewsWithoutReadingTheirTerminator) {
  const std::array<char, 5> storage = {'x', 'a', '\0', 'b', 'y'};
  const std::string_view binary_view(storage.data() + 1, 3);
  const std::string_view short_view(storage.data() + 3, 1);
  EXPECT_EQ(hash_string(std::string_view("a\0b", 3), /*seed=*/42),
            hash_string(binary_view, /*seed=*/42));
  EXPECT_EQ(hash_string("b", /*seed=*/42),
            hash_string(short_view, /*seed=*/42));
  EXPECT_FALSE(hash_string(binary_view, /*seed=*/42) ==
               hash_string("a", /*seed=*/42));
}

TEST(HashUtilTest, ComparesAllKeyBytesAndSupportsCacheLookup) {
  const XXH3Key key = hash_string("content", /*seed=*/42);
  XXH3Key copy(key.data);
  EXPECT_EQ(key, copy);
  EXPECT_TRUE(FixedStringKeyEqual()(key, copy));

  copy.data[XXH3_128BITS_HASH_VALUE_LEN - 1] ^= 1;
  EXPECT_FALSE(key == copy);
  EXPECT_FALSE(FixedStringKeyEqual()(key, copy));
  copy.set(key.data);
  EXPECT_EQ(key, copy);

  std::unordered_map<XXH3Key, int32_t, FixedStringKeyHash, FixedStringKeyEqual>
      cache;
  cache.emplace(key, 7);
  EXPECT_EQ(7, cache.at(hash_string("content", /*seed=*/42)));
  EXPECT_EQ(cache.end(), cache.find(hash_string("content", /*seed=*/43)));
}

TEST(HashUtilTest, LeavesUnsetKeysAndPrefixHashesZeroInitialized) {
  PrefixHash prefix{};
  EXPECT_EQ(XXH3Key(), XXH3Key(prefix.data()));
  EXPECT_TRUE(is_zero_prefix_hash(prefix));
  prefix.back() = 1;
  EXPECT_FALSE(is_zero_prefix_hash(prefix));
}

}  // namespace
}  // namespace xllm

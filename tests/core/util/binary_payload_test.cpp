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

#include "core/util/binary_payload.h"

#include <gtest/gtest.h>

#include <array>
#include <string>

namespace xllm {
namespace {

butil::IOBuf make_segmented_payload() {
  butil::IOBuf first;
  butil::IOBuf second;
  butil::IOBuf third;
  first.append("abcd", 4);
  second.append("efgh", 4);
  third.append("ijkl", 4);

  butil::IOBuf payload;
  payload.append(first);
  payload.append(second);
  payload.append(third);
  return payload;
}

TEST(BinaryPayloadTest, CopiesRangeAcrossBlocks) {
  BinaryPayload payload(make_segmented_payload());
  std::array<char, 6> output{};

  ASSERT_TRUE(payload.copy_to(/*offset=*/3, output.size(), output.data()));
  EXPECT_EQ(std::string(output.data(), output.size()), "defghi");
}

TEST(BinaryPayloadTest, CopiesRangeToString) {
  BinaryPayload payload(make_segmented_payload());
  std::string output;

  ASSERT_TRUE(payload.copy_to(/*offset=*/4, /*length=*/5, output));
  EXPECT_EQ(output, "efghi");
}

TEST(BinaryPayloadTest, RejectsInvalidRange) {
  BinaryPayload payload(make_segmented_payload());
  std::string output = "unchanged";

  EXPECT_FALSE(payload.copy_to(/*offset=*/11, /*length=*/2, output));
  EXPECT_EQ(output, "unchanged");
  EXPECT_FALSE(payload.copy_to(/*offset=*/13, /*length=*/0, output));
}

TEST(BinaryPayloadTest, KeepsBlocksAliveAfterSourceRelease) {
  butil::IOBuf source = make_segmented_payload();
  BinaryPayload payload(source);
  source.clear();
  std::string output;

  ASSERT_TRUE(payload.copy_to(/*offset=*/0, payload.size(), output));
  EXPECT_EQ(output, "abcdefghijkl");
}

}  // namespace
}  // namespace xllm

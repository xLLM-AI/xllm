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

#include "api_service/mm_service_utils.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "common.pb.h"

namespace xllm::mm_service_utils {
namespace {

TEST(MMServiceUtilsTest, ReservesCombinedPayloadCapacity) {
  const std::vector<std::string> outputs = {"image", "audio", "video"};
  std::string payload;

  ASSERT_TRUE(reserve_binary_payload(
      outputs,
      [](const std::string& output) -> std::string_view { return output; },
      payload));

  EXPECT_GE(payload.capacity(), 15u);
  EXPECT_TRUE(payload.empty());
}

TEST(MMServiceUtilsTest, AppendsPayloadAndTracksOffsets) {
  std::string payload;
  proto::BinaryRef first;
  proto::BinaryRef second;

  append_binary_payload("audio", first, payload);
  append_binary_payload("video", second, payload);

  EXPECT_EQ(first.offset(), 0u);
  EXPECT_EQ(first.length(), 5u);
  EXPECT_EQ(second.offset(), 5u);
  EXPECT_EQ(second.length(), 5u);
  EXPECT_EQ(payload, "audiovideo");
}

TEST(MMServiceUtilsTest, SupportsEmptyPayloadEntry) {
  std::string payload = "prefix";
  proto::BinaryRef binary_ref;

  append_binary_payload("", binary_ref, payload);

  EXPECT_EQ(binary_ref.offset(), 6u);
  EXPECT_EQ(binary_ref.length(), 0u);
  EXPECT_EQ(payload, "prefix");
}

TEST(MMServiceUtilsTest, FillsBase64MediaSource) {
  proto::MediaSource source;
  source.mutable_binary()->set_length(1);
  std::string payload;

  fill_media_source("media", "audio", false, source, payload);

  EXPECT_EQ(source.type(), "base64");
  EXPECT_EQ(source.name(), "audio");
  EXPECT_EQ(source.base64(), "bWVkaWE=");
  EXPECT_FALSE(source.has_binary());
  EXPECT_TRUE(payload.empty());
}

TEST(MMServiceUtilsTest, FillsBinaryMediaSource) {
  proto::MediaSource source;
  source.set_base64("stale");
  std::string payload = "prefix";

  fill_media_source("media", "video", true, source, payload);

  EXPECT_EQ(source.type(), "binary");
  EXPECT_EQ(source.name(), "video");
  ASSERT_TRUE(source.has_binary());
  EXPECT_EQ(source.binary().offset(), 6u);
  EXPECT_EQ(source.binary().length(), 5u);
  EXPECT_TRUE(source.base64().empty());
  EXPECT_EQ(payload, "prefixmedia");
}

}  // namespace
}  // namespace xllm::mm_service_utils

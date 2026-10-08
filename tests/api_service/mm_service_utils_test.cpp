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

#include <memory>
#include <string>
#include <vector>

#include "common.pb.h"
#include "multimodal.pb.h"

namespace xllm::mm_service_utils {
namespace {

class FakeCall final {
 public:
  bool finish_with_error(const StatusCode& code,
                         const std::string& error_message) {
    code_ = code;
    error_message_ = error_message;
    return true;
  }

  StatusCode code() const { return code_; }
  const std::string& error_message() const { return error_message_; }

 private:
  StatusCode code_ = StatusCode::OK;
  std::string error_message_;
};

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

TEST(MMServiceUtilsTest, RejectsInvalidMessageContentType) {
  proto::MMChatRequest request;
  proto::MMInputData* content = request.add_messages()->add_content();
  content->set_type("invalid");
  auto call = std::make_shared<FakeCall>();
  std::vector<Message> messages;

  EXPECT_FALSE(
      build_messages(request.messages(), messages, call, /*image_limit=*/1));
  EXPECT_EQ(call->code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(call->error_message(), "message content type is invalid.");
}

TEST(MMServiceUtilsTest, RejectsMessageExceedingImageLimit) {
  proto::MMChatRequest request;
  proto::MMChatMessage* message = request.add_messages();
  for (const std::string& url : {"first", "second"}) {
    proto::MMInputData* content = message->add_content();
    content->set_type("image_url");
    content->mutable_image_url()->set_url(url);
  }
  auto call = std::make_shared<FakeCall>();
  std::vector<Message> messages;

  EXPECT_FALSE(
      build_messages(request.messages(), messages, call, /*image_limit=*/1));
  EXPECT_EQ(call->code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(call->error_message(),
            "Number of images in a single message exceeds the allowed image "
            "limit.");
}

TEST(MMServiceUtilsTest, PreservesRequestStrings) {
  proto::MMChatRequest request;
  proto::MMChatMessage* message = request.add_messages();
  message->set_role("user");
  const std::string payload(4096, 'x');
  proto::MMInputData* text = message->add_content();
  text->set_type("text");
  text->set_text(payload);
  proto::MMInputData* image = message->add_content();
  image->set_type("image_url");
  image->mutable_image_url()->set_url(payload);
  (*image->mutable_image_url()->mutable_headers())["key"] = "image";
  proto::MMInputData* video = message->add_content();
  video->set_type("video_url");
  video->mutable_video_url()->set_url(payload);
  (*video->mutable_video_url()->mutable_headers())["key"] = "video";
  proto::MMInputData* audio = message->add_content();
  audio->set_type("audio_url");
  audio->mutable_audio_url()->set_url(payload);
  (*audio->mutable_audio_url()->mutable_headers())["key"] = "audio";
  auto call = std::make_shared<FakeCall>();
  std::vector<Message> messages;

  ASSERT_TRUE(
      build_messages(request.messages(), messages, call, /*image_limit=*/1));
  ASSERT_TRUE(text->has_text());
  EXPECT_EQ(text->text(), payload);
  EXPECT_EQ(image->image_url().url(), payload);
  EXPECT_EQ(video->video_url().url(), payload);
  EXPECT_EQ(audio->audio_url().url(), payload);
  ASSERT_EQ(messages.size(), 1u);
  const auto& contents = std::get<MMContentVec>(messages.front().content);
  ASSERT_EQ(contents.size(), 4u);
  EXPECT_EQ(contents[0].text, payload);
  EXPECT_EQ(contents[1].image_url.url, payload);
  EXPECT_EQ(contents[2].video_url.url, payload);
  EXPECT_EQ(contents[3].audio_url.url, payload);
  EXPECT_EQ(contents[1].image_url.headers.at("key"), "image");
  EXPECT_EQ(contents[2].video_url.headers.at("key"), "video");
  EXPECT_EQ(contents[3].audio_url.headers.at("key"), "audio");
}

TEST(MMServiceUtilsTest, PreservesTextOnValidationFailure) {
  proto::MMChatRequest request;
  proto::MMChatMessage* message = request.add_messages();
  proto::MMInputData* text = message->add_content();
  text->set_type("text");
  const std::string payload(4096, 'x');
  text->set_text(payload);
  message->add_content()->set_type("invalid");
  auto call = std::make_shared<FakeCall>();
  std::vector<Message> messages;

  EXPECT_FALSE(
      build_messages(request.messages(), messages, call, /*image_limit=*/1));
  EXPECT_EQ(call->code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_TRUE(text->has_text());
  EXPECT_EQ(text->text(), payload);
}

TEST(MMServiceUtilsTest, BuildsValidMessages) {
  proto::MMChatRequest request;
  proto::MMChatMessage* message = request.add_messages();
  message->set_role("user");
  proto::MMInputData* content = message->add_content();
  content->set_type("text");
  content->set_text("hello");
  auto call = std::make_shared<FakeCall>();
  std::vector<Message> messages;

  ASSERT_TRUE(
      build_messages(request.messages(), messages, call, /*image_limit=*/1));
  ASSERT_EQ(messages.size(), 1u);
  EXPECT_EQ(messages.front().role, "user");
  EXPECT_EQ(call->code(), StatusCode::OK);
  EXPECT_TRUE(call->error_message().empty());
}

}  // namespace
}  // namespace xllm::mm_service_utils

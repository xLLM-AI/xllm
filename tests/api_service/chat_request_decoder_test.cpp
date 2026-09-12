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

#include "api_service/chat_request_decoder.h"

#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/message_differencer.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "chat.pb.h"
#include "multimodal.pb.h"

namespace xllm {
namespace {

enum class ExpectedPath : int8_t {
  FAST,
  FALLBACK,
};

// The contract of the decoder: whatever it produces must be what protobuf's
// JSON parser -- the reference the chat handler has always used -- produces
// for the same body, success or failure alike. `path` additionally pins
// whether the body was in canonical shape, so a corpus entry meant to exercise
// the fast path cannot silently pass through the fallback.
template <typename RequestT>
void expect_matches_reference(const std::string& body, ExpectedPath path) {
  RequestT reference;
  google::protobuf::util::JsonParseOptions options;
  options.ignore_unknown_fields = true;
  const auto reference_status =
      google::protobuf::util::JsonStringToMessage(body, &reference, options);

  RequestT decoded;
  const Status status = decode_chat_request(body, &decoded);

  ASSERT_EQ(status.ok(), reference_status.ok())
      << "body: " << body << "\ndecoder: " << status.message()
      << "\nreference: " << reference_status.ToString();
  if (!reference_status.ok()) {
    EXPECT_EQ(status.message(), reference_status.ToString());
    return;
  }
  EXPECT_TRUE(
      google::protobuf::util::MessageDifferencer::Equals(decoded, reference))
      << "body: " << body << "\ndecoded:\n"
      << decoded.DebugString() << "\nreference:\n"
      << reference.DebugString();
  EXPECT_EQ(locate_chat_struct_members(body).has_value(),
            path == ExpectedPath::FAST)
      << "body: " << body;
}

const char* const kPlainChat =
    R"({"model":"m","messages":[{"role":"system","content":"You help."},)"
    R"({"role":"user","content":"Hi \"there\" \\ \n \u00e9"}],)"
    R"("max_tokens":128,"temperature":0.7,"top_p":0.9,"stream":true,)"
    R"("stop":["</s>"],"stop_token_ids":[2,3],"top_k":40,"priority":"HIGH",)"
    R"("stream_options":{"include_usage":true},"n":1,"logprobs":true,)"
    R"("top_logprobs":5,"ignore_eos":false,"unknown_field":{"x":[1,2]}})";

const char* const kKwargsOnly =
    R"({"messages":[{"role":"user","content":"Hi"}],)"
    R"("chat_template_kwargs":{"enable_thinking":false,"depth":2,)"
    R"("ratio":-1.5e3,"name":"uni\u00e9 \"q\" \ud83d\ude00","none":null,)"
    R"("nested":{"a":[1,2.5,"x",null,true,{"b":[]}],"c":{}}}})";

const char* const kToolsOnly =
    R"({"messages":[{"role":"user","content":"Weather?"}],"tools":[)"
    R"({"type":"function","function":{"name":"get_weather",)"
    R"("description":"Weather for a city.","parameters":{"type":"object",)"
    R"("properties":{"city":{"type":"string"},"unit":{"type":"string",)"
    R"("enum":["celsius","fahrenheit"]},"days":{"type":"integer",)"
    R"("minimum":1,"maximum":14}},"required":["city"],)"
    R"("additionalProperties":false}}},)"
    R"({"type":"function","function":{"name":"no_parameters"}},)"
    R"({"type":"function","function":{"name":"null_parameters",)"
    R"("parameters":null}},)"
    R"({"type":"function","function":{"name":"empty_parameters",)"
    R"("parameters":{}}}],"tool_choice":"auto"})";

// chat_template_kwargs after tools, so the blanked spans are out of order.
const char* const kToolsThenKwargs =
    R"({"messages":[{"role":"user","content":"Hi"}],)"
    R"("tools":[{"type":"function","function":{"name":"f",)"
    R"("parameters":{"type":"object","properties":{"q":{"type":"string"}}}}},)"
    R"({"type":"function","function":{"name":"g",)"
    R"("parameters":{"type":"object"}}}],)"
    R"("chat_template_kwargs":{"enable_thinking":true},"max_tokens":8})";

const char* const kAssistantToolCalls =
    R"({"messages":[{"role":"user","content":"Weather in Paris?"},)"
    R"({"role":"assistant","content":null,"tool_calls":[{"id":"call_1",)"
    R"("type":"function","function":{"name":"get_weather",)"
    R"("arguments":"{\"city\":\"Paris\"}"}}]},)"
    R"({"role":"tool","tool_call_id":"call_1","content":"18C"}]})";

const char* const kMultimodal =
    R"({"model":"vlm","messages":[{"role":"user","content":[)"
    R"({"type":"text","text":"What is this?"},)"
    R"({"type":"image_url","image_url":{"url":"data:image/png;base64,abc",)"
    R"("headers":{"Authorization":"Bearer t"}}}]}],)"
    R"("tools":[{"type":"function","function":{"name":"describe",)"
    R"("parameters":{"type":"object","properties":{}}}}],)"
    R"("chat_template_kwargs":{"enable_thinking":false},"max_tokens":64})";

}  // namespace

TEST(ChatRequestDecoderTest, PlainChatTakesTheFastPath) {
  expect_matches_reference<proto::ChatRequest>(kPlainChat, ExpectedPath::FAST);
  const auto spans = locate_chat_struct_members(kPlainChat);
  ASSERT_TRUE(spans.has_value());
  EXPECT_FALSE(spans->chat_template_kwargs.has_value());
  EXPECT_TRUE(spans->tool_parameters.empty());
}

TEST(ChatRequestDecoderTest, ChatTemplateKwargsAreCarvedOut) {
  expect_matches_reference<proto::ChatRequest>(kKwargsOnly, ExpectedPath::FAST);
  const auto spans = locate_chat_struct_members(kKwargsOnly);
  ASSERT_TRUE(spans.has_value());
  ASSERT_TRUE(spans->chat_template_kwargs.has_value());
  const std::string body = kKwargsOnly;
  const JsonSpan& span = spans->chat_template_kwargs.value();
  EXPECT_EQ(body[span.begin], '{');
  EXPECT_EQ(body[span.end - 1], '}');

  proto::ChatRequest decoded;
  ASSERT_TRUE(decode_chat_request(body, &decoded).ok());
  ASSERT_TRUE(decoded.has_chat_template_kwargs());
  const auto& fields = decoded.chat_template_kwargs().fields();
  EXPECT_FALSE(fields.at("enable_thinking").bool_value());
  EXPECT_DOUBLE_EQ(fields.at("depth").number_value(), 2.0);
  EXPECT_DOUBLE_EQ(fields.at("ratio").number_value(), -1500.0);
  EXPECT_EQ(fields.at("name").string_value(),
            "uni\xc3\xa9 \"q\" \xf0\x9f\x98\x80");
  EXPECT_TRUE(fields.at("none").has_null_value());
  EXPECT_EQ(fields.at("nested")
                .struct_value()
                .fields()
                .at("a")
                .list_value()
                .values_size(),
            6);
}

TEST(ChatRequestDecoderTest, ToolParametersAreCarvedOutPerTool) {
  expect_matches_reference<proto::ChatRequest>(kToolsOnly, ExpectedPath::FAST);
  const auto spans = locate_chat_struct_members(kToolsOnly);
  ASSERT_TRUE(spans.has_value());
  // Tools 0 and 3 carry an object schema; 1 has none and 2 has null.
  ASSERT_EQ(spans->tool_parameters.size(), 2U);
  EXPECT_EQ(spans->tool_parameters[0].first, 0U);
  EXPECT_EQ(spans->tool_parameters[1].first, 3U);

  proto::ChatRequest decoded;
  ASSERT_TRUE(decode_chat_request(std::string(kToolsOnly), &decoded).ok());
  ASSERT_EQ(decoded.tools_size(), 4);
  EXPECT_TRUE(decoded.tools(0).function().has_parameters());
  EXPECT_EQ(decoded.tools(0)
                .function()
                .parameters()
                .fields()
                .at("required")
                .list_value()
                .values(0)
                .string_value(),
            "city");
  EXPECT_FALSE(decoded.tools(1).function().has_parameters());
  EXPECT_FALSE(decoded.tools(2).function().has_parameters());
  EXPECT_TRUE(decoded.tools(3).function().has_parameters());
  EXPECT_EQ(decoded.tools(3).function().parameters().fields_size(), 0);
}

TEST(ChatRequestDecoderTest, StructMembersOutOfOrderAreBlankedCorrectly) {
  expect_matches_reference<proto::ChatRequest>(kToolsThenKwargs,
                                               ExpectedPath::FAST);
}

TEST(ChatRequestDecoderTest, AssistantToolCallsRoundTrip) {
  expect_matches_reference<proto::ChatRequest>(kAssistantToolCalls,
                                               ExpectedPath::FAST);
}

TEST(ChatRequestDecoderTest, MultimodalRequestMatchesReference) {
  expect_matches_reference<proto::MMChatRequest>(kMultimodal,
                                                 ExpectedPath::FAST);
}

// Shapes the scan refuses to plan; each still has to decode exactly as the
// reference parser does, because the fallback *is* the reference parser.
TEST(ChatRequestDecoderTest, NonCanonicalShapesFallBackAndStillMatch) {
  const std::vector<std::string> bodies = {
      // tools is not an array / holds a non-object.
      R"({"messages":[],"tools":"x"})",
      R"({"messages":[],"tools":[1]})",
      // function or parameters of an unexpected kind.
      R"({"messages":[],"tools":[{"type":"function","function":"f"}]})",
      R"({"messages":[],"tools":[{"function":{"name":"f","parameters":[]}}]})",
      R"({"messages":[],"tools":[{"function":{"name":"f","parameters":"s"}}]})",
      // chat_template_kwargs of an unexpected kind.
      R"({"messages":[],"chat_template_kwargs":[1]})",
      R"({"messages":[],"chat_template_kwargs":"s"})",
      // The two parsers disagree on which duplicate wins.
      R"({"messages":[],"tools":[],"tools":[]})",
      R"({"messages":[],"chat_template_kwargs":{},"chat_template_kwargs":{}})",
      // Root-level lowerCamelCase alias that only the reference parser maps.
      R"({"messages":[{"role":"user","content":"Hi"}],"maxTokens":5})",
      // Malformed.
      R"({"messages":[{"role":"user","content":"Hi"}])",
      "not json",
      "",
  };
  for (const std::string& body : bodies) {
    expect_matches_reference<proto::ChatRequest>(body, ExpectedPath::FALLBACK);
  }
}

// json2pb drops a mismatched optional scalar instead of rejecting it; the
// decoder must notice and let the reference parser coerce it as it always has.
TEST(ChatRequestDecoderTest, MismatchedScalarDefersToTheReferenceParser) {
  const std::string body =
      R"({"messages":[{"role":"user","content":"Hi"}],"max_tokens":"128"})";
  expect_matches_reference<proto::ChatRequest>(body, ExpectedPath::FAST);
  proto::ChatRequest decoded;
  ASSERT_TRUE(decode_chat_request(body, &decoded).ok());
  EXPECT_EQ(decoded.max_tokens(), 128U);
}

TEST(ChatRequestDecoderTest, RejectedBodiesCarryTheReferenceParserMessage) {
  const std::vector<std::string> bodies = {
      // Rejected by both parsers: the message must be the reference one.
      R"({"messages":[{"role":"user","content":"Hi"}],"stream":"yes"})",
      R"({"messages":[{"role":"user","content":"Hi"}],"max_tokens":1.5})",
      R"({"messages":"not an array"})",
      R"({"messages":[{"role":"user","content":"Hi"}]} trailing)",
  };
  for (const std::string& body : bodies) {
    proto::ChatRequest reference;
    google::protobuf::util::JsonParseOptions options;
    options.ignore_unknown_fields = true;
    const auto reference_status =
        google::protobuf::util::JsonStringToMessage(body, &reference, options);
    ASSERT_FALSE(reference_status.ok()) << body;

    proto::ChatRequest decoded;
    const Status status = decode_chat_request(body, &decoded);
    ASSERT_FALSE(status.ok()) << body;
    EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(status.message(), reference_status.ToString());
  }
}

TEST(ChatRequestDecoderTest, DeeplyNestedStructDefersToTheReferenceParser) {
  // Nested past what the scan skips through, so the body never reaches the
  // fast path. Whatever the reference parser decides about the nesting, the
  // decoder must decide the same.
  constexpr int32_t kDepth = 80;
  std::string body = R"({"messages":[],"chat_template_kwargs":)";
  for (int32_t i = 0; i < kDepth; ++i) {
    body += R"({"a":)";
  }
  body += "1";
  body.append(kDepth, '}');
  body += "}";
  EXPECT_FALSE(locate_chat_struct_members(body).has_value());
  proto::ChatRequest reference;
  google::protobuf::util::JsonParseOptions options;
  options.ignore_unknown_fields = true;
  const auto reference_status =
      google::protobuf::util::JsonStringToMessage(body, &reference, options);
  proto::ChatRequest decoded;
  const Status status = decode_chat_request(body, &decoded);
  EXPECT_EQ(status.ok(), reference_status.ok());
}

}  // namespace xllm

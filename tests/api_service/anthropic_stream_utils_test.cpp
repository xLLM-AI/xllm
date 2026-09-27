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

#include "api_service/anthropic_stream_utils.h"

#include <gtest/gtest.h>

#include <optional>
#include <string>

namespace xllm {
namespace {

TEST(AnthropicStreamUtilsTest, MapsToolFinishReasonsToToolUse) {
  EXPECT_EQ(api_service::convert_finish_reason_to_anthropic("tool_calls"),
            "tool_use");
  EXPECT_EQ(api_service::convert_finish_reason_to_anthropic("function_call"),
            "tool_use");
}

TEST(AnthropicStreamUtilsTest, MapsTextFinishReasons) {
  EXPECT_EQ(api_service::convert_finish_reason_to_anthropic("stop"),
            "end_turn");
  EXPECT_EQ(api_service::convert_finish_reason_to_anthropic("length"),
            "max_tokens");
  EXPECT_EQ(api_service::convert_finish_reason_to_anthropic("unknown"),
            "end_turn");
}

TEST(AnthropicStreamUtilsTest, ToolCallOverridesStreamStopReason) {
  EXPECT_EQ(api_service::get_stream_stop_reason(true, true, "stop"),
            "tool_use");
  EXPECT_EQ(api_service::get_stream_stop_reason(true, true, "length"),
            "tool_use");
  EXPECT_EQ(api_service::get_stream_stop_reason(true, true, ""), "tool_use");
}

TEST(AnthropicStreamUtilsTest, NamedToolUsesUnderlyingFinishReason) {
  EXPECT_EQ(api_service::get_stream_stop_reason(true, true, "stop", true),
            "end_turn");
  EXPECT_EQ(api_service::get_stream_stop_reason(true, true, "length", true),
            "max_tokens");
}

TEST(AnthropicStreamUtilsTest, TextStreamKeepsMappedStopReason) {
  EXPECT_EQ(api_service::get_stream_stop_reason(true, false, "stop"),
            "end_turn");
  EXPECT_EQ(api_service::get_stream_stop_reason(true, false, "length"),
            "max_tokens");
}

TEST(AnthropicStreamUtilsTest, CancelledStreamUsesValidAnthropicReason) {
  EXPECT_EQ(api_service::get_stream_stop_reason(false, true, "length"),
            "end_turn");
  EXPECT_EQ(api_service::get_stream_stop_reason(false, false, "stop"),
            "end_turn");
}

TEST(AnthropicStreamUtilsTest, EmptyToolArgsDoNotCreateInputDelta) {
  std::optional<proto::AnthropicStreamEvent> event =
      api_service::make_input_json_delta_event(2, "");

  EXPECT_FALSE(event.has_value());
}

TEST(AnthropicStreamUtilsTest, NonEmptyToolArgsCreateInputDelta) {
  std::optional<proto::AnthropicStreamEvent> event =
      api_service::make_input_json_delta_event(2, "{\"city\":\"Paris\"}");

  ASSERT_TRUE(event.has_value());
  EXPECT_EQ(event->type(), "content_block_delta");
  ASSERT_TRUE(event->has_index());
  EXPECT_EQ(event->index(), 2);
  ASSERT_TRUE(event->has_delta());
  EXPECT_EQ(event->delta().type(), "input_json_delta");
  EXPECT_EQ(event->delta().partial_json(), "{\"city\":\"Paris\"}");
}

TEST(AnthropicStreamUtilsTest, PreservesThinkingBeforeToolsWithoutEmptyText) {
  proto::ChatResponse chat;
  chat.set_id("msg_1");
  chat.set_model("test");
  auto* choice = chat.add_choices();
  choice->set_finish_reason("tool_calls");
  choice->mutable_message()->set_reasoning_content("reason");
  auto* tool = choice->mutable_message()->add_tool_calls();
  tool->set_id("call_1");
  tool->mutable_function()->set_name("search");
  tool->mutable_function()->set_arguments(R"({"query":"x"})");
  chat.mutable_usage()->set_prompt_tokens(100);
  chat.mutable_usage()->set_completion_tokens(8);
  chat.mutable_usage()->mutable_prompt_tokens_details()->set_cached_tokens(60);
  proto::AnthropicMessagesResponse response;
  ASSERT_FALSE(
      api_service::convert_anthropic_response(chat, "signature", response)
          .has_value());
  ASSERT_EQ(response.content_size(), 2);
  EXPECT_EQ(response.content(0).type(), "thinking");
  EXPECT_EQ(response.content(0).thinking(), "reason");
  EXPECT_EQ(response.content(0).signature(), "signature");
  EXPECT_EQ(response.content(1).type(), "tool_use");
  EXPECT_EQ(response.content(1).input().fields().at("query").string_value(),
            "x");
  EXPECT_EQ(response.stop_reason(), "tool_use");
  EXPECT_EQ(response.usage().input_tokens(), 100);
  EXPECT_FALSE(response.usage().has_cache_read_input_tokens());
  EXPECT_EQ(response.usage().output_tokens(), 8);
}

TEST(AnthropicStreamUtilsTest, EmptyCompletionHasEmptyContentArray) {
  proto::ChatResponse chat;
  chat.add_choices()->set_finish_reason("stop");
  proto::AnthropicMessagesResponse response;
  ASSERT_FALSE(
      api_service::convert_anthropic_response(chat, "signature", response)
          .has_value());
  EXPECT_EQ(response.content_size(), 0);
}

TEST(AnthropicStreamUtilsTest, ForcedToolResponseOmitsTextButKeepsReasoning) {
  proto::ChatResponse chat;
  auto* choice = chat.add_choices();
  choice->set_finish_reason("stop");
  auto* message = choice->mutable_message();
  message->set_content("I will call the weather tool.");
  message->set_reasoning_content("Need the current weather.");
  auto* tool = message->add_tool_calls();
  tool->set_id("call_weather");
  tool->mutable_function()->set_name("get_weather");
  tool->mutable_function()->set_arguments(R"({"city":"Paris"})");
  proto::AnthropicMessagesResponse response;
  ASSERT_FALSE(api_service::convert_anthropic_response(
                   chat, "signature", response, /*include_text=*/false)
                   .has_value());
  ASSERT_EQ(response.content_size(), 2);
  EXPECT_EQ(response.content(0).type(), "thinking");
  EXPECT_EQ(response.content(0).thinking(), "Need the current weather.");
  EXPECT_EQ(response.content(1).type(), "tool_use");
  EXPECT_EQ(response.content(1).name(), "get_weather");
  EXPECT_EQ(response.stop_reason(), "end_turn");
}

TEST(AnthropicStreamUtilsTest, ForcedToolWithoutCallHasEmptyContent) {
  proto::ChatResponse chat;
  auto* choice = chat.add_choices();
  choice->set_finish_reason("length");
  choice->mutable_message()->set_content("No tool call was produced.");
  proto::AnthropicMessagesResponse response;
  ASSERT_FALSE(api_service::convert_anthropic_response(
                   chat, "signature", response, /*include_text=*/false)
                   .has_value());
  EXPECT_EQ(response.content_size(), 0);
  EXPECT_EQ(response.stop_reason(), "max_tokens");
}

TEST(AnthropicStreamUtilsTest, InvalidToolJsonIsAnError) {
  proto::ChatResponse chat;
  chat.add_choices()
      ->mutable_message()
      ->add_tool_calls()
      ->mutable_function()
      ->set_arguments("{");
  proto::AnthropicMessagesResponse response;
  EXPECT_TRUE(
      api_service::convert_anthropic_response(chat, "signature", response)
          .has_value());
}

TEST(AnthropicStreamUtilsTest, StopMetadataDoesNotChangeVllm023EndTurn) {
  proto::ChatResponse chat;
  chat.add_choices()->set_finish_reason("stop");
  chat.mutable_choices(0)->mutable_stop_reason()->set_stop_string("END");
  proto::AnthropicMessagesResponse response;
  ASSERT_FALSE(
      api_service::convert_anthropic_response(chat, "signature", response)
          .has_value());
  EXPECT_EQ(response.stop_reason(), "end_turn");
  EXPECT_FALSE(response.has_stop_sequence());
  chat.mutable_choices(0)->set_finish_reason("length");
  ASSERT_FALSE(
      api_service::convert_anthropic_response(chat, "signature", response)
          .has_value());
  EXPECT_EQ(response.stop_reason(), "max_tokens");
  EXPECT_FALSE(response.has_stop_sequence());
}

}  // namespace
}  // namespace xllm

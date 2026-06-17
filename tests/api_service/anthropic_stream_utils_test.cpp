/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

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

TEST(AnthropicStreamUtilsTest, TextStreamKeepsMappedStopReason) {
  EXPECT_EQ(api_service::get_stream_stop_reason(true, false, "stop"),
            "end_turn");
  EXPECT_EQ(api_service::get_stream_stop_reason(true, false, "length"),
            "max_tokens");
}

TEST(AnthropicStreamUtilsTest, CancelledStreamUsesStop) {
  EXPECT_EQ(api_service::get_stream_stop_reason(false, true, "length"), "stop");
  EXPECT_EQ(api_service::get_stream_stop_reason(false, false, "stop"), "stop");
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

}  // namespace
}  // namespace xllm

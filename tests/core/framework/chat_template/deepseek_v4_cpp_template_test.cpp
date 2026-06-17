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

#include "framework/chat_template/deepseek_v4_cpp_template.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {

DeepseekV4CppTemplate make_encoder() {
  TokenizerArgs args;
  args.bos_token("<｜begin▁of▁sentence｜>");
  return DeepseekV4CppTemplate(args);
}

Message make_assistant_tool_call(const std::string& content,
                                 const std::string& call_id,
                                 const std::string& function_name,
                                 const std::string& arguments) {
  Message assistant_msg("assistant", content);
  Message::ToolCall tc;
  tc.id = call_id;
  tc.type = "function";
  tc.function.name = function_name;
  tc.function.arguments = arguments;
  assistant_msg.tool_calls = Message::ToolCallVec{tc};
  return assistant_msg;
}

Message make_multi_tool_call() {
  Message assistant_msg("assistant", "");
  Message::ToolCall first;
  first.id = "call_first";
  first.type = "function";
  first.function.name = "first_tool";
  first.function.arguments = R"({"value":1})";

  Message::ToolCall second;
  second.id = "call_second";
  second.type = "function";
  second.function.name = "second_tool";
  second.function.arguments = R"({"value":2})";

  assistant_msg.tool_calls = Message::ToolCallVec{first, second};
  return assistant_msg;
}

TEST(DeepseekV4CppTemplate, BasicChatModeUserMessage) {
  auto encoder = make_encoder();

  ChatMessages messages;
  messages.emplace_back("system", "You are a helpful assistant.");
  messages.emplace_back("user", "Hello");

  nlohmann::ordered_json kwargs = nlohmann::json::object();
  auto prompt = encoder.apply(messages, /*json_tools=*/{}, kwargs);
  ASSERT_TRUE(prompt.has_value());

  EXPECT_EQ(*prompt,
            "<｜begin▁of▁sentence｜>You are a helpful assistant."
            "<｜User｜>Hello<｜Assistant｜></think>");
}

TEST(DeepseekV4CppTemplate, ThinkingModeAddsThinkAfterLastUser) {
  auto encoder = make_encoder();

  ChatMessages messages;
  messages.emplace_back("user", "Hello");

  nlohmann::ordered_json kwargs = nlohmann::json::object();
  kwargs["thinking"] = true;
  auto prompt = encoder.apply(messages, /*json_tools=*/{}, kwargs);
  ASSERT_TRUE(prompt.has_value());

  EXPECT_EQ(*prompt,
            "<｜begin▁of▁sentence｜><｜User｜>Hello"
            "<｜Assistant｜><think>");
}

TEST(DeepseekV4CppTemplate, UsesToolCallsBlockName) {
  auto encoder = make_encoder();

  ChatMessages messages;
  messages.emplace_back("user", "weather?");
  Message assistant_msg = make_assistant_tool_call(
      "", "call_001", "get_weather", R"({"location":"Beijing"})");
  assistant_msg.reasoning_content = "Need weather data.";
  messages.push_back(assistant_msg);

  nlohmann::ordered_json kwargs = nlohmann::json::object();
  kwargs["thinking"] = true;
  auto prompt = encoder.apply(messages, /*json_tools=*/{}, kwargs);
  ASSERT_TRUE(prompt.has_value());

  EXPECT_NE(prompt->find("<｜DSML｜tool_calls>"), std::string::npos);
  EXPECT_NE(prompt->find("</｜DSML｜tool_calls>"), std::string::npos);
  EXPECT_EQ(prompt->find("<｜DSML｜function_calls>"), std::string::npos);
  EXPECT_NE(prompt->find("Need weather data.</think>"), std::string::npos);
}

TEST(DeepseekV4CppTemplate, ToolResultIsMergedAsUserToolResult) {
  auto encoder = make_encoder();

  ChatMessages messages;
  messages.emplace_back("user", "weather?");
  messages.push_back(make_assistant_tool_call(
      "", "call_001", "get_weather", R"({"location":"Beijing"})"));

  Message tool_msg("tool", R"({"temperature":22})");
  tool_msg.tool_call_id = "call_001";
  messages.push_back(tool_msg);

  nlohmann::ordered_json kwargs = nlohmann::json::object();
  auto prompt = encoder.apply(messages, /*json_tools=*/{}, kwargs);
  ASSERT_TRUE(prompt.has_value());

  EXPECT_NE(prompt->find("<｜User｜><tool_result>{\"temperature\":22}"
                         "</tool_result><｜Assistant｜></think>"),
            std::string::npos);
  EXPECT_EQ(prompt->find("<function_results>"), std::string::npos);
  EXPECT_EQ(prompt->find("<result>"), std::string::npos);
}

TEST(DeepseekV4CppTemplate, ToolsInjectionUsesV4SchemaText) {
  auto encoder = make_encoder();

  ChatMessages messages;
  messages.emplace_back("user", "weather in beijing");

  std::vector<JsonTool> tools;
  JsonTool tool;
  tool.type = "function";
  tool.function.name = "get_weather";
  tool.function.description = "query weather";
  tool.function.parameters = nlohmann::json{
      {"type", "object"}, {"properties", {{"city", {{"type", "string"}}}}}};
  tools.push_back(tool);

  nlohmann::ordered_json kwargs = nlohmann::json::object();
  auto prompt = encoder.apply(messages, tools, kwargs);
  ASSERT_TRUE(prompt.has_value());

  EXPECT_NE(prompt->find("You can invoke tools by writing a "
                         "\"<｜DSML｜tool_calls>\" block"),
            std::string::npos);
  EXPECT_NE(prompt->find("### Available Tool Schemas"), std::string::npos);
  EXPECT_NE(prompt->find("get_weather"), std::string::npos);
  EXPECT_EQ(prompt->find("<functions>"), std::string::npos);
}

TEST(DeepseekV4CppTemplate, ToolsKeepHistoricalReasoning) {
  auto encoder = make_encoder();

  ChatMessages messages;
  messages.emplace_back("user", "hello");
  Message old_assistant("assistant", "hi");
  old_assistant.reasoning_content = "historical reasoning";
  messages.push_back(old_assistant);
  messages.emplace_back("user", "weather?");

  std::vector<JsonTool> tools;
  JsonTool tool;
  tool.type = "function";
  tool.function.name = "get_weather";
  tool.function.parameters = nlohmann::json{{"type", "object"}};
  tools.push_back(tool);

  nlohmann::ordered_json kwargs = nlohmann::json::object();
  kwargs["thinking"] = true;
  auto prompt = encoder.apply(messages, tools, kwargs);
  ASSERT_TRUE(prompt.has_value());

  EXPECT_NE(prompt->find("historical reasoning</think>"), std::string::npos);
}

TEST(DeepseekV4CppTemplate, LatestReminderUsesDedicatedToken) {
  auto encoder = make_encoder();

  ChatMessages messages;
  messages.emplace_back("system", "该助手为DeepSeek。");
  messages.emplace_back("latest_reminder", "2026-02-21,星期六,广州,App,中文");
  messages.emplace_back("user", "你好");

  nlohmann::ordered_json kwargs = nlohmann::json::object();
  auto prompt = encoder.apply(messages, /*json_tools=*/{}, kwargs);
  ASSERT_TRUE(prompt.has_value());

  EXPECT_EQ(*prompt,
            "<｜begin▁of▁sentence｜>该助手为DeepSeek。"
            "<｜latest_reminder｜>2026-02-21,星期六,广州,App,中文"
            "<｜User｜>你好<｜Assistant｜></think>");
}

TEST(DeepseekV4CppTemplate, AdjacentUserMessagesAreMergedAsContentBlocks) {
  auto encoder = make_encoder();

  ChatMessages messages;
  messages.emplace_back("user", "first");
  messages.emplace_back("user", "second");

  nlohmann::ordered_json kwargs = nlohmann::json::object();
  auto prompt = encoder.apply(messages, /*json_tools=*/{}, kwargs);
  ASSERT_TRUE(prompt.has_value());

  EXPECT_EQ(*prompt,
            "<｜begin▁of▁sentence｜><｜User｜>first\n\nsecond"
            "<｜Assistant｜></think>");
}

TEST(DeepseekV4CppTemplate, ToolResultsAreSortedByToolCallId) {
  auto encoder = make_encoder();

  ChatMessages messages;
  messages.emplace_back("user", "run tools");
  messages.push_back(make_multi_tool_call());

  Message second_result("tool", "second result");
  second_result.tool_call_id = "call_second";
  messages.push_back(second_result);

  Message first_result("tool", "first result");
  first_result.tool_call_id = "call_first";
  messages.push_back(first_result);

  nlohmann::ordered_json kwargs = nlohmann::json::object();
  auto prompt = encoder.apply(messages, /*json_tools=*/{}, kwargs);
  ASSERT_TRUE(prompt.has_value());

  size_t first_pos = prompt->find("<tool_result>first result</tool_result>");
  size_t second_pos = prompt->find("<tool_result>second result</tool_result>");
  ASSERT_NE(first_pos, std::string::npos);
  ASSERT_NE(second_pos, std::string::npos);
  EXPECT_LT(first_pos, second_pos);
}

TEST(DeepseekV4CppTemplate, ReasoningEffortMaxPrefixesThinkingPrompt) {
  auto encoder = make_encoder();

  ChatMessages messages;
  messages.emplace_back("user", "hard problem");

  nlohmann::ordered_json kwargs = nlohmann::json::object();
  kwargs["thinking"] = true;
  kwargs["reasoning_effort"] = "max";
  auto prompt = encoder.apply(messages, /*json_tools=*/{}, kwargs);
  ASSERT_TRUE(prompt.has_value());

  EXPECT_NE(prompt->find("Reasoning Effort: Absolute maximum"),
            std::string::npos);
  EXPECT_LT(prompt->find("Reasoning Effort: Absolute maximum"),
            prompt->find("<｜User｜>hard problem"));
}

}  // namespace
}  // namespace xllm

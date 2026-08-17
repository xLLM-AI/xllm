/* Copyright 2025-2026 The xLLM Authors.

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

#include "framework/chat_template/deepseek_v32_cpp_template.h"

#include <gtest/gtest.h>

namespace xllm {

TEST(DeepseekV32CppTemplate, BasicUserMessage) {
  TokenizerArgs args;
  args.bos_token("<｜begin▁of▁sentence｜>");
  DeepseekV32CppTemplate encoder(args);

  ChatMessages messages;
  messages.emplace_back("user", "hello");

  nlohmann::ordered_json kwargs = nlohmann::json::object();
  auto prompt = encoder.apply(messages, /*json_tools=*/{}, kwargs);
  ASSERT_TRUE(prompt.has_value());

  EXPECT_NE(prompt->find("<｜begin▁of▁sentence｜>"), std::string::npos);
  EXPECT_NE(prompt->find("<｜User｜>hello<｜Assistant｜>"), std::string::npos);
}

TEST(DeepseekV32CppTemplate, DefaultThinkingModeIsChat) {
  TokenizerArgs args;
  args.bos_token("<｜begin▁of▁sentence｜>");
  DeepseekV32CppTemplate encoder(args);

  ChatMessages messages;
  messages.emplace_back("user", "hello");

  nlohmann::ordered_json kwargs = nlohmann::json::object();
  auto prompt = encoder.apply(messages, /*json_tools=*/{}, kwargs);
  ASSERT_TRUE(prompt.has_value());

  EXPECT_NE(prompt->find("</think>"), std::string::npos);
  EXPECT_EQ(prompt->find("<think>"), std::string::npos);
}

TEST(DeepseekV32CppTemplate, ThinkingModeEnabledByKwargs) {
  TokenizerArgs args;
  args.bos_token("<｜begin▁of▁sentence｜>");
  DeepseekV32CppTemplate encoder(args);

  ChatMessages messages;
  messages.emplace_back("user", "hello");

  nlohmann::ordered_json kwargs = nlohmann::json::object();
  kwargs["thinking"] = true;

  auto prompt = encoder.apply(messages, /*json_tools=*/{}, kwargs);
  ASSERT_TRUE(prompt.has_value());

  EXPECT_NE(prompt->find("<think>"), std::string::npos);
}

TEST(DeepseekV32CppTemplate, ReportsRenderedGenerationMode) {
  TokenizerArgs args;
  args.bos_token("<｜begin▁of▁sentence｜>");
  DeepseekV32CppTemplate encoder(args);
  ChatMessages messages;
  messages.emplace_back("user", "hello");

  auto chat = encoder.apply_with_generation_mode(
      messages, /*json_tools=*/{}, nlohmann::ordered_json::object());
  ASSERT_TRUE(chat.has_value());
  EXPECT_EQ(chat->generation_mode, ChatTemplateGenerationMode::CHAT);

  auto reasoning = encoder.apply_with_generation_mode(
      messages, /*json_tools=*/{}, nlohmann::ordered_json{{"thinking", true}});
  ASSERT_TRUE(reasoning.has_value());
  EXPECT_EQ(reasoning->generation_mode, ChatTemplateGenerationMode::REASONING);
}

TEST(DeepseekV32CppTemplate, ToolsInjectionFormat) {
  TokenizerArgs args;
  args.bos_token("<｜begin▁of▁sentence｜>");
  DeepseekV32CppTemplate encoder(args);

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

  // vLLM DSML format
  EXPECT_NE(prompt->find("## Tools"), std::string::npos);
  EXPECT_NE(prompt->find("get_weather"), std::string::npos);
  EXPECT_NE(prompt->find("<functions>"), std::string::npos);
  EXPECT_NE(prompt->find("</functions>"), std::string::npos);
  // User message after tools
  EXPECT_NE(prompt->find("<｜User｜>weather in beijing"
                         "<｜Assistant｜>"),
            std::string::npos);
}

TEST(DeepseekV32CppTemplate, ToolsInjectedAsNewSystemMessage) {
  TokenizerArgs args;
  args.bos_token("<｜begin▁of▁sentence｜>");
  DeepseekV32CppTemplate encoder(args);

  ChatMessages messages;
  messages.emplace_back("system", "You are helpful.");
  messages.emplace_back("user", "hi");

  std::vector<JsonTool> tools;
  JsonTool tool;
  tool.type = "function";
  tool.function.name = "search";
  tool.function.description = "search the web";
  tool.function.parameters = nlohmann::json{{"type", "object"}};
  tools.push_back(tool);

  nlohmann::ordered_json kwargs = nlohmann::json::object();
  auto prompt = encoder.apply(messages, tools, kwargs);
  ASSERT_TRUE(prompt.has_value());

  // Tools section before system content
  size_t tools_pos = prompt->find("## Tools");
  size_t content_pos = prompt->find("You are helpful.");
  ASSERT_NE(tools_pos, std::string::npos);
  ASSERT_NE(content_pos, std::string::npos);
  EXPECT_LT(tools_pos, content_pos);
}

TEST(DeepseekV32CppTemplate, DropThinkingOnlyWhenLastMessageIsUser) {
  TokenizerArgs args;
  args.bos_token("<｜begin▁of▁sentence｜>");
  DeepseekV32CppTemplate encoder(args);

  // user -> assistant(with reasoning) -> tool
  ChatMessages messages;
  messages.emplace_back("user", "weather?");

  Message assistant_msg("assistant", "calling tool");
  assistant_msg.reasoning_content = "thinking about it";
  Message::ToolCall tc;
  tc.id = "1";
  tc.type = "function";
  tc.function.name = "get_weather";
  tc.function.arguments = R"({"city":"beijing"})";
  assistant_msg.tool_calls = Message::ToolCallVec{tc};
  messages.push_back(assistant_msg);

  messages.emplace_back("tool", "sunny");

  nlohmann::ordered_json kwargs = nlohmann::json::object();
  kwargs["thinking"] = true;

  auto prompt = encoder.apply(messages, /*json_tools=*/{}, kwargs);
  ASSERT_TRUE(prompt.has_value());

  // Last message is "tool", NOT "user",
  // so thinking should NOT be dropped.
  EXPECT_NE(prompt->find("thinking about it"), std::string::npos);
}

}  // namespace xllm

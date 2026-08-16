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

#include "jinja_chat_template.h"

#include <gtest/gtest.h>

namespace xllm {

class TestableJinjaChatTemplate : public JinjaChatTemplate {
 public:
  TestableJinjaChatTemplate(const TokenizerArgs& args)
      : JinjaChatTemplate(args) {}

  using JinjaChatTemplate::apply;
};

TEST(JinjaChatTemplate, OpenChatModel) {
  // clang-format off
  const std::string template_str =
      "<s>"
      "{% for message in messages %}"
        "{{ 'GPT4 Correct ' + message['role'] + ': ' + message['content'] + '<|end_of_turn|>'}}"
      "{% endfor %}"
      "{% if add_generation_prompt %}{{ 'GPT4 Correct Assistant:' }}{% endif %}";

  nlohmann::ordered_json messages = {
      {{"role", "system"}, {"content", "you are a helpful assistant."}},
      {{"role", "user"}, {"content", "hi"}},
      {{"role", "assistant"}, {"content", "what i can do for you?"}},
      {{"role", "user"}, {"content", "how are you?"}}};
  const std::string expected =
    "<s>"
    "GPT4 Correct system: you are a helpful assistant.<|end_of_turn|>"
    "GPT4 Correct user: hi<|end_of_turn|>"
    "GPT4 Correct assistant: what i can do for you?<|end_of_turn|>"
    "GPT4 Correct user: how are you?<|end_of_turn|>"
    "GPT4 Correct Assistant:";
  // clang-format on

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("<|end_of_turn|>");
  TestableJinjaChatTemplate template_(args);
  auto result = template_.apply(messages);
  ASSERT_TRUE(result.has_value());

  EXPECT_EQ(result.value(), expected);
}

TEST(JinjaChatTemplate, AppliesChatTemplateKwargs) {
  const std::string template_str =
      "{% if enable_thinking %}<think>{% endif %}"
      "{% for message in messages %}"
      "{{ message['role'] + ': ' + message['content'] }}"
      "{% endfor %}"
      "{% if not enable_thinking %}<no_think>{% endif %}";

  nlohmann::ordered_json messages = {
      {{"role", "user"}, {"content", "describe this image"}}};
  nlohmann::ordered_json chat_template_kwargs = {{"enable_thinking", false}};

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  TestableJinjaChatTemplate template_(args);
  const nlohmann::ordered_json tools = nlohmann::json::array();
  auto result = template_.apply(messages, tools, chat_template_kwargs);
  ASSERT_TRUE(result.has_value());

  EXPECT_EQ(result.value(), "user: describe this image<no_think>");
}

TEST(JinjaChatTemplate, ReportsRenderedGenerationMode) {
  const std::string template_str =
      "{% if enable_thinking %}<think>{% else %}</think>{% endif %}";
  ChatMessages messages;
  messages.emplace_back("user", "hello");
  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  TestableJinjaChatTemplate template_(args);

  const std::vector<JsonTool> tools;
  auto reasoning = template_.apply_with_generation_mode(
      messages, tools, nlohmann::ordered_json{{"enable_thinking", true}});
  ASSERT_TRUE(reasoning.has_value());
  EXPECT_EQ(reasoning->generation_mode, ChatTemplateGenerationMode::REASONING);

  auto chat = template_.apply_with_generation_mode(
      messages, tools, nlohmann::ordered_json{{"enable_thinking", false}});
  ASSERT_TRUE(chat.has_value());
  EXPECT_EQ(chat->generation_mode, ChatTemplateGenerationMode::CHAT);

  args.chat_template("assistant:");
  TestableJinjaChatTemplate unknown_template(args);
  auto unknown = unknown_template.apply_with_generation_mode(
      messages, tools, nlohmann::ordered_json::object());
  ASSERT_TRUE(unknown.has_value());
  EXPECT_EQ(unknown->generation_mode, ChatTemplateGenerationMode::UNKNOWN);
}

TEST(ChatTemplate, ClassifiesTerminalGenerationMarkers) {
  EXPECT_EQ(ChatTemplate::generation_mode_from_prompt("prefix <think>\n"),
            ChatTemplateGenerationMode::REASONING);
  EXPECT_EQ(ChatTemplate::generation_mode_from_prompt("prefix </think>\t"),
            ChatTemplateGenerationMode::CHAT);
  EXPECT_EQ(ChatTemplate::generation_mode_from_prompt("prefix"),
            ChatTemplateGenerationMode::UNKNOWN);
}

TEST(JinjaChatTemplate, SupportsUndefinedTests) {
  // Qwen3.8 uses both forms for optional chat-template arguments. Quoted text
  // must remain unchanged while executable Jinja expressions are normalized.
  const std::string template_str =
      "plain text: value is undefined|"
      "{{ 'quoted value is undefined' }}|"
      "{% if enable_thinking is undefined %}default"
      "{% else %}configured{% endif %}|"
      "{% if enable_thinking is not undefined %}present"
      "{% else %}missing{% endif %}";

  nlohmann::ordered_json messages = {{{"role", "user"}, {"content", "hello"}}};
  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  TestableJinjaChatTemplate template_(args);

  auto default_result = template_.apply(messages);
  ASSERT_TRUE(default_result.has_value());
  EXPECT_EQ(default_result.value(),
            "plain text: value is undefined|quoted value is undefined|default|"
            "missing");

  const nlohmann::ordered_json tools = nlohmann::json::array();
  const nlohmann::ordered_json kwargs = {{"enable_thinking", false}};
  auto configured_result = template_.apply(messages, tools, kwargs);
  ASSERT_TRUE(configured_result.has_value());
  EXPECT_EQ(configured_result.value(),
            "plain text: value is undefined|quoted value is undefined|"
            "configured|present");
}

}  // namespace xllm

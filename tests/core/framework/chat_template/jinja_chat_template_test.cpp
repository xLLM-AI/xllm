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

#include "jinja_chat_template.h"

#include <gtest/gtest.h>

#include <ctime>
#include <minja/chat-template.hpp>
#include <nlohmann/json.hpp>
#include <string>

namespace xllm {

class TestableJinjaChatTemplate : public JinjaChatTemplate {
 public:
  TestableJinjaChatTemplate(const TokenizerArgs& args)
      : JinjaChatTemplate(args) {}

  using JinjaChatTemplate::apply;
  using JinjaChatTemplate::needs_polyfills;
};

// Qwen2.5's published template: supports system messages, tools, tool calls
// and tool responses natively, and renders tool-call arguments with `tojson`.
constexpr char kQwen25ChatTemplate[] = R"JINJA({%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\n<tool_call>\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
)JINJA";

// What minja::chat_template::apply itself produces for these inputs; the
// native render must match it byte for byte.
std::string reference_render(const std::string& template_str,
                             const std::string& eos_token,
                             const nlohmann::ordered_json& messages,
                             const nlohmann::ordered_json& tools,
                             const nlohmann::ordered_json& kwargs) {
  minja::chat_template reference(template_str, "", eos_token);
  minja::chat_template_inputs inputs;
  inputs.messages = messages;
  inputs.tools = tools;
  inputs.add_generation_prompt = true;
  inputs.extra_context = kwargs;
  return reference.apply(inputs, minja::chat_template_options());
}

nlohmann::ordered_json make_weather_tools() {
  return nlohmann::ordered_json::array(
      {{{"type", "function"},
        {"function",
         {{"name", "get_weather"},
          {"description", "Get the weather for a city."},
          {"parameters",
           {{"type", "object"},
            {"properties", {{"city", {{"type", "string"}}}}},
            {"required", {"city"}}}}}}}});
}

TEST(JinjaChatTemplate, NativeRenderMatchesMinja) {
  const nlohmann::ordered_json messages = {
      {{"role", "system"}, {"content", "You are a helpful assistant."}},
      {{"role", "user"}, {"content", "hi"}},
      {{"role", "assistant"}, {"content", "Hello! How can I help?"}},
      {{"role", "user"}, {"content", "What is the weather in Paris?"}}};
  const nlohmann::ordered_json kwargs = nlohmann::ordered_json::object();

  TokenizerArgs args;
  args.chat_template(kQwen25ChatTemplate);
  args.bos_token("");
  args.eos_token("<|im_end|>");
  TestableJinjaChatTemplate template_(args);

  // A plain conversation and one carrying tool definitions both render
  // natively on this template.
  for (const nlohmann::ordered_json& tools :
       {nlohmann::ordered_json::array(), make_weather_tools()}) {
    EXPECT_FALSE(template_.needs_polyfills(messages, tools));
    auto result = template_.apply(messages, tools, kwargs);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(),
              reference_render(
                  kQwen25ChatTemplate, "<|im_end|>", messages, tools, kwargs));
  }
}

TEST(JinjaChatTemplate, PolyfilledInputsMatchMinja) {
  // Qwen2.5 renders tool-call arguments with `tojson`, so minja rewrites
  // string arguments into objects first; whichever path that takes, the
  // output must equal minja's own.
  const nlohmann::ordered_json messages = {
      {{"role", "system"}, {"content", "You are a helpful assistant."}},
      {{"role", "user"}, {"content", "What is the weather in Paris?"}},
      {{"role", "assistant"},
       {"content", ""},
       {"tool_calls",
        {{{"id", "call_1"},
          {"type", "function"},
          {"function",
           {{"name", "get_weather"}, {"arguments", R"({"city":"Paris"})"}}}}}}},
      {{"role", "tool"}, {"tool_call_id", "call_1"}, {"content", "18C"}}};
  const nlohmann::ordered_json tools = make_weather_tools();
  const nlohmann::ordered_json kwargs = nlohmann::ordered_json::object();

  TokenizerArgs args;
  args.chat_template(kQwen25ChatTemplate);
  args.bos_token("");
  args.eos_token("<|im_end|>");
  TestableJinjaChatTemplate template_(args);

  auto result = template_.apply(messages, tools, kwargs);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(),
            reference_render(
                kQwen25ChatTemplate, "<|im_end|>", messages, tools, kwargs));
}

TEST(JinjaChatTemplate, NativeRenderDefinesMinjaGlobals) {
  // bos_token / eos_token, strftime_now and chat_template_kwargs are globals
  // minja::chat_template::apply installs; the native render must too.
  const std::string template_str =
      "{{ bos_token }}"
      "{% for message in messages %}"
      "{{ message['role'] + ': ' + message['content'] }}{{ eos_token }}"
      "{% endfor %}"
      "{{ strftime_now('%Y') }}|{{ mode }}";
  const nlohmann::ordered_json messages = {
      {{"role", "system"}, {"content", "be brief"}},
      {{"role", "user"}, {"content", "hello"}}};
  const nlohmann::ordered_json tools = nlohmann::ordered_json::array();
  const nlohmann::ordered_json kwargs = {{"mode", "fast"}};

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("<s>");
  args.eos_token("</s>");
  TestableJinjaChatTemplate template_(args);
  ASSERT_FALSE(template_.needs_polyfills(messages, tools));

  auto result = template_.apply(messages, tools, kwargs);
  ASSERT_TRUE(result.has_value());

  const std::time_t now = std::time(nullptr);
  char year[8];
  std::strftime(year, sizeof(year), "%Y", std::localtime(&now));
  EXPECT_EQ(
      result.value(),
      std::string("<s>system: be brief</s>user: hello</s>") + year + "|fast");
}

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

TEST(JinjaChatTemplate, PreservesGlmToolResponses) {
  const std::string template_str = R"JINJA(
{%- macro is_list_of_outputs(m) -%}
  {%- if m.content and m.content.0.output is defined -%}1{%- endif -%}
{%- endmacro -%}
{%- for m in messages -%}
  {%- if m.role == 'tool' -%}
    {{- '<|observation|><tool_response>' -}}
    {%- if is_list_of_outputs(m) -%}
      {%- for item in m.content -%}{{- item.output -}}{%- endfor -%}
    {%- else -%}{{- m.content -}}{%- endif -%}
    {{- '</tool_response>' -}}
  {%- elif m.tool_calls -%}
    {%- for call in m.tool_calls -%}
      {{- call.function.name -}}{{- call.function.arguments | tojson -}}
    {%- endfor -%}
  {%- else -%}{{- m.content -}}{%- endif -%}
{%- endfor -%}
)JINJA";
  TokenizerArgs args;
  args.chat_template(template_str);
  TestableJinjaChatTemplate template_(args);
  const nlohmann::ordered_json tools = nlohmann::ordered_json::array();
  const nlohmann::ordered_json string_messages = {
      {{"role", "tool"}, {"content", "Paris is sunny."}}};
  EXPECT_FALSE(template_.needs_polyfills(string_messages, tools));
  auto string_result = template_.apply(string_messages);
  ASSERT_TRUE(string_result.has_value());
  EXPECT_EQ(*string_result,
            "<|observation|><tool_response>Paris is sunny.</tool_response>");

  const nlohmann::ordered_json list_messages = {
      {{"role", "tool"},
       {"content", {{{"output", "Paris "}}, {{"output", "is sunny."}}}}}};
  auto list_result = template_.apply(list_messages);
  ASSERT_TRUE(list_result.has_value());
  EXPECT_EQ(*list_result, *string_result);

  args.chat_template(
      "m.content.0.output is defined|"
      "{{ 'm.content.0.output is defined' }}");
  TestableJinjaChatTemplate literal_template(args);
  auto literal_result = literal_template.apply(string_messages);
  ASSERT_TRUE(literal_result.has_value());
  EXPECT_EQ(*literal_result,
            "m.content.0.output is defined|m.content.0.output is defined");
}

TEST(JinjaChatTemplate, PreservesDeferredToolMetadata) {
  TokenizerArgs args;
  args.chat_template(kQwen25ChatTemplate);
  TestableJinjaChatTemplate chat_template(args);
  const ChatMessages messages = {Message("user", "hello")};
  JsonTool tool;
  tool.function.name = "search";
  tool.function.parameters = {{"type", "object"}};
  tool.function.defer_loading = true;
  const std::vector<JsonTool> tools = {tool};
  const auto prompt =
      chat_template.apply(messages, tools, nlohmann::ordered_json::object());
  ASSERT_TRUE(prompt.has_value());
  EXPECT_NE(prompt->find("\"defer_loading\": true"), std::string::npos);
}

}  // namespace xllm

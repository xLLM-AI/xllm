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

// Micro-benchmark for the chat-template render on the request path: the
// ChatMessages -> JSON conversion plus the minja render of a real-world
// template, exactly the call LLMRequestFactory::create(messages, ...) makes.
//
// The other hop benchmarks stub the chat template out (llm_request_factory
// benchmark uses a FakeChatTemplate), so this is the only place the render
// itself is measured. The conversation shape mirrors
// api_service_benchmark's make_chat_json (system + alternating user/assistant
// turns of 256 chars) so the two hops can be compared per request.
//
//   * BM_ChatTemplate_JinjaRender          - n messages, no tools
//   * BM_ChatTemplate_JinjaRenderWithTools - a 4-turn tool-calling
//                                            conversation plus n tools, which
//                                            exercises the `tool | tojson`
//                                            branch of the template
//   * BM_ChatTemplate_MinjaApply           - the n-message conversation
//                                            through
//                                            minja::chat_template::apply
//                                            directly (the pre-native path)
//   * BM_ChatTemplate_MinjaBuiltins        - constructing minja's builtin
//                                            globals, which apply does per
//                                            render
//   * BM_ChatTemplate_MessagesToJson       - the ChatMessages -> ordered_json
//                                            conversion alone
//   * BM_ChatTemplate_MinjaRenderCachedBuiltins - the render alone on a
//                                            pre-built document and cached
//                                            builtins (render_native's core)
//   * BM_ChatTemplate_AllocatorControl     - eight small malloc/free pairs, the
//                                            allocator floor under all of it
//
// Build & run (example):
//   python setup.py test --test-name chat_template_benchmark
//   ./chat_template_benchmark --benchmark_min_time=0.2s

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <minja/chat-template.hpp>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "core/common/message.h"
#include "core/common/types.h"
#include "core/framework/chat_template/chat_template.h"
#include "core/framework/chat_template/jinja_chat_template.h"
#include "core/framework/tokenizer/tokenizer_args.h"

namespace xllm {
namespace {

// benchmark 1.8.x exposes only DoNotOptimize(Tp const&) -- deprecated -- and
// DoNotOptimize(Tp&); there is no rvalue overload, so a temporary or a const
// local resolves to the deprecated one. Sinking the value into a non-const
// parameter first selects the supported overload.
template <typename T>
inline BENCHMARK_ALWAYS_INLINE void do_not_optimize(T value) {
  benchmark::DoNotOptimize(value);
}

// Qwen2.5's published chat template: a system-prompt branch that serialises
// the tool list, per-role message rendering including tool calls and tool
// responses, and the generation prompt. A representative production template
// rather than a toy loop.
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

constexpr size_t kTurnChars = 256;

TokenizerArgs make_tokenizer_args() {
  TokenizerArgs args;
  args.chat_template(kQwen25ChatTemplate);
  args.bos_token("");
  args.eos_token("<|im_end|>");
  return args;
}

// System prompt plus alternating user / assistant turns, the same shape as
// api_service_benchmark's make_chat_json.
ChatMessages make_messages(size_t num_messages) {
  ChatMessages messages;
  messages.reserve(num_messages);
  messages.emplace_back("system", "You are a helpful assistant.");
  for (size_t i = 1; i < num_messages; ++i) {
    messages.emplace_back(i % 2 == 1 ? "user" : "assistant",
                          std::string(kTurnChars, 'y'));
  }
  return messages;
}

// A completed tool call round trip: the assistant turn carries tool_calls and
// the tool turn answers it, so both template branches run.
ChatMessages make_tool_messages() {
  ChatMessages messages;
  messages.reserve(4);
  messages.emplace_back("system", "You are a helpful assistant.");
  messages.emplace_back("user", "What is the weather in Paris?");
  messages.emplace_back("assistant", "");
  Message::ToolCall tool_call;
  tool_call.id = "call_1";
  tool_call.type = "function";
  tool_call.function.name = "get_weather_0";
  tool_call.function.arguments = R"({"city":"Paris"})";
  messages.back().tool_calls = Message::ToolCallVec{tool_call};
  messages.emplace_back("tool", "18C, cloudy");
  messages.back().tool_call_id = "call_1";
  return messages;
}

std::vector<JsonTool> make_tools(size_t num_tools) {
  std::vector<JsonTool> tools;
  tools.reserve(num_tools);
  for (size_t i = 0; i < num_tools; ++i) {
    const nlohmann::json parameters = {
        {"type", "object"},
        {"properties",
         {{"city",
           {{"type", "string"},
            {"description", "The city to look up, e.g. Paris"}}},
          {"unit", {{"type", "string"}, {"enum", {"celsius", "fahrenheit"}}}},
          {"days", {{"type", "integer"}, {"minimum", 1}, {"maximum", 14}}}}},
        {"required", {"city"}}};
    tools.emplace_back(
        "function",
        JsonFunction("get_weather_" + std::to_string(i),
                     "Get the current weather and forecast for a city.",
                     parameters));
  }
  return tools;
}

size_t content_bytes(const ChatMessages& messages) {
  size_t bytes = 0;
  for (const auto& message : messages) {
    if (const auto* text = std::get_if<std::string>(&message.content)) {
      bytes += text->size();
    }
  }
  return bytes;
}

void BM_ChatTemplate_JinjaRender(benchmark::State& state) {
  const JinjaChatTemplate chat_template(make_tokenizer_args());
  const ChatMessages messages =
      make_messages(static_cast<size_t>(state.range(0)));
  const std::vector<JsonTool> no_tools;
  const nlohmann::ordered_json kwargs = nlohmann::ordered_json::object();

  for (auto _ : state) {
    std::optional<ChatTemplateRenderResult> rendered =
        chat_template.apply_with_generation_mode(messages, no_tools, kwargs);
    if (!rendered.has_value()) {
      state.SkipWithError("chat template render failed");
      break;
    }
    do_not_optimize(rendered->prompt.data());
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(content_bytes(messages)));
}

void BM_ChatTemplate_JinjaRenderWithTools(benchmark::State& state) {
  const JinjaChatTemplate chat_template(make_tokenizer_args());
  const ChatMessages messages = make_tool_messages();
  const std::vector<JsonTool> tools =
      make_tools(static_cast<size_t>(state.range(0)));
  const nlohmann::ordered_json kwargs = nlohmann::ordered_json::object();

  for (auto _ : state) {
    std::optional<ChatTemplateRenderResult> rendered =
        chat_template.apply_with_generation_mode(messages, tools, kwargs);
    if (!rendered.has_value()) {
      state.SkipWithError("chat template render failed");
      break;
    }
    do_not_optimize(rendered->prompt.data());
  }
}

// The same conversation through minja::chat_template::apply directly -- the
// path JinjaChatTemplate used before rendering natively -- for the A/B.
void BM_ChatTemplate_MinjaApply(benchmark::State& state) {
  const minja::chat_template chat_template(
      kQwen25ChatTemplate, "", "<|im_end|>");
  const ChatMessages messages =
      make_messages(static_cast<size_t>(state.range(0)));
  nlohmann::ordered_json messages_json = nlohmann::ordered_json::array();
  for (const auto& message : messages) {
    messages_json.push_back(
        {{"role", message.role},
         {"content", std::get<std::string>(message.content)}});
  }

  for (auto _ : state) {
    minja::chat_template_inputs inputs;
    inputs.messages = messages_json;
    inputs.tools = nlohmann::ordered_json::array();
    inputs.add_generation_prompt = true;
    inputs.extra_context = nlohmann::ordered_json::object();
    std::string prompt =
        chat_template.apply(inputs, minja::chat_template_options());
    do_not_optimize(prompt.data());
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(content_bytes(messages)));
}

// The builtin globals minja::chat_template::apply constructs on every render
// and JinjaChatTemplate now builds once.
void BM_ChatTemplate_MinjaBuiltins(benchmark::State& state) {
  for (auto _ : state) {
    std::shared_ptr<minja::Context> builtins = minja::Context::builtins();
    do_not_optimize(builtins.get());
  }
}

// --------------------------------------------------------------------------
// Attribution of JinjaRender against MinjaApply
//
// JinjaRender = ChatMessages -> ordered_json conversion + needs_polyfills scan
//             + render on the cached builtins. The two below time the first
// and last term on their own, so the difference to MinjaApply can be placed.
// --------------------------------------------------------------------------

// The per-message ChatMessages -> ordered_json conversion, as
// JinjaChatTemplate::apply performs it for text-only messages.
void BM_ChatTemplate_MessagesToJson(benchmark::State& state) {
  const ChatMessages messages =
      make_messages(static_cast<size_t>(state.range(0)));

  for (auto _ : state) {
    nlohmann::ordered_json messages_json = nlohmann::json::array();
    for (const auto& message : messages) {
      nlohmann::ordered_json message_json;
      message_json["role"] = message.role;
      message_json["content"] = std::get<std::string>(message.content);
      messages_json.emplace_back(std::move(message_json));
    }
    do_not_optimize(messages_json.size());
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(content_bytes(messages)));
}

// The render alone on a pre-built messages document: parsed root plus cached
// builtins, i.e. what JinjaChatTemplate::render_native does after the
// conversion. Comparable one-to-one with MinjaApply.
void BM_ChatTemplate_MinjaRenderCachedBuiltins(benchmark::State& state) {
  const std::shared_ptr<minja::TemplateNode> root =
      minja::Parser::parse(kQwen25ChatTemplate,
                           {/*trim_blocks=*/true,
                            /*lstrip_blocks=*/true,
                            /*keep_trailing_newline=*/false});
  const std::shared_ptr<minja::Context> builtins = minja::Context::builtins();
  const ChatMessages messages =
      make_messages(static_cast<size_t>(state.range(0)));
  nlohmann::ordered_json messages_json = nlohmann::ordered_json::array();
  for (const auto& message : messages) {
    messages_json.push_back(
        {{"role", message.role},
         {"content", std::get<std::string>(message.content)}});
  }
  const nlohmann::ordered_json tools = nlohmann::ordered_json::array();

  for (auto _ : state) {
    minja::Value values = minja::Value::object();
    values.set("messages", minja::Value(messages_json));
    values.set("add_generation_prompt", minja::Value(true));
    auto context = minja::Context::make(std::move(values), builtins);
    context->set("bos_token", minja::Value(""));
    context->set("eos_token", minja::Value("<|im_end|>"));
    context->set("tools", minja::Value(tools));
    std::string prompt = root->render(context);
    do_not_optimize(prompt.data());
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(content_bytes(messages)));
}

// Control: the allocator every number above is built on. One iteration is
// the eight small allocations a message conversion costs, freed in reverse.
// A healthy malloc lands at a few hundred nanoseconds per iteration; anything
// in the microseconds says the allocator, not the template code, is the cost.
void BM_ChatTemplate_AllocatorControl(benchmark::State& state) {
  constexpr size_t kSizes[] = {32, 48, 64, 96, 128, 192, 256, 320};
  for (auto _ : state) {
    void* blocks[std::size(kSizes)];
    for (size_t i = 0; i < std::size(kSizes); ++i) {
      blocks[i] = std::malloc(kSizes[i]);
      do_not_optimize(blocks[i]);
    }
    for (size_t i = std::size(kSizes); i-- > 0;) {
      std::free(blocks[i]);
    }
  }
}

// Conversation length in messages.
BENCHMARK(BM_ChatTemplate_JinjaRender)
    ->RangeMultiplier(4)
    ->Range(1, 64)
    ->Unit(benchmark::kMicrosecond);
// Number of tools, each with a parameter schema.
BENCHMARK(BM_ChatTemplate_JinjaRenderWithTools)
    ->RangeMultiplier(4)
    ->Range(1, 16)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_ChatTemplate_MinjaApply)
    ->RangeMultiplier(4)
    ->Range(1, 64)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_ChatTemplate_MinjaBuiltins)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_ChatTemplate_MessagesToJson)
    ->RangeMultiplier(4)
    ->Range(1, 64)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_ChatTemplate_MinjaRenderCachedBuiltins)
    ->RangeMultiplier(4)
    ->Range(1, 64)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_ChatTemplate_AllocatorControl)->Unit(benchmark::kNanosecond);

}  // namespace
}  // namespace xllm

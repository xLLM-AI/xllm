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

#include <glog/logging.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

namespace xllm {

namespace {

// The `strftime_now(format)` global minja::chat_template::apply defines for
// every render, bound to the render's wall-clock time.
minja::Value make_strftime_now() {
  const auto now = std::chrono::system_clock::now();
  return minja::Value::callable(
      [now](const std::shared_ptr<minja::Context>& /*context*/,
            minja::ArgumentsValue& args) -> minja::Value {
        args.expectArgs("strftime_now", {1, 1}, {0, 0});
        const auto format = args.args[0].get<std::string>();
        const std::time_t time = std::chrono::system_clock::to_time_t(now);
        const std::tm local_time = *std::localtime(&time);
        std::ostringstream out;
        out << std::put_time(&local_time, format.c_str());
        return minja::Value(out.str());
      });
}
const std::unordered_map<std::string, std::string> type_to_modality = {
    {"video_url", "video"},
    {"image_url", "image"},
    {"audio_url", "audio"},
    {"image_embedding", "image"},
    {"video_embedding", "video"},
    {"audio_embedding", "audio"}};

void replace_undefined_tests(std::string& block) {
  char quote = '\0';
  for (size_t pos = 0; pos < block.size(); ++pos) {
    const char current = block[pos];
    if (quote != '\0') {
      if (current == '\\') {
        ++pos;
      } else if (current == quote) {
        quote = '\0';
      }
      continue;
    }
    if (current == '\'' || current == '"') {
      quote = current;
      continue;
    }

    static constexpr char kIsNotUndefined[] = " is not undefined";
    static constexpr char kIsNotNone[] = " is not none";
    static constexpr char kIsUndefined[] = " is undefined";
    static constexpr char kIsNone[] = " is none";
    static constexpr char kGlmOutputTest[] = "m.content.0.output is defined";
    static constexpr char kGuardedGlmOutputTest[] =
        "(m.content is not string and m.content.0.output is defined)";
    if (block.compare(pos, sizeof(kIsNotUndefined) - 1, kIsNotUndefined) == 0) {
      block.replace(pos, sizeof(kIsNotUndefined) - 1, kIsNotNone);
    } else if (block.compare(pos, sizeof(kIsUndefined) - 1, kIsUndefined) ==
               0) {
      block.replace(pos, sizeof(kIsUndefined) - 1, kIsNone);
    } else if (block.compare(pos, sizeof(kGlmOutputTest) - 1, kGlmOutputTest) ==
               0) {
      block.replace(pos, sizeof(kGlmOutputTest) - 1, kGuardedGlmOutputTest);
      pos += sizeof(kGuardedGlmOutputTest) - 2;
    }
  }
}

std::string normalize_minja_tests(std::string chat_template) {
  size_t search_pos = 0;
  while (search_pos < chat_template.size()) {
    const size_t expression_pos = chat_template.find("{{", search_pos);
    const size_t statement_pos = chat_template.find("{%", search_pos);
    const size_t block_pos = std::min(expression_pos, statement_pos);
    if (block_pos == std::string::npos) {
      break;
    }

    const std::string close = block_pos == expression_pos ? "}}" : "%}";
    const size_t close_pos = chat_template.find(close, block_pos + 2);
    if (close_pos == std::string::npos) {
      break;
    }

    std::string block =
        chat_template.substr(block_pos, close_pos + close.size() - block_pos);
    // Qwen3.8's official template uses Jinja's `is undefined` test for optional
    // arguments. Minja represents missing arguments as null and only supports
    // `is none`, so normalize the test before Minja parses the template.
    replace_undefined_tests(block);
    chat_template.replace(
        block_pos, close_pos + close.size() - block_pos, block);
    search_pos = block_pos + block.size();
  }
  return chat_template;
}
}  // namespace

JinjaChatTemplate::JinjaChatTemplate(const TokenizerArgs& args) : args_(args) {
  try {
    template_ = std::make_unique<minja::chat_template>(
        normalize_minja_tests(args_.chat_template()),
        args_.bos_token(),
        args_.eos_token());
    // Same parse options minja::chat_template uses for its own root.
    template_root_ = minja::Parser::parse(template_->source(),
                                          {/*trim_blocks=*/true,
                                           /*lstrip_blocks=*/true,
                                           /*keep_trailing_newline=*/false});
    // minja rebuilds its ~30 builtin callables for every render; they are
    // immutable, so build them once and share them across renders.
    builtins_ = minja::Context::builtins();
    LOG(INFO) << "Jinja chat template init succeed.";

  } catch (const std::exception& e) {
    LOG(FATAL) << "Failed to parse jinja chat template, TokenizerArgs: "
               << args_ << std::endl
               << "Error message: " << e.what();
  }
}

std::optional<std::string> JinjaChatTemplate::apply(
    const ChatMessages& messages) const {
  const std::vector<xllm::JsonTool> empty_tools;
  const nlohmann::ordered_json chat_template_kwargs;
  return apply(messages, empty_tools, chat_template_kwargs);
}

std::optional<std::string> JinjaChatTemplate::apply(
    const ChatMessages& messages,
    const nlohmann::ordered_json& chat_template_kwargs) const {
  const std::vector<xllm::JsonTool> empty_tools;
  return apply(messages, empty_tools, chat_template_kwargs);
}

std::optional<std::string> JinjaChatTemplate::apply(
    nlohmann::ordered_json messages) const {
  // Call the overloaded method with empty tools
  const nlohmann::ordered_json chat_template_kwargs = nlohmann::json::object();
  return apply(std::move(messages),
               nlohmann::ordered_json::array(),
               chat_template_kwargs);
}

std::optional<std::string> JinjaChatTemplate::apply(
    const ChatMessages& messages,
    const std::vector<xllm::JsonTool>& json_tools,
    const nlohmann::ordered_json& chat_template_kwargs) const {
  // convert the messages to json object
  nlohmann::ordered_json messages_json = nlohmann::json::array();
  for (const auto& message : messages) {
    nlohmann::ordered_json message_json;
    message_json["role"] = message.role;

    if (std::holds_alternative<std::string>(message.content)) {
      message_json["content"] = std::get<std::string>(message.content);
    } else if (std::holds_alternative<MMContentVec>(message.content)) {
      message_json["content"] =
          get_mm_content(std::get<MMContentVec>(message.content));
    }

    if (message.tool_call_id.has_value()) {
      message_json["tool_call_id"] = *message.tool_call_id;
    }

    if (message.reasoning_content.has_value()) {
      message_json["reasoning_content"] = *message.reasoning_content;
    }

    if (message.tool_calls.has_value()) {
      nlohmann::ordered_json tool_calls_json = nlohmann::json::array();
      const auto& tool_calls = *message.tool_calls;

      for (const auto& tool_call : tool_calls) {
        tool_calls_json.emplace_back(nlohmann::ordered_json{
            {"id", tool_call.id},
            {"type", tool_call.type},
            {"function",
             nlohmann::ordered_json{
                 {"name", tool_call.function.name},
                 {"arguments", tool_call.function.arguments}}}});
      }
      message_json["tool_calls"] = std::move(tool_calls_json);
    }

    messages_json.emplace_back(std::move(message_json));
  }

  nlohmann::ordered_json tools_json = nlohmann::json::array();

  for (const auto& json_tool : json_tools) {
    tools_json.emplace_back(nlohmann::ordered_json{
        {"type", json_tool.type},
        {"function",
         nlohmann::ordered_json{
             {"name", json_tool.function.name},
             {"description", json_tool.function.description},
             {"parameters", json_tool.function.parameters}}}});
    if (json_tool.function.defer_loading.has_value()) {
      tools_json.back()["function"]["defer_loading"] =
          *json_tool.function.defer_loading;
    }
  }
  // apply the template
  return apply(
      std::move(messages_json), std::move(tools_json), chat_template_kwargs);
}

std::optional<std::string> JinjaChatTemplate::apply(
    nlohmann::ordered_json messages,
    nlohmann::ordered_json tools,
    const nlohmann::ordered_json& chat_template_kwargs) const {
  try {
    if (!needs_polyfills(messages, tools)) {
      return render_native(messages, tools, chat_template_kwargs);
    }

    // The template cannot take these inputs as they are; let minja rewrite
    // them into a shape it does support.
    minja::chat_template_inputs input;
    // The documents were built for this call alone; moving them in spares a
    // deep copy of every message body on the request path.
    input.messages = std::move(messages);
    input.tools = std::move(tools);
    input.add_generation_prompt = true;
    input.extra_context = chat_template_kwargs;
    minja::chat_template_options options;

    return template_->apply(input, options);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to apply chat template: " << e.what();
    return std::nullopt;
  }
}

bool JinjaChatTemplate::needs_polyfills(
    const nlohmann::ordered_json& messages,
    const nlohmann::ordered_json& tools) const {
  // Mirrors the decision in minja::chat_template::apply with its default
  // chat_template_options (every polyfill enabled).
  const minja::chat_template_caps& caps = template_->original_caps();
  const bool has_tools = tools.is_array() && !tools.empty();
  bool has_tool_calls = false;
  bool has_tool_responses = false;
  bool has_string_content = false;
  for (const auto& message : messages) {
    if (const auto it = message.find("tool_calls");
        it != message.end() && !it->is_null()) {
      has_tool_calls = true;
    }
    if (const auto it = message.find("role");
        it != message.end() && *it == "tool") {
      has_tool_responses = true;
    }
    if (const auto it = message.find("content");
        it != message.end() && it->is_string()) {
      has_string_content = true;
    }
  }
  return !caps.supports_system_role || (has_tools && !caps.supports_tools) ||
         (has_tool_calls && !caps.supports_tool_calls) ||
         (has_tool_responses && !caps.supports_tool_responses) ||
         (has_tool_calls && caps.requires_object_arguments) ||
         (has_string_content && caps.requires_typed_content);
}

std::string JinjaChatTemplate::render_native(
    const nlohmann::ordered_json& messages,
    const nlohmann::ordered_json& tools,
    const nlohmann::ordered_json& chat_template_kwargs) const {
  // The context minja::chat_template::apply assembles on its no-polyfill path,
  // minus its copies of the message list, on the shared builtin globals.
  minja::Value root = minja::Value::object();
  root.set("messages", minja::Value(messages));
  root.set("add_generation_prompt", minja::Value(true));
  auto context = minja::Context::make(std::move(root), builtins_);
  context->set("bos_token", minja::Value(template_->bos_token()));
  context->set("eos_token", minja::Value(template_->eos_token()));
  context->set("strftime_now", make_strftime_now());
  if (!tools.is_null()) {
    context->set("tools", minja::Value(tools));
  }
  if (!chat_template_kwargs.is_null()) {
    for (const auto& [key, value] : chat_template_kwargs.items()) {
      context->set(key, minja::Value(value));
    }
  }
  return template_root_->render(context);
}

nlohmann::ordered_json JinjaChatTemplate::get_mm_content(
    const MMContentVec& vec) const {
  nlohmann::ordered_json content_json = nlohmann::json::array();

  for (const auto& item : vec) {
    nlohmann::ordered_json item_json;
    item_json["type"] = item.type;
    if (item.type == "text") {
      item_json["text"] = item.text;
    } else if (auto it = type_to_modality.find(item.type);
               it != type_to_modality.end()) {
      const std::string& modality = it->second;
      item_json[modality] = "mm place holder";
      item_json[item.type] = "mm place holder";
    } else {
      item_json[item.type] = "mm place holder";
    }

    content_json.emplace_back(std::move(item_json));
  }

  return content_json;
}

}  // namespace xllm

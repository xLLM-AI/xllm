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

#include "api_service/anthropic_request_utils.h"

#include <google/protobuf/util/json_util.h>

#include <cmath>
#include <string>
#include <unordered_set>

#include "core/util/uuid.h"

namespace xllm::api_service {
namespace {

Status invalid_request(std::string message) {
  return Status(StatusCode::INVALID_ARGUMENT, std::move(message));
}

void normalize_bool(nlohmann::json& value) {
  if (value.is_number() && (value == 0 || value == 1)) {
    value = value == 1;
  }
}

void normalize_block_bools(nlohmann::json& blocks) {
  if (!blocks.is_object() || !blocks.contains("blocks") ||
      !blocks["blocks"].is_array()) {
    return;
  }
  for (auto& block : blocks["blocks"]) {
    if (block.is_object() && block.contains("is_error")) {
      normalize_bool(block["is_error"]);
    }
  }
}

std::string system_text(const proto::AnthropicContentBlockList& blocks) {
  std::string text;
  for (const auto& block : blocks.blocks()) {
    if (block.type() != "text" ||
        block.text().starts_with("x-anthropic-billing-header")) {
      continue;
    }
    text += block.text();
  }
  return text;
}

std::string json_string(const google::protobuf::Message& value) {
  std::string result;
  const auto status =
      google::protobuf::util::MessageToJsonString(value, &result);
  CHECK(status.ok()) << status.ToString();
  return result;
}

void append_tool_result(const proto::AnthropicContentBlock& block,
                        std::vector<Message>& messages) {
  Message tool_message("tool", "");
  tool_message.tool_call_id = block.tool_use_id();
  std::string text = block.has_content_string() ? block.content_string() : "";
  bool first_text = true;
  for (const auto& item : block.content_list().items()) {
    const auto& fields = item.fields();
    auto type = fields.find("type");
    if (type == fields.end()) {
      continue;
    }
    if (type->second.string_value() == "text") {
      auto part = fields.find("text");
      if (!first_text) {
        text += "\n";
      }
      first_text = false;
      if (part != fields.end()) {
        text += part->second.string_value();
      }
    }
  }
  tool_message.content = std::move(text);
  messages.emplace_back(std::move(tool_message));
}

}  // namespace

Status parse_anthropic_request(const std::string& json,
                               bool count_tokens,
                               proto::AnthropicMessagesRequest& request) {
  auto body = nlohmann::json::parse(json, nullptr, false);
  if (!body.is_object() || !body.contains("messages") ||
      !body["messages"].is_array()) {
    return invalid_request("messages is required and must be an array");
  }
  if (body.contains("tools") && body["tools"].is_array()) {
    for (auto& tool : body["tools"]) {
      if (!tool.is_object() || !tool.contains("name") ||
          !tool["name"].is_string()) {
        return invalid_request("tools.name is required and must be a string");
      }
      if (tool.contains("defer_loading")) {
        normalize_bool(tool["defer_loading"]);
      }
    }
  }
  // Match Pydantic's numeric/boolean coercions before protobuf decoding.
  if (body.contains("stream")) {
    normalize_bool(body["stream"]);
  }
  for (const char* field : {"max_tokens", "temperature", "top_p", "top_k"}) {
    if (body.contains(field) && body[field].is_boolean()) {
      body[field] = body[field].get<bool>() ? 1 : 0;
    }
  }
  if (body.contains("system_blocks")) {
    normalize_block_bools(body["system_blocks"]);
  }
  for (auto& message : body["messages"]) {
    if (message.is_object() && message.contains("content_blocks")) {
      normalize_block_bools(message["content_blocks"]);
    }
  }
  const std::string normalized = body.dump();
  google::protobuf::util::JsonParseOptions options;
  options.ignore_unknown_fields = true;
  request.Clear();
  if (!count_tokens) {
    const auto status = google::protobuf::util::JsonStringToMessage(
        normalized, &request, options);
    return status.ok() ? Status() : invalid_request(status.ToString());
  }
  proto::AnthropicCountTokensRequest count;
  const auto status =
      google::protobuf::util::JsonStringToMessage(normalized, &count, options);
  if (!status.ok()) {
    return invalid_request(status.ToString());
  }
  request.set_model(count.model());
  request.set_max_tokens(1);
  request.mutable_messages()->Swap(count.mutable_messages());
  request.mutable_tools()->Swap(count.mutable_tools());
  if (count.has_system_string()) {
    request.set_system_string(count.system_string());
  } else if (count.has_system_blocks()) {
    request.mutable_system_blocks()->Swap(count.mutable_system_blocks());
  }
  if (count.has_tool_choice()) {
    request.mutable_tool_choice()->Swap(count.mutable_tool_choice());
  }
  if (count.has_chat_template_kwargs()) {
    request.mutable_chat_template_kwargs()->Swap(
        count.mutable_chat_template_kwargs());
  }
  return Status();
}

Status validate_anthropic_request(
    const proto::AnthropicMessagesRequest& request,
    bool count_tokens) {
  if (request.model().empty()) {
    return invalid_request("model is required");
  }
  if (!count_tokens && request.max_tokens() <= 0) {
    return invalid_request("max_tokens must be positive");
  }
  static const std::unordered_set<std::string> kBlockTypes = {
      "text",
      "image",
      "tool_use",
      "tool_result",
      "tool_reference",
      "thinking",
      "redacted_thinking"};
  for (const auto& block : request.system_blocks().blocks()) {
    if (!kBlockTypes.contains(block.type())) {
      return invalid_request("Unsupported system content block type");
    }
  }
  for (const auto& message : request.messages()) {
    if (message.role() != "user" && message.role() != "assistant" &&
        message.role() != "system") {
      return invalid_request(
          "messages.role must be user, assistant, or system");
    }
    if (message.message_content_case() ==
        proto::AnthropicMessage::MESSAGE_CONTENT_NOT_SET) {
      return invalid_request(
          "messages.content is required and must be a string or array");
    }
    for (const auto& block : message.content_blocks().blocks()) {
      if (!kBlockTypes.contains(block.type())) {
        return invalid_request("Unsupported content block type: " +
                               block.type());
      }
    }
  }
  if (request.has_tool_choice()) {
    const auto& choice = request.tool_choice();
    if (choice.type() != "auto" && choice.type() != "any" &&
        choice.type() != "tool" && choice.type() != "none") {
      return invalid_request(
          "tool_choice.type must be auto, any, tool, or none");
    }
    if (choice.type() == "tool" && choice.name().empty()) {
      return invalid_request("tool_choice.name is required when type is tool");
    }
  }
  for (const auto& tool : request.tools()) {
    if (!tool.has_input_schema()) {
      return invalid_request("Each tool requires name and input_schema");
    }
  }
  if (count_tokens) {
    return Status();
  }
  if (request.has_output_config()) {
    const auto& config = request.output_config();
    static const std::unordered_set<std::string> kEfforts = {
        "low", "medium", "high", "xhigh", "max"};
    if (config.has_effort() && !kEfforts.contains(config.effort())) {
      return invalid_request("Invalid output_config.effort");
    }
    if (config.has_format() && !config.format().type().empty() &&
        config.format().type() != "json_schema") {
      return invalid_request("output_config.format.type must be json_schema");
    }
  }
  return Status();
}

Status validate_anthropic_backend(
    const proto::AnthropicMessagesRequest& request,
    bool count_tokens) {
  if (request.messages().empty()) {
    return Status(StatusCode::UNKNOWN, "list index out of range");
  }
  for (const auto& message : request.messages()) {
    for (const auto& block : message.content_blocks().blocks()) {
      if (block.type() == "image") {
        return invalid_request(
            "Image input is not supported by the xLLM Messages API");
      }
      for (const auto& item : block.content_list().items()) {
        const auto& fields = item.fields();
        const auto type = fields.find("type");
        if (type == fields.end()) {
          continue;
        }
        const std::string& name = type->second.string_value();
        if (name == "image" || name == "tool_reference") {
          return invalid_request(
              "Image and tool-reference results are not supported by xLLM");
        }
        const auto text = fields.find("text");
        if (name == "text" && text != fields.end() &&
            text->second.kind_case() != google::protobuf::Value::kStringValue) {
          return Status(StatusCode::UNKNOWN,
                        "tool_result text must contain a string");
        }
      }
    }
  }
  if (count_tokens) {
    return Status();
  }
  // vLLM 0.23's router translates sampling-construction failures to HTTP 500.
  if (request.has_temperature() &&
      (!std::isfinite(request.temperature()) || request.temperature() < 0)) {
    return Status(StatusCode::UNKNOWN, "temperature must be non-negative");
  }
  if (request.has_top_p() && (!std::isfinite(request.top_p()) ||
                              request.top_p() <= 0 || request.top_p() > 1)) {
    return Status(StatusCode::UNKNOWN, "top_p must be in (0, 1]");
  }
  if (request.has_top_k() && request.top_k() < -1) {
    return Status(StatusCode::UNKNOWN,
                  "top_k must be 0 (disable), or at least 1");
  }
  if (request.has_output_config()) {
    const auto& config = request.output_config();
    if (config.has_format() && config.format().has_schema() &&
        !config.format().schema().fields().empty()) {
      return invalid_request(
          "output_config.format JSON schema constraints are not supported by "
          "xLLM");
    }
  }
  if (request.has_kv_transfer_params()) {
    return invalid_request("KV transfer parameters are not supported by xLLM");
  }
  return Status();
}

std::vector<Message> build_anthropic_messages(
    const proto::AnthropicMessagesRequest& request) {
  std::vector<Message> messages;
  messages.reserve(request.messages_size() + 1);
  std::string system = request.has_system_string()
                           ? request.system_string()
                           : system_text(request.system_blocks());
  for (const auto& message : request.messages()) {
    if (message.role() == "system") {
      system += message.has_content_string()
                    ? message.content_string()
                    : system_text(message.content_blocks());
    }
  }
  if (!system.empty()) {
    messages.emplace_back("system", std::move(system));
  }
  for (const auto& message : request.messages()) {
    const std::string& role = message.role();
    if (role == "system") {
      continue;
    }
    if (message.has_content_string()) {
      messages.emplace_back(role, message.content_string());
      continue;
    }
    MMContentVec content;
    Message::ToolCallVec calls;
    content.reserve(message.content_blocks().blocks_size());
    calls.reserve(message.content_blocks().blocks_size());
    std::string reasoning;
    bool has_reasoning = false;
    for (const auto& block : message.content_blocks().blocks()) {
      if (block.type() == "text" && !block.text().empty()) {
        content.emplace_back("text", block.text());
      } else if (block.type() == "thinking" && block.has_thinking()) {
        reasoning += block.thinking();
        has_reasoning = true;
      } else if (block.type() == "tool_use") {
        Message::ToolCall call;
        call.id = block.has_id() ? block.id() : "call_" + ShortUUID().random();
        call.type = "function";
        call.function.name = block.name();
        call.function.arguments =
            block.has_input() ? json_string(block.input()) : "{}";
        calls.emplace_back(std::move(call));
      } else if (block.type() == "tool_result") {
        if (role == "user") {
          append_tool_result(block, messages);
        } else {
          std::string text =
              block.has_content_string() ? block.content_string() : "";
          content.emplace_back("text", "Tool result: " + text);
        }
      }
    }
    if (role == "user" && content.empty()) {
      continue;
    }
    Message converted(role, "");
    if (!content.empty()) {
      // vLLM's text renderer joins content parts before applying an LLM
      // template. Passing the array through would break string-only templates.
      std::string text;
      for (const auto& part : content) {
        if (!text.empty()) {
          text += "\n";
        }
        text += part.text;
      }
      converted.content = std::move(text);
    }
    if (!calls.empty()) {
      converted.tool_calls = std::move(calls);
    }
    if (has_reasoning) {
      converted.reasoning_content = std::move(reasoning);
    }
    messages.emplace_back(std::move(converted));
  }
  return messages;
}

}  // namespace xllm::api_service

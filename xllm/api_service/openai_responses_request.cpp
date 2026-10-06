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

#include "api_service/openai_responses_request.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <unordered_set>

#include "core/util/uuid.h"

namespace xllm::api_service {
namespace {

using Json = nlohmann::json;

Status invalid(const std::string& message,
               const std::string& param,
               std::string* error_param) {
  if (error_param != nullptr) {
    *error_param = param;
  }
  return Status(StatusCode::INVALID_ARGUMENT, message);
}

Status check_fields(const Json& value,
                    std::initializer_list<std::string_view> fields,
                    const std::string& path,
                    std::string* error_param) {
  if (!value.is_object()) {
    return invalid("Expected an object", path, error_param);
  }
  for (auto it = value.begin(); it != value.end(); ++it) {
    if (std::find(fields.begin(), fields.end(), it.key()) != fields.end()) {
      continue;
    }
    return invalid("Unsupported field: " + it.key(),
                   path.empty() ? it.key() : path + "." + it.key(),
                   error_param);
  }
  return Status();
}

Status required_string(const Json& value,
                       const char* field,
                       const std::string& path,
                       std::string* result,
                       std::string* error_param,
                       bool allow_empty = false) {
  auto it = value.find(field);
  if (it == value.end() || !it->is_string()) {
    return invalid("Expected a string", path, error_param);
  }
  *result = it->get<std::string>();
  if (!allow_empty && result->empty()) {
    return invalid("String must not be empty", path, error_param);
  }
  return Status();
}

bool valid_function_name(const std::string& name) {
  return !name.empty() && name.size() <= 64 &&
         name.find_first_not_of(
             "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
             "-") == std::string::npos;
}

Status nullable_false(const Json& json,
                      const char* field,
                      std::string* error_param) {
  auto it = json.find(field);
  if (it == json.end() || it->is_null()) {
    return Status();
  }
  if (!it->is_boolean()) {
    return invalid("Expected a boolean or null", field, error_param);
  }
  if (it->get<bool>()) {
    return invalid(
        std::string(field) + " is not supported by stateless xLLM Responses",
        field,
        error_param);
  }
  return Status();
}

Status parse_content(const Json& content,
                     const std::string& role,
                     const std::string& path,
                     std::string* text,
                     std::string* error_param) {
  if (content.is_string()) {
    *text = content.get<std::string>();
    return Status();
  }
  if (!content.is_array()) {
    return invalid("Expected text or a text content array", path, error_param);
  }
  for (size_t i = 0; i < content.size(); ++i) {
    const Json& part = content[i];
    const std::string part_path = path + "[" + std::to_string(i) + "]";
    Status status = check_fields(part,
                                 {"type", "text", "annotations", "logprobs"},
                                 part_path,
                                 error_param);
    if (!status.ok()) {
      return status;
    }
    std::string type;
    status =
        required_string(part, "type", part_path + ".type", &type, error_param);
    if (!status.ok()) {
      return status;
    }
    if (type != "input_text" &&
        !(role == "assistant" && type == "output_text")) {
      return invalid(
          "Only textual input and assistant output_text are supported",
          part_path + ".type",
          error_param);
    }
    std::string value;
    status = required_string(
        part, "text", part_path + ".text", &value, error_param, true);
    if (!status.ok()) {
      return status;
    }
    for (const char* field : {"annotations", "logprobs"}) {
      auto it = part.find(field);
      if (it == part.end()) {
        continue;
      }
      if (type != "output_text" || !it->is_array()) {
        return invalid("Expected an assistant output_text array",
                       part_path + "." + field,
                       error_param);
      }
      if (std::string_view(field) == "annotations" && !it->empty()) {
        return invalid("Annotated hosted-tool output is not supported",
                       part_path + ".annotations",
                       error_param);
      }
    }
    text->append(value);
  }
  return Status();
}

Status parse_input(const Json& input,
                   std::vector<Message>* messages,
                   std::string* error_param) {
  if (input.is_string()) {
    messages->emplace_back("user", input.get<std::string>());
    return Status();
  }
  if (!input.is_array() || input.empty()) {
    return invalid(
        "input must be a string or a nonempty array", "input", error_param);
  }
  messages->reserve(messages->size() + input.size());
  std::unordered_set<std::string> calls;
  std::unordered_set<std::string> results;
  std::string pending_reasoning;
  for (size_t i = 0; i < input.size(); ++i) {
    const Json& item = input[i];
    const std::string path = "input[" + std::to_string(i) + "]";
    if (!item.is_object()) {
      return invalid("Expected an input item object", path, error_param);
    }
    std::string type = "message";
    if (item.contains("type")) {
      Status status =
          required_string(item, "type", path + ".type", &type, error_param);
      if (!status.ok()) {
        return status;
      }
    }
    if (calls.size() != results.size() && type != "function_call" &&
        type != "function_call_output") {
      return invalid(
          "Pending function calls require their outputs before a new "
          "conversation item",
          path,
          error_param);
    }
    if (type == "message") {
      Status status =
          check_fields(item,
                       {"type", "role", "content", "id", "status", "phase"},
                       path,
                       error_param);
      if (!status.ok()) {
        return status;
      }
      std::string role;
      status =
          required_string(item, "role", path + ".role", &role, error_param);
      if (!status.ok()) {
        return status;
      }
      if (role != "user" && role != "assistant" && role != "system" &&
          role != "developer") {
        return invalid("Unsupported message role", path + ".role", error_param);
      }
      if (!pending_reasoning.empty() && role != "assistant") {
        return invalid(
            "Reasoning must precede an assistant item", path, error_param);
      }
      if (!item.contains("content")) {
        return invalid(
            "Missing message content", path + ".content", error_param);
      }
      if (item.contains("phase") && !item["phase"].is_null()) {
        return invalid("Assistant phase control is not supported",
                       path + ".phase",
                       error_param);
      }
      std::string text;
      status = parse_content(
          item["content"], role, path + ".content", &text, error_param);
      if (!status.ok()) {
        return status;
      }
      // Jinja templates consume developer instructions at the system priority.
      messages->emplace_back(role == "developer" ? "system" : role,
                             std::move(text));
      if (!pending_reasoning.empty()) {
        messages->back().reasoning_content = std::move(pending_reasoning);
        pending_reasoning.clear();
      }
    } else if (type == "function_call") {
      Status status =
          check_fields(item,
                       {"type", "id", "status", "call_id", "name", "arguments"},
                       path,
                       error_param);
      if (!status.ok()) {
        return status;
      }
      Message::ToolCall call;
      call.type = "function";
      for (auto [field, value] :
           {std::pair{"call_id", &call.id},
            std::pair{"name", &call.function.name},
            std::pair{"arguments", &call.function.arguments}}) {
        status = required_string(item,
                                 field,
                                 path + "." + field,
                                 value,
                                 error_param,
                                 std::string_view(field) == "arguments");
        if (!status.ok()) {
          return status;
        }
      }
      if (!valid_function_name(call.function.name)) {
        return invalid("Invalid function name", path + ".name", error_param);
      }
      if (calls.size() != results.size() &&
          (messages->empty() || messages->back().role != "assistant")) {
        return invalid(
            "Finish the current function output group before a new "
            "function call",
            path,
            error_param);
      }
      const Json arguments =
          Json::parse(call.function.arguments, nullptr, false);
      if (!arguments.is_object()) {
        return invalid("Function arguments must encode a JSON object",
                       path + ".arguments",
                       error_param);
      }
      if (!calls.insert(call.id).second) {
        return invalid(
            "Duplicate function call_id", path + ".call_id", error_param);
      }
      if (messages->empty() || messages->back().role != "assistant") {
        messages->emplace_back("assistant", "");
      }
      Message& message = messages->back();
      if (!pending_reasoning.empty()) {
        message.reasoning_content = std::move(pending_reasoning);
        pending_reasoning.clear();
      }
      if (!message.tool_calls.has_value()) {
        message.tool_calls.emplace();
        message.tool_calls->reserve(input.size() - i);
      }
      message.tool_calls->emplace_back(std::move(call));
    } else if (type == "function_call_output") {
      Status status =
          check_fields(item,
                       {"type", "id", "status", "call_id", "output"},
                       path,
                       error_param);
      if (!status.ok()) {
        return status;
      }
      std::string call_id;
      status = required_string(
          item, "call_id", path + ".call_id", &call_id, error_param);
      if (!status.ok()) {
        return status;
      }
      if (!calls.contains(call_id) || !results.insert(call_id).second) {
        return invalid("Function output requires a unique preceding call_id",
                       path + ".call_id",
                       error_param);
      }
      if (!item.contains("output")) {
        return invalid(
            "Missing function output", path + ".output", error_param);
      }
      std::string text;
      status = parse_content(
          item["output"], "tool", path + ".output", &text, error_param);
      if (!status.ok()) {
        return status;
      }
      messages->emplace_back("tool", std::move(text));
      messages->back().tool_call_id = std::move(call_id);
    } else if (type == "reasoning") {
      Status status =
          check_fields(item,
                       {"type", "id", "status", "summary", "content"},
                       path,
                       error_param);
      if (!status.ok()) {
        return status;
      }
      if (!item.contains("summary") || !item["summary"].is_array() ||
          !item["summary"].empty()) {
        return invalid("Only raw reasoning with an empty summary is supported",
                       path + ".summary",
                       error_param);
      }
      if (!item.contains("content") || !item["content"].is_array()) {
        return invalid("Raw reasoning requires reasoning_text content",
                       path + ".content",
                       error_param);
      }
      for (size_t j = 0; j < item["content"].size(); ++j) {
        const Json& part = item["content"][j];
        const std::string part_path =
            path + ".content[" + std::to_string(j) + "]";
        status = check_fields(part, {"type", "text"}, part_path, error_param);
        if (!status.ok()) {
          return status;
        }
        if (!part.contains("type") || part["type"] != "reasoning_text") {
          return invalid(
              "Expected reasoning_text", part_path + ".type", error_param);
        }
        std::string text;
        status = required_string(
            part, "text", part_path + ".text", &text, error_param, true);
        if (!status.ok()) {
          return status;
        }
        pending_reasoning.append(text);
      }
    } else {
      return invalid(
          "Unsupported input item type: " + type, path + ".type", error_param);
    }
    for (const char* field : {"id", "status"}) {
      auto it = item.find(field);
      if (it != item.end() && !it->is_string()) {
        return invalid("Expected a string", path + "." + field, error_param);
      }
    }
    if (item.contains("status") && item["status"] != "in_progress" &&
        item["status"] != "completed" && item["status"] != "incomplete") {
      return invalid(
          "Unsupported input item status", path + ".status", error_param);
    }
  }
  if (!pending_reasoning.empty()) {
    return invalid("Reasoning must be followed by an assistant item",
                   "input",
                   error_param);
  }
  if (calls.size() != results.size()) {
    return invalid("Every historical function call requires its output",
                   "input",
                   error_param);
  }
  return Status();
}

Status parse_tools(const Json& tools,
                   ResponsesRequest* request,
                   std::string* error_param) {
  if (!tools.is_array()) {
    return invalid("tools must be an array", "tools", error_param);
  }
  request->params.tools.reserve(tools.size());
  std::unordered_set<std::string> names;
  for (size_t i = 0; i < tools.size(); ++i) {
    const Json& value = tools[i];
    const std::string path = "tools[" + std::to_string(i) + "]";
    Status status =
        check_fields(value,
                     {"type", "name", "description", "parameters", "strict"},
                     path,
                     error_param);
    if (!status.ok()) {
      return status;
    }
    if (!value.contains("type") || value["type"] != "function") {
      return invalid("Only client function tools are supported",
                     path + ".type",
                     error_param);
    }
    JsonTool tool;
    status = required_string(
        value, "name", path + ".name", &tool.function.name, error_param);
    if (!status.ok()) {
      return status;
    }
    if (!valid_function_name(tool.function.name)) {
      return invalid("Invalid function name", path + ".name", error_param);
    }
    if (!names.insert(tool.function.name).second) {
      return invalid("Duplicate function name", path + ".name", error_param);
    }
    if (value.contains("strict") && !value["strict"].is_null()) {
      if (!value["strict"].is_boolean()) {
        return invalid(
            "strict must be boolean or null", path + ".strict", error_param);
      }
      if (value["strict"].get<bool>()) {
        return invalid("Strict function constraints are not implemented",
                       path + ".strict",
                       error_param);
      }
    }
    if (value.contains("description") && !value["description"].is_null()) {
      status = required_string(value,
                               "description",
                               path + ".description",
                               &tool.function.description,
                               error_param,
                               true);
      if (!status.ok()) {
        return status;
      }
    }
    if (value.contains("parameters") && !value["parameters"].is_null()) {
      if (!value["parameters"].is_object()) {
        return invalid("Function parameters must be a JSON schema object",
                       path + ".parameters",
                       error_param);
      }
      tool.function.parameters = value["parameters"];
    } else {
      tool.function.parameters =
          Json{{"type", "object"}, {"properties", Json::object()}};
    }
    request->tools.emplace_back(Json{{"type", "function"},
                                     {"name", tool.function.name},
                                     {"description", tool.function.description},
                                     {"parameters", tool.function.parameters},
                                     {"strict", false}});
    request->params.tools.emplace_back(std::move(tool));
  }
  return Status();
}

}  // namespace

std::pair<Status, ResponsesRequest> parse_responses_request(
    std::string_view body,
    const std::string& default_model,
    std::string* error_param) {
  if (error_param != nullptr) {
    error_param->clear();
  }
  ResponsesRequest request;
  const Json json = Json::parse(body, nullptr, false);
  Status status = check_fields(json,
                               {"model",
                                "input",
                                "instructions",
                                "max_output_tokens",
                                "temperature",
                                "top_p",
                                "stream",
                                "store",
                                "background",
                                "previous_response_id",
                                "tools",
                                "tool_choice",
                                "parallel_tool_calls",
                                "text",
                                "reasoning",
                                "metadata",
                                "truncation",
                                "include"},
                               "",
                               error_param);
  if (!status.ok()) {
    return {status, std::move(request)};
  }
  request.model = default_model;
  if (json.contains("model")) {
    status =
        required_string(json, "model", "model", &request.model, error_param);
    if (!status.ok()) {
      return {status, std::move(request)};
    }
  }
  if (request.model.empty()) {
    return {invalid("model is required", "model", error_param),
            std::move(request)};
  }
  for (const char* field : {"store", "background"}) {
    status = nullable_false(json, field, error_param);
    if (!status.ok()) {
      return {status, std::move(request)};
    }
  }
  if (json.contains("previous_response_id") &&
      !json["previous_response_id"].is_null()) {
    return {invalid("previous_response_id requires server storage; replay "
                    "history in input instead",
                    "previous_response_id",
                    error_param),
            std::move(request)};
  }
  request.params.temperature = 1.0f;
  request.params.top_p = 1.0f;
  request.params.n = 1;
  request.params.responses_request = true;
  request.params.request_id = "resp_" + ShortUUID().random();
  for (auto [field, target, upper] :
       {std::tuple{"temperature", &request.params.temperature, 2.0},
        std::tuple{"top_p", &request.params.top_p, 1.0}}) {
    auto it = json.find(field);
    if (it == json.end() || it->is_null()) {
      continue;
    }
    if (!it->is_number()) {
      return {invalid("Expected a number or null", field, error_param),
              std::move(request)};
    }
    const double value = it->get<double>();
    if (!std::isfinite(value) || value < 0 || value > upper) {
      return {invalid(std::string(field) + " is outside its supported range",
                      field,
                      error_param),
              std::move(request)};
    }
    *target = static_cast<float>(value);
  }
  if (json.contains("max_output_tokens") &&
      !json["max_output_tokens"].is_null()) {
    const Json& value = json["max_output_tokens"];
    if (!value.is_number_integer() || value < 16 ||
        value > std::numeric_limits<uint32_t>::max()) {
      return {invalid("max_output_tokens must be an integer between 16 and the "
                      "engine token limit",
                      "max_output_tokens",
                      error_param),
              std::move(request)};
    }
    request.params.max_tokens = value.get<uint32_t>();
    request.max_output_tokens = value;
  }
  if (json.contains("stream") && !json["stream"].is_null()) {
    if (!json["stream"].is_boolean()) {
      return {
          invalid("stream must be a boolean or null", "stream", error_param),
          std::move(request)};
    }
    request.params.streaming = json["stream"].get<bool>();
  }
  if (json.contains("instructions") && !json["instructions"].is_null()) {
    if (!json["instructions"].is_string()) {
      return {
          invalid(
              "instructions must be text or null", "instructions", error_param),
          std::move(request)};
    }
    request.instructions = json["instructions"];
    request.messages.emplace_back("system",
                                  request.instructions.get<std::string>());
  }
  if (!json.contains("input")) {
    return {
        invalid(
            "input is required for stateless generation", "input", error_param),
        std::move(request)};
  }
  status = parse_input(json["input"], &request.messages, error_param);
  if (!status.ok()) {
    return {status, std::move(request)};
  }
  if (json.contains("tools")) {
    status = parse_tools(json["tools"], &request, error_param);
    if (!status.ok()) {
      return {status, std::move(request)};
    }
  }
  if (json.contains("tool_choice")) {
    if (!json["tool_choice"].is_string() ||
        (json["tool_choice"] != "auto" && json["tool_choice"] != "none")) {
      return {
          invalid(
              "Only auto and none tool_choice are enforceable by this backend",
              "tool_choice",
              error_param),
          std::move(request)};
    }
    request.params.tool_choice = json["tool_choice"].get<std::string>();
  }
  if (request.params.tool_choice == "none") {
    request.params.tools.clear();
  }
  if (json.contains("parallel_tool_calls") &&
      !json["parallel_tool_calls"].is_null()) {
    if (!json["parallel_tool_calls"].is_boolean() ||
        !json["parallel_tool_calls"].get<bool>()) {
      return {
          invalid("parallel_tool_calls=false is not enforced by this backend",
                  "parallel_tool_calls",
                  error_param),
          std::move(request)};
    }
  }
  if (json.contains("text")) {
    status = check_fields(json["text"], {"format"}, "text", error_param);
    if (!status.ok()) {
      return {status, std::move(request)};
    }
    if (json["text"].contains("format")) {
      const Json& format = json["text"]["format"];
      status = check_fields(format, {"type"}, "text.format", error_param);
      if (!status.ok()) {
        return {status, std::move(request)};
      }
      if (!format.contains("type") || !format["type"].is_string() ||
          (format["type"] != "text" && format["type"] != "json_object")) {
        return {
            invalid(
                "Only text and engine-constrained json_object are supported",
                "text.format.type",
                error_param),
            std::move(request)};
      }
      if (format["type"] == "json_object") {
        if (!request.params.tools.empty()) {
          return {invalid("json_object and function tools cannot be combined",
                          "text.format",
                          error_param),
                  std::move(request)};
        }
        request.params.response_format = ResponseFormatType::JSON_OBJECT;
      }
    }
  }
  if (json.contains("reasoning") && !json["reasoning"].is_null()) {
    status = check_fields(
        json["reasoning"], {"effort", "summary"}, "reasoning", error_param);
    if (!status.ok()) {
      return {status, std::move(request)};
    }
    for (const char* field : {"effort", "summary"}) {
      if (json["reasoning"].contains(field) &&
          !json["reasoning"][field].is_null()) {
        return {invalid("The engine has no enforceable reasoning " +
                            std::string(field) + " control",
                        std::string("reasoning.") + field,
                        error_param),
                std::move(request)};
      }
    }
    request.reasoning = json["reasoning"];
  }
  if (json.contains("include") && !json["include"].is_null() &&
      (!json["include"].is_array() || !json["include"].empty())) {
    return {invalid("No optional hosted include capability is supported",
                    "include",
                    error_param),
            std::move(request)};
  }
  if (json.contains("truncation") && !json["truncation"].is_null() &&
      json["truncation"] != "disabled") {
    return {
        invalid(
            "Only truncation=disabled is supported", "truncation", error_param),
        std::move(request)};
  }
  if (json.contains("metadata") && !json["metadata"].is_null()) {
    const Json& metadata = json["metadata"];
    if (!metadata.is_object() || metadata.size() > 16) {
      return {invalid("metadata must contain at most 16 string pairs",
                      "metadata",
                      error_param),
              std::move(request)};
    }
    for (auto it = metadata.begin(); it != metadata.end(); ++it) {
      if (it.key().size() > 64 || !it.value().is_string() ||
          it.value().get_ref<const std::string&>().size() > 512) {
        return {invalid("Invalid metadata key or value",
                        "metadata." + it.key(),
                        error_param),
                std::move(request)};
      }
    }
    request.metadata = metadata;
  }
  return {Status(), std::move(request)};
}

nlohmann::json responses_initial_response(const ResponsesRequest& request) {
  return {{"id", request.params.request_id},
          {"object", "response"},
          {"created_at", absl::ToUnixSeconds(absl::Now())},
          {"status", "in_progress"},
          {"completed_at", nullptr},
          {"access_programs", nullptr},
          {"error", nullptr},
          {"incomplete_details", nullptr},
          {"model", request.model},
          {"instructions", request.instructions},
          {"output", Json::array()},
          {"usage", nullptr},
          {"tools", request.tools},
          {"tool_choice", request.params.tool_choice},
          {"parallel_tool_calls", true},
          {"temperature", request.params.temperature},
          {"top_p", request.params.top_p},
          {"max_output_tokens", request.max_output_tokens},
          {"text",
           {{"format",
             {{"type",
               request.params.response_format == ResponseFormatType::JSON_OBJECT
                   ? "json_object"
                   : "text"}}}}},
          {"reasoning", request.reasoning},
          {"metadata", request.metadata},
          {"store", false},
          {"background", false},
          {"previous_response_id", nullptr},
          {"truncation", "disabled"}};
}

}  // namespace xllm::api_service

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

#include "function_call/base_format_detector.h"

#include <algorithm>
#include <utility>

#include "function_call/partial_json_parser/include/partial_json_parser/parser.h"

namespace xllm {
namespace function_call {

BaseFormatDetector::BaseFormatDetector()
    : current_tool_id_(-1),
      current_tool_name_sent_(false),
      bot_token_(""),
      eot_token_(""),
      tool_call_separator_(", ") {}

std::unordered_map<std::string, int32_t> BaseFormatDetector::get_tool_indices(
    const std::vector<JsonTool>& tools) const {
  std::unordered_map<std::string, int32_t> indices;
  for (size_t i = 0; i < tools.size(); ++i) {
    if (!tools[i].function.name.empty()) {
      indices[tools[i].function.name] = static_cast<int32_t>(i);
    } else {
      LOG(ERROR) << "Tool at index " << i
                 << " has empty function name, skipping";
    }
  }
  return indices;
}

std::vector<ToolCallItem> BaseFormatDetector::parse_base_json(
    const nlohmann::json& json_obj,
    const std::vector<JsonTool>& tools) {
  auto tool_indices = get_tool_indices(tools);
  std::vector<ToolCallItem> results;

  std::vector<nlohmann::json> actions;
  if (json_obj.is_array()) {
    for (const auto& item : json_obj) {
      actions.emplace_back(item);
    }
  } else {
    actions.emplace_back(json_obj);
  }

  for (const auto& act : actions) {
    if (!act.is_object()) {
      LOG(ERROR) << "Invalid tool call item, expected object, got: "
                 << act.type_name();
      continue;
    }

    std::string name;
    if (act.contains("name") && act["name"].is_string()) {
      name = act["name"].get<std::string>();
    } else {
      LOG(ERROR) << "Invalid tool call: missing 'name' field or invalid type";
      continue;
    }

    if (tool_indices.find(name) == tool_indices.end()) {
      LOG(ERROR) << "Model attempted to call undefined function: " << name;
      continue;
    }

    nlohmann::json parameters = nlohmann::json::object();

    if (act.contains("parameters")) {
      parameters = act["parameters"];
    } else if (act.contains("arguments")) {
      parameters = act["arguments"];
    } else {
      LOG(ERROR) << "No parameters or arguments field found for tool: " << name;
    }

    if (!parameters.is_object()) {
      LOG(ERROR) << "Invalid arguments type for tool: " << name
                 << ", expected object, got: " << parameters.type_name();
      parameters = nlohmann::json::object();
    }

    std::string parameters_str;
    try {
      parameters_str = parameters.dump(
          -1, ' ', false, nlohmann::json::error_handler_t::ignore);
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to serialize arguments for tool: " << name
                 << ", error: " << e.what();
      parameters_str = "{}";
    }

    results.emplace_back(-1, name, parameters_str);
  }

  return results;
}

int32_t BaseFormatDetector::ends_with_partial_token(
    const std::string& buffer,
    const std::string& bot_token) const {
  const size_t limit = std::min(buffer.size(), bot_token.size());
  for (size_t length = limit; length > 0; --length) {
    if (buffer.compare(buffer.size() - length, length, bot_token, 0, length) ==
        0) {
      return static_cast<int32_t>(length);
    }
  }
  return 0;
}

StreamingParseResult BaseFormatDetector::parse_streaming_increment(
    const std::string& new_text,
    const std::vector<JsonTool>& tools) {
  CHECK(!stream_finished_) << "Tool parsing continued after finish";
  buffer_ += new_text;
  return consume_json_buffer(tools);
}

StreamingParseResult BaseFormatDetector::consume_json_buffer(
    const std::vector<JsonTool>& tools,
    bool final) {
  StreamingParseResult result;
  if (tool_indices_.empty()) {
    tool_indices_ = get_tool_indices(tools);
  }

  while (!buffer_.empty()) {
    if (awaiting_tool_end_) {
      if (!eot_token_.empty() && buffer_.find(eot_token_) == 0) {
        buffer_.erase(0, eot_token_.size());
        awaiting_tool_end_ = false;
        continue;
      }
      if (!eot_token_.empty() && eot_token_.find(buffer_) == 0) {
        if (final) {
          buffer_.clear();
        }
        break;
      }
      awaiting_tool_end_ = false;
    }

    if (!current_tool_name_sent_ && current_tool_id_ > 0 &&
        !tool_call_separator_.empty() &&
        buffer_.find(tool_call_separator_) == 0) {
      const std::string following = buffer_.substr(tool_call_separator_.size());
      if (following.find(bot_token_) == 0) {
        buffer_.erase(0, tool_call_separator_.size());
      } else if (!final && bot_token_.find(following) == 0) {
        break;
      }
    }

    const size_t start = buffer_.find(bot_token_);
    if (start == std::string::npos) {
      const size_t held = final ? 0
                                : static_cast<size_t>(ends_with_partial_token(
                                      buffer_, bot_token_));
      result.normal_text += buffer_.substr(0, buffer_.size() - held);
      buffer_.erase(0, buffer_.size() - held);
      break;
    }
    if (start > 0) {
      result.normal_text += buffer_.substr(0, start);
      buffer_.erase(0, start);
    }

    const std::string json_part = buffer_.substr(bot_token_.size());
    const int32_t leading = partial_json_parser::skip_blank(json_part, 0);
    const std::string value = json_part.substr(leading);
    if (value.empty()) {
      break;
    }

    try {
      if (value.front() != '{') {
        break;
      }
      std::string name;
      size_t arguments_start = std::string::npos;
      size_t arguments_end = std::string::npos;
      bool arguments_complete = false;
      int32_t position = partial_json_parser::skip_blank(value, 1);
      while (static_cast<size_t>(position) < value.size() &&
             value[position] != '}') {
        const auto key = partial_json_parser::complete_string(
            value.substr(position), partial_json_parser::ALL);
        if (!key.string.empty()) {
          break;
        }
        const std::string field =
            nlohmann::json::parse(value.substr(position, key.index))
                .get<std::string>();
        position = partial_json_parser::skip_blank(value, position + key.index);
        if (static_cast<size_t>(position) >= value.size() ||
            value[position] != ':') {
          break;
        }
        position = partial_json_parser::skip_blank(value, position + 1);
        if (static_cast<size_t>(position) >= value.size()) {
          break;
        }
        const bool arguments_field =
            field == "arguments" || field == "parameters";
        if (arguments_field) {
          if (arguments_start != std::string::npos) {
            LOG(ERROR) << "Model generated duplicate argument fields";
            has_pending_tool_ = true;
            break;
          }
          if (value[position] != '{') {
            has_pending_tool_ = true;
            break;
          }
          arguments_start = static_cast<size_t>(position);
          arguments_end = arguments_start + 1;
        }
        partial_json_parser::JsonCompletion item;
        try {
          item = partial_json_parser::complete_any(
              value.substr(position), partial_json_parser::ALL, false);
        } catch (const partial_json_parser::MalformedJSONException&) {
          break;
        }
        if (arguments_field) {
          arguments_end = arguments_start + item.index;
          arguments_complete = item.string.empty();
        } else if (field == "name" && item.string.empty()) {
          const auto parsed =
              nlohmann::json::parse(value.substr(position, item.index));
          if (parsed.is_string()) {
            name = parsed.get<std::string>();
          }
        }
        if (!item.string.empty()) {
          break;
        }
        position =
            partial_json_parser::skip_blank(value, position + item.index);
        if (static_cast<size_t>(position) >= value.size() ||
            value[position] != ',') {
          break;
        }
        position = partial_json_parser::skip_blank(value, position + 1);
      }

      if (has_pending_tool_ || name.empty()) {
        break;
      }
      if (tool_indices_.find(name) == tool_indices_.end()) {
        LOG(ERROR) << "Model attempted to call undefined function: " << name;
        buffer_.clear();
        current_tool_name_sent_ = false;
        break;
      }
      if (!current_tool_name_sent_) {
        if (current_tool_id_ < 0) {
          current_tool_id_ = 0;
        }
        const size_t count = static_cast<size_t>(current_tool_id_) + 1;
        prev_tool_call_arr_.resize(count);
        streamed_args_for_tool_.resize(count);
        result.calls.emplace_back(current_tool_id_, name, "");
        current_tool_name_sent_ = true;
        prev_tool_call_arr_[current_tool_id_]["name"] = name;
      }
      if (prev_tool_call_arr_[current_tool_id_]["name"] != name) {
        LOG(ERROR) << "Tool parser changed an emitted function name";
        has_pending_tool_ = true;
        break;
      }

      if (arguments_start != std::string::npos) {
        if (final && !arguments_complete) {
          arguments_end = value.size();
        }
        std::string& sent = streamed_args_for_tool_[current_tool_id_];
        if (value.compare(arguments_start, sent.size(), sent) != 0) {
          LOG(ERROR) << "Tool arguments do not extend the emitted prefix";
          has_pending_tool_ = true;
          break;
        }
        // Completion cursors may retreat while a later escape is incomplete.
        arguments_end = std::max(arguments_end, arguments_start + sent.size());
        const std::string arguments =
            value.substr(arguments_start, arguments_end - arguments_start);
        if (arguments.size() > sent.size()) {
          result.calls.emplace_back(
              current_tool_id_, std::nullopt, arguments.substr(sent.size()));
          sent = arguments;
        }
        prev_tool_call_arr_[current_tool_id_]["arguments"] = sent;
      }

      const auto completion = partial_json_parser::complete_any(
          value, partial_json_parser::ALL, true);
      const size_t end = static_cast<size_t>(leading + completion.index);
      const bool complete = completion.string.empty();
      const bool complete_arguments =
          final && arguments_complete && completion.string == "}" &&
          static_cast<size_t>(partial_json_parser::skip_blank(
              json_part, static_cast<int32_t>(end))) == json_part.size();

      if ((!complete && !complete_arguments) ||
          arguments_start == std::string::npos) {
        break;
      }
      buffer_.erase(0, bot_token_.size() + end);
      current_tool_name_sent_ = false;
      ++current_tool_id_;
      awaiting_tool_end_ = true;
    } catch (const partial_json_parser::MalformedJSONException&) {
      break;
    } catch (const nlohmann::json::exception&) {
      break;
    }
  }
  return result;
}

StreamingFinishResult BaseFormatDetector::finish_stream(
    const std::vector<JsonTool>& /*tools*/) {
  stream_finished_ = true;
  return {{}, has_pending_tool_};
}

StreamingFinishResult BaseFormatDetector::finish_json_buffer(
    const std::vector<JsonTool>& tools) {
  if (stream_finished_) {
    return {{}, has_pending_tool_};
  }
  StreamingParseResult result = consume_json_buffer(tools, /*final=*/true);
  has_pending_tool_ = has_pending_tool_ || current_tool_name_sent_ ||
                      buffer_.find(bot_token_) != std::string::npos;
  buffer_.clear();
  current_tool_name_sent_ = false;
  awaiting_tool_end_ = false;
  stream_finished_ = true;
  return {std::move(result), has_pending_tool_};
}

}  // namespace function_call
}  // namespace xllm

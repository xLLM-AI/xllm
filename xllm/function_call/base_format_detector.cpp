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

#include "base_format_detector.h"

#include <algorithm>
#include <iostream>
#include <regex>
#include <sstream>

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
  // Check if buffer ends with a partial bot_token.
  // Return the length of the partial bot_token.
  // For some format, the bot_token is not a token in model's vocabulary, such
  // as
  // `[TOOL_CALLS] [` in Mistral.
  for (int32_t i = 1; i <= std::min(static_cast<int32_t>(buffer.length()),
                                    static_cast<int32_t>(bot_token.length()));
       ++i) {
    if (bot_token.substr(0, i) == buffer.substr(buffer.length() - i)) {
      return i;
    }
  }
  return 0;
}

StreamingParseResult BaseFormatDetector::parse_streaming_increment(
    const std::string& new_text,
    const std::vector<JsonTool>& tools) {
  // Streaming incremental parsing with tool validation.
  // This base implementation works best with formats where:
  // 1. bot_token is followed immediately by JSON (e.g., bot_token + JSON_array)
  // 2. JSON can be parsed incrementally using partial_json_loads
  // 3. Multiple tool calls are separated by "; " or ", "
  //
  // Examples of incompatible formats (need custom implementation, may reuse
  // some logic from this class):
  // - Each tool call is wrapped in a separate block: See Qwen25Detector
  // - Multiple separate blocks: [TOOL_CALLS] [...] \n [TOOL_CALLS] [...]
  // - Tool call is Pythonic style
  //
  // For incompatible formats, detectors should override this method with custom
  // logic.

  // Append new text to buffer
  buffer_ += new_text;
  std::string current_text = buffer_;

  // The current_text has tool_call if it is the start of a new tool call
  // sequence or it is the start of a new tool call after a tool call separator,
  // when there is a previous tool call
  if (!(has_tool_call(current_text) ||
        (current_tool_id_ > 0 &&
         current_text.find(tool_call_separator_) == 0))) {
    if (ends_with_partial_token(buffer_, bot_token_) == 0) {
      std::string normal_text = buffer_;
      buffer_.clear();

      size_t eot_pos = normal_text.find(eot_token_);
      if (eot_pos != std::string::npos) {
        normal_text = normal_text.substr(0, eot_pos) +
                      normal_text.substr(eot_pos + eot_token_.length());
      }

      return StreamingParseResult(normal_text, {});
    } else {
      return StreamingParseResult();
    }
  }

  if (tool_indices_.empty()) {
    tool_indices_ = get_tool_indices(tools);
  }

  Allow flags =
      current_tool_name_sent_ ? Allow::ALL : (Allow::ALL & ~Allow::STR);

  try {
    int32_t start_idx = 0;

    if (current_text.find(bot_token_) == 0) {
      start_idx = bot_token_.length();
    } else if (current_tool_id_ > 0 &&
               current_text.find(tool_call_separator_ + bot_token_) == 0) {
      start_idx = tool_call_separator_.length() + bot_token_.length();
    } else if (current_tool_id_ > 0 &&
               current_text.find(tool_call_separator_) == 0) {
      start_idx = tool_call_separator_.length();
    }

    if (start_idx >= static_cast<int32_t>(current_text.length())) {
      return StreamingParseResult();
    }

    std::string json_part = current_text.substr(start_idx);
    auto [obj, end_idx] = partial_json_loads(json_part, flags);

    bool is_current_complete = is_complete_json(json_part.substr(0, end_idx));

    if (obj.contains("name") && obj["name"].is_string()) {
      std::string tool_name = obj["name"].get<std::string>();
      if (tool_indices_.find(tool_name) == tool_indices_.end()) {
        buffer_.clear();
        current_tool_id_ = -1;
        current_tool_name_sent_ = false;
        if (!streamed_args_for_tool_.empty()) {
          streamed_args_for_tool_.pop_back();
        }
        return StreamingParseResult();
      }
    }

    nlohmann::json current_tool_call = obj;
    if (current_tool_call.contains("parameters")) {
      if (current_tool_call.contains("arguments")) {
        LOG(ERROR) << "Model generated both parameters and arguments";
        return StreamingParseResult();
      }
      current_tool_call["arguments"] = current_tool_call["parameters"];
    }

    if (current_tool_call.empty()) {
      return StreamingParseResult();
    }

    StreamingParseResult res;

    // Case 1: Handle tool name streaming
    if (!current_tool_name_sent_) {
      if (current_tool_call.contains("name") &&
          current_tool_call["name"].is_string()) {
        std::string function_name =
            current_tool_call["name"].get<std::string>();

        if (tool_indices_.find(function_name) != tool_indices_.end()) {
          // If this is a new tool (current_tool_id was -1), initialize it
          if (current_tool_id_ == -1) {
            current_tool_id_ = 0;
            streamed_args_for_tool_.push_back("");
          }
          // If this is a subsequent tool, ensure streamed_args_for_tool is
          // large enough
          else if (current_tool_id_ >=
                   static_cast<int32_t>(streamed_args_for_tool_.size())) {
            while (static_cast<int32_t>(streamed_args_for_tool_.size()) <=
                   current_tool_id_) {
              streamed_args_for_tool_.push_back("");
            }
          }

          // Send the tool name with empty parameters
          res = StreamingParseResult(
              "", {ToolCallItem(current_tool_id_, function_name, "")});
          current_tool_name_sent_ = true;
        } else {
          res = StreamingParseResult();
        }
      } else {
        res = StreamingParseResult();
      }
    }
    // Case 2: Handle streaming arguments
    else {
      if (current_tool_call.contains("arguments")) {
        nlohmann::json cur_arguments = current_tool_call["arguments"];

        // Calculate how much of the arguments we've already streamed
        int sent = streamed_args_for_tool_[current_tool_id_].length();
        std::string cur_args_json = cur_arguments.dump();

        std::string argument_diff;
        int completing_tool_id = current_tool_id_;

        // If the current tool's JSON is complete, send all remaining arguments
        if (is_current_complete) {
          argument_diff = cur_args_json.substr(sent);

          // Only remove the processed portion, keep unprocessed content
          buffer_ = current_text.substr(start_idx + end_idx);

          if (current_tool_id_ < static_cast<int>(prev_tool_call_arr_.size())) {
            prev_tool_call_arr_[current_tool_id_].clear();
          }
          current_tool_name_sent_ = false;
          streamed_args_for_tool_[current_tool_id_] = "";
          current_tool_id_++;
        }
        // If the tool is still being parsed, send incremental changes
        else if (current_tool_id_ <
                 static_cast<int>(prev_tool_call_arr_.size())) {
          auto prev_args_it =
              prev_tool_call_arr_[current_tool_id_].find("arguments");
          if (prev_args_it != prev_tool_call_arr_[current_tool_id_].end()) {
            std::string prev_args_json = prev_args_it->second;
            if (cur_args_json != prev_args_json) {
              std::string prefix =
                  find_common_prefix(prev_args_json, cur_args_json);
              argument_diff = prefix.substr(sent);
            }
          }
        }

        if (!argument_diff.empty()) {
          int tool_index_to_use =
              is_current_complete ? completing_tool_id : current_tool_id_;
          res = StreamingParseResult(
              "",
              {ToolCallItem(tool_index_to_use, std::nullopt, argument_diff)});

          if (!is_current_complete) {
            streamed_args_for_tool_[current_tool_id_] += argument_diff;
          }
        } else {
          res = StreamingParseResult();
        }
      } else {
        res = StreamingParseResult();
      }
    }

    if (current_tool_id_ >= 0) {
      while (static_cast<int>(prev_tool_call_arr_.size()) <= current_tool_id_) {
        prev_tool_call_arr_.push_back({});
      }

      std::unordered_map<std::string, std::string> tool_call_map;
      if (current_tool_call.contains("name") &&
          current_tool_call["name"].is_string()) {
        tool_call_map["name"] = current_tool_call["name"].get<std::string>();
      }
      if (current_tool_call.contains("arguments")) {
        tool_call_map["arguments"] = current_tool_call["arguments"].dump();
      }

      prev_tool_call_arr_[current_tool_id_] = tool_call_map;
    }

    return res;

  } catch (const std::exception& e) {
    return StreamingParseResult();
  }
}

}  // namespace function_call
}  // namespace xllm

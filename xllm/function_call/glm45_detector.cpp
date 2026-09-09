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

#include "glm45_detector.h"

#include <algorithm>
#include <iostream>
#include <sstream>

namespace xllm {
namespace function_call {

Glm45Detector::Glm45Detector() : BaseFormatDetector() {
  bot_token_ = "<tool_call>";
  eot_token_ = "</tool_call>";

  // Regex patterns for GLM-4.5 format
  func_call_regex_ = std::regex("<tool_call>[\\s\\S]*?</tool_call>",
                                std::regex_constants::ECMAScript);
  func_detail_regex_ =
      std::regex("<tool_call>([^\\n]*)\\n([\\s\\S]*?)</tool_call>",
                 std::regex_constants::ECMAScript);
  func_arg_regex_ = std::regex(
      "<arg_key>([\\s\\S]*?)</arg_key>\\s*<arg_value>([\\s\\S]*?)</arg_value>",
      std::regex_constants::ECMAScript);
}

std::string Glm45Detector::trim_whitespace(std::string_view str) const {
  const char* whitespace = " \t\n\r";

  size_t start = str.find_first_not_of(whitespace);
  if (start == std::string_view::npos) {
    return std::string{};
  }

  size_t end = str.find_last_not_of(whitespace);

  return std::string(str.substr(start, end - start + 1));
}

bool Glm45Detector::has_tool_call(const std::string& text) {
  return text.find(bot_token_) != std::string::npos;
}

StreamingParseResult Glm45Detector::detect_and_parse(
    const std::string& text,
    const std::vector<JsonTool>& tools) {
  size_t idx = text.find(bot_token_);
  std::string normal_text =
      (idx != std::string::npos) ? text.substr(0, idx) : text;

  // Trim normal text
  if (!normal_text.empty()) {
    normal_text = trim_whitespace(normal_text);
  }

  if (idx == std::string::npos) {
    return StreamingParseResult(normal_text, {});
  }

  std::vector<ToolCallItem> calls;

  try {
    std::sregex_iterator iter(text.begin(), text.end(), func_call_regex_);
    std::sregex_iterator end;

    for (; iter != end; ++iter) {
      std::smatch match = *iter;
      std::string match_result = match.str();

      // Parse function name and arguments
      std::smatch func_detail;
      if (std::regex_search(match_result, func_detail, func_detail_regex_)) {
        std::string func_name = func_detail[1].str();
        std::string func_args = func_detail[2].str();

        // Parse arguments using regex
        std::unordered_map<std::string, nlohmann::json> arguments;
        std::sregex_iterator arg_iter(
            func_args.begin(), func_args.end(), func_arg_regex_);
        std::sregex_iterator arg_end;

        for (; arg_iter != arg_end; ++arg_iter) {
          std::smatch arg_match = *arg_iter;
          if (arg_match.size() >= 3) {
            std::string arg_key = arg_match[1].str();
            std::string arg_value = arg_match[2].str();

            arg_key = trim_whitespace(arg_key);

            arg_value = trim_whitespace(arg_value);

            try {
              nlohmann::json parsed_value = nlohmann::json::parse(arg_value);
              arguments[arg_key] = parsed_value;
            } catch (const nlohmann::json::parse_error&) {
              arguments[arg_key] = nlohmann::json(arg_value);
            }
          }
        }

        // Create JSON object for parse_base_json
        nlohmann::json match_json;
        match_json["name"] = func_name;
        match_json["parameters"] = arguments;

        auto parsed_calls = parse_base_json(match_json, tools);
        calls.insert(calls.end(),
                     std::make_move_iterator(parsed_calls.begin()),
                     std::make_move_iterator(parsed_calls.end()));
      }
    }

    return StreamingParseResult(normal_text, calls);

  } catch (const std::exception& e) {
    LOG(ERROR) << "Error in GLM-4.5 detect_and_parse: " << e.what();
    return StreamingParseResult(text, {});
  }
}

StreamingParseResult Glm45Detector::parse_streaming_increment(
    const std::string& new_text,
    const std::vector<JsonTool>& tools) {
  buffer_ += new_text;
  std::string current_text = buffer_;

  size_t start = current_text.find(bot_token_);
  if (start == std::string::npos) {
    buffer_.clear();
    if (current_tool_id_ > 0) {
      current_text = "";
    }
    return StreamingParseResult(current_text, {});
  }

  // Look for complete tool call
  size_t end = current_text.find(eot_token_);
  if (end != std::string::npos) {
    // Initialize state if this is the first tool call
    if (current_tool_id_ == -1) {
      current_tool_id_ = 0;
      prev_tool_call_arr_.clear();
      streamed_args_for_tool_.clear();
      streamed_args_for_tool_.push_back("");
    }

    // Ensure we have enough entries in tracking arrays
    while (prev_tool_call_arr_.size() <= current_tool_id_) {
      prev_tool_call_arr_.push_back({});
    }
    while (streamed_args_for_tool_.size() <= current_tool_id_) {
      streamed_args_for_tool_.push_back("");
    }

    // Parse the complete tool call
    std::string complete_call =
        current_text.substr(0, end + eot_token_.length());
    StreamingParseResult result = detect_and_parse(complete_call, tools);

    if (!result.calls.empty()) {
      // Store tool call info for serving layer
      prev_tool_call_arr_[current_tool_id_]["name"] =
          result.calls[0].name.value_or("");
      prev_tool_call_arr_[current_tool_id_]["arguments"] =
          result.calls[0].parameters;
      streamed_args_for_tool_[current_tool_id_] = result.calls[0].parameters;

      // Update tool index
      result.calls[0].tool_index = current_tool_id_;
      current_tool_id_++;
    }

    // Update buffer with remaining text
    buffer_ = current_text.substr(end + eot_token_.length());
    return result;
  }

  // Return normal text before tool call start
  std::string normal_text = current_text.substr(0, start);
  buffer_ = current_text.substr(start);
  return StreamingParseResult(normal_text, {});
}

}  // namespace function_call
}  // namespace xllm

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

#include "deepseekv3_detector.h"

#include <algorithm>
#include <iostream>
#include <nlohmann/json.hpp>
#include <regex>
#include <string_view>

namespace xllm {
namespace function_call {

DeepSeekV3Detector::DeepSeekV3Detector() : BaseFormatDetector() {
  bot_token_ = "<｜tool▁calls▁begin｜>";
  eot_token_ = "<｜tool▁calls▁end｜>";
  func_call_regex_ = "<｜tool▁call▁begin｜>.*?<｜tool▁call▁end｜>";
  func_detail_regex_ =
      "<｜tool▁call▁begin｜>(.*)<｜tool▁sep｜>(.*)\n```json\n(.*)\n```<"
      "｜tool▁call▁end｜>";
  last_arguments_ = "";
  current_tool_id_ = -1;
}

bool DeepSeekV3Detector::has_tool_call(const std::string& text) {
  return text.find(bot_token_) != std::string::npos;
}

std::string_view DeepSeekV3Detector::trim_whitespace(
    std::string_view str) const {
  const char* whitespace = " \t\n\r";

  size_t start = str.find_first_not_of(whitespace);
  if (start == std::string_view::npos) {
    return std::string_view{};
  }

  size_t end = str.find_last_not_of(whitespace);

  return str.substr(start, end - start + 1);
}

std::vector<std::pair<size_t, size_t>>
DeepSeekV3Detector::find_tool_call_ranges(const std::string& text) const {
  std::vector<std::pair<size_t, size_t>> ranges;
  ranges.reserve(4);

  const std::string call_begin = "<｜tool▁call▁begin｜>";
  const std::string call_end = "<｜tool▁call▁end｜>";

  size_t search_pos = 0;
  const size_t call_begin_len = call_begin.length();
  const size_t call_end_len = call_end.length();

  while (search_pos < text.length()) {
    size_t start_pos = text.find(call_begin, search_pos);
    if (start_pos == std::string::npos) {
      break;
    }

    size_t content_start = start_pos + call_begin_len;
    size_t end_pos = text.find(call_end, content_start);
    if (end_pos == std::string::npos) {
      break;
    }

    ranges.emplace_back(content_start, end_pos);
    search_pos = end_pos + call_end_len;
  }

  return ranges;
}

StreamingParseResult DeepSeekV3Detector::detect_and_parse(
    const std::string& text,
    const std::vector<JsonTool>& tools) {
  size_t bot_token_pos = text.find(bot_token_);

  std::string normal_text;
  if (bot_token_pos != std::string::npos) {
    std::string_view normal_text_view(text.data(), bot_token_pos);
    std::string_view trimmed = trim_whitespace(normal_text_view);
    normal_text = std::string(trimmed);
  } else {
    std::string_view trimmed = trim_whitespace(text);
    normal_text = std::string(trimmed);
    return StreamingParseResult(normal_text);
  }

  auto tool_call_ranges = find_tool_call_ranges(text);

  std::vector<ToolCallItem> calls;
  calls.reserve(tool_call_ranges.size());

  for (const auto& range : tool_call_ranges) {
    std::string_view content_view(text.data() + range.first,
                                  range.second - range.first);
    std::string_view trimmed_content = trim_whitespace(content_view);

    if (trimmed_content.empty()) {
      continue;
    }

    try {
      // Parse DeepSeek V3 format: <tool_sep>function_name\n```json\n{args}\n```
      const std::string tool_sep = "<｜tool▁sep｜>";
      size_t sep_pos = trimmed_content.find(tool_sep);
      if (sep_pos == std::string_view::npos) {
        LOG(ERROR) << "Failed to find tool separator in: "
                   << std::string(trimmed_content);
        continue;
      }

      // Extract function name (between tool_sep and first newline)
      size_t name_start = sep_pos + tool_sep.length();
      size_t name_end = trimmed_content.find('\n', name_start);
      if (name_end == std::string_view::npos) {
        LOG(ERROR) << "Failed to find function name end in: "
                   << std::string(trimmed_content);
        continue;
      }

      std::string_view func_name_view =
          trimmed_content.substr(name_start, name_end - name_start);
      std::string_view func_name_trimmed = trim_whitespace(func_name_view);
      std::string func_name(func_name_trimmed);

      // Find JSON block (between ```json\n and \n```)
      const std::string json_start = "```json\n";
      const std::string json_end = "\n```";

      size_t json_start_pos = trimmed_content.find(json_start, name_end);
      if (json_start_pos == std::string_view::npos) {
        LOG(ERROR) << "Failed to find JSON start in: "
                   << std::string(trimmed_content);
        continue;
      }

      size_t json_content_start = json_start_pos + json_start.length();
      size_t json_end_pos = trimmed_content.find(json_end, json_content_start);
      if (json_end_pos == std::string_view::npos) {
        LOG(ERROR) << "Failed to find JSON end in: "
                   << std::string(trimmed_content);
        continue;
      }

      std::string_view json_view = trimmed_content.substr(
          json_content_start, json_end_pos - json_content_start);
      std::string_view json_trimmed = trim_whitespace(json_view);

      // Parse JSON arguments
      nlohmann::json func_args;
      try {
        std::string json_content(json_trimmed);
        func_args = nlohmann::json::parse(json_content);
      } catch (const nlohmann::json::parse_error& e) {
        LOG(ERROR) << "Failed to parse JSON arguments: "
                   << std::string(json_trimmed)
                   << ", JSON parse error: " << e.what();
        continue;
      }

      // Create JSON object for parse_base_json
      nlohmann::json match_json;
      match_json["name"] = func_name;
      match_json["parameters"] = func_args;

      auto parsed_calls = parse_base_json(match_json, tools);
      calls.insert(calls.end(),
                   std::make_move_iterator(parsed_calls.begin()),
                   std::make_move_iterator(parsed_calls.end()));
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to parse tool call: "
                 << std::string(trimmed_content) << ", error: " << e.what();
      continue;
    }
  }

  return StreamingParseResult(std::move(normal_text), std::move(calls));
}

StreamingParseResult DeepSeekV3Detector::parse_streaming_increment(
    const std::string& new_text,
    const std::vector<JsonTool>& tools) {
  buffer_ += new_text;
  std::string current_text = buffer_;

  bool has_tool_call =
      (current_text.find(bot_token_) != std::string::npos ||
       current_text.find("<｜tool▁call▁begin｜>") != std::string::npos);

  if (!has_tool_call) {
    buffer_.clear();
    std::string result_text = new_text;

    std::vector<std::string> end_tokens = {
        eot_token_, "```", "<｜tool▁call▁end｜>"};
    for (const auto& e_token : end_tokens) {
      size_t pos = result_text.find(e_token);
      if (pos != std::string::npos) {
        result_text = result_text.substr(0, pos) +
                      result_text.substr(pos + e_token.length());
      }
    }
    return StreamingParseResult(result_text);
  }

  if (tool_indices_.empty()) {
    tool_indices_ = get_tool_indices(tools);
  }

  std::vector<ToolCallItem> calls;
  try {
    std::regex partial_match_regex(
        R"(<｜tool▁call▁begin｜>(.*)<｜tool▁sep｜>(.*)\n```json\n(.*)\n```.*)");
    std::smatch match;

    if (std::regex_search(current_text, match, partial_match_regex)) {
      std::string func_name = match[2].str();
      // Trim whitespace
      func_name.erase(0, func_name.find_first_not_of(" \t\n\r"));
      func_name.erase(func_name.find_last_not_of(" \t\n\r") + 1);

      std::string func_args_raw = match[3].str();
      // Trim whitespace
      func_args_raw.erase(0, func_args_raw.find_first_not_of(" \t\n\r"));
      func_args_raw.erase(func_args_raw.find_last_not_of(" \t\n\r") + 1);

      if (current_tool_id_ == -1) {
        current_tool_id_ = 0;
        prev_tool_call_arr_.clear();
        streamed_args_for_tool_ = {""};
      }

      while (static_cast<int>(prev_tool_call_arr_.size()) <= current_tool_id_) {
        prev_tool_call_arr_.push_back({});
      }
      while (static_cast<int>(streamed_args_for_tool_.size()) <=
             current_tool_id_) {
        streamed_args_for_tool_.push_back("");
      }

      if (!current_tool_name_sent_) {
        calls.push_back(ToolCallItem(current_tool_id_, func_name, ""));
        current_tool_name_sent_ = true;
        prev_tool_call_arr_[current_tool_id_]["name"] = func_name;
        prev_tool_call_arr_[current_tool_id_]["arguments"] = "{}";
      } else {
        std::string argument_diff;
        if (func_args_raw.length() > last_arguments_.length() &&
            func_args_raw.substr(0, last_arguments_.length()) ==
                last_arguments_) {
          argument_diff = func_args_raw.substr(last_arguments_.length());
        } else {
          argument_diff = func_args_raw;
        }

        if (!argument_diff.empty()) {
          calls.push_back(
              ToolCallItem(current_tool_id_, std::nullopt, argument_diff));
          last_arguments_ += argument_diff;
          streamed_args_for_tool_[current_tool_id_] += argument_diff;
        }

        if (is_complete_json(func_args_raw)) {
          try {
            nlohmann::json parsed_args = nlohmann::json::parse(func_args_raw);
            prev_tool_call_arr_[current_tool_id_]["arguments"] =
                parsed_args.dump();
          } catch (const nlohmann::json::parse_error&) {
            // Ignore parse errors for partial JSON
          }

          std::regex tool_call_end_pattern(
              R"(<｜tool▁call▁begin｜>.*?<｜tool▁call▁end｜>)");
          std::smatch end_match;

          if (std::regex_search(
                  current_text, end_match, tool_call_end_pattern)) {
            buffer_ =
                current_text.substr(end_match.position() + end_match.length());
          } else {
            buffer_.clear();
          }

          StreamingParseResult result("", calls);
          current_tool_id_++;
          last_arguments_.clear();
          current_tool_name_sent_ = false;
          return result;
        }
      }
    }

    return StreamingParseResult("", calls);

  } catch (const std::exception& e) {
    LOG(ERROR) << "Error in parse_streaming_increment: " << e.what();
    return StreamingParseResult(current_text);
  }
}

}  // namespace function_call
}  // namespace xllm

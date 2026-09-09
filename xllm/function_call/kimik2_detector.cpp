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

#include "kimik2_detector.h"

#include <iostream>
#include <regex>
#include <stdexcept>

namespace xllm {
namespace function_call {

KimiK2Detector::KimiK2Detector() : BaseFormatDetector() {
  // Initialize KimiK2 specific tokens
  bot_token_ = "<|tool_calls_section_begin|>";
  eot_token_ = "<|tool_calls_section_end|>";
  tool_call_start_token_ = "<|tool_call_begin|>";
  tool_call_end_token_ = "<|tool_call_end|>";
  tool_call_argument_begin_token_ = "<|tool_call_argument_begin|>";

  // Regex pattern for parsing tool calls with the following format:
  //   <|tool_call_begin|>functions.{func_name}:{index}
  //   <|tool_call_argument_begin|>{json_args}<|tool_call_end|>
  // Note: C++ regex doesn't support named groups, so we use numbered groups:
  //   Group 1: tool_call_id (functions.{func_name}:{index})
  //   Group 2: function_arguments ({json_args})
  std::string pattern =
      R"(<\|tool_call_begin\|>\s*([\w\.]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(\{.*?\})\s*<\|tool_call_end\|>)";

  tool_call_regex_ = std::regex(pattern, std::regex_constants::ECMAScript);

  // Regex pattern for streaming parsing (partial tool calls)
  std::string stream_pattern =
      R"(<\|tool_call_begin\|>\s*([\w\.]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(\{.*))";

  stream_tool_call_portion_regex_ =
      std::regex(stream_pattern, std::regex_constants::ECMAScript);

  last_arguments_ = "";
}

bool KimiK2Detector::has_tool_call(const std::string& text) {
  // Check if the text contains the KimiK2 tool call section begin token
  return text.find(bot_token_) != std::string::npos;
}

StreamingParseResult KimiK2Detector::detect_and_parse(
    const std::string& text,
    const std::vector<JsonTool>& tools) {
  size_t bot_pos = text.find(bot_token_);

  std::string normal_text =
      (bot_pos != std::string::npos) ? text.substr(0, bot_pos) : text;

  if (bot_pos == std::string::npos) {
    return StreamingParseResult(normal_text);
  }

  std::vector<ToolCallItem> calls;

  try {
    std::sregex_iterator iter(
        text.begin() + bot_pos, text.end(), tool_call_regex_);
    std::sregex_iterator end;

    for (; iter != end; ++iter) {
      std::smatch match = *iter;

      if (match.size() >= 3) {
        std::string tool_call_id = match[1].str();
        std::string function_arguments = match[2].str();

        std::string function_name = extract_function_name(tool_call_id);
        int32_t function_index = extract_function_index(tool_call_id);

        calls.emplace_back(function_index,     // Use the call index in the
                                               // response, not tool position
                           function_name,      // Function name
                           function_arguments  // JSON parameters
        );
      }
    }

    return StreamingParseResult(normal_text, calls);

  } catch (const std::exception& e) {
    LOG(ERROR) << "Error in KimiK2 detect_and_parse: " << e.what();
    // Return the normal text if parsing fails
    return StreamingParseResult(normal_text);
  }
}

std::string KimiK2Detector::extract_function_name(
    const std::string& tool_call_id) const {
  // tool_call_id format: functions.{func_name}:{index}
  // Example: functions.get_weather:0

  try {
    // Find the position of "functions."
    size_t functions_pos = tool_call_id.find("functions.");
    if (functions_pos == std::string::npos) {
      LOG(WARNING)
          << "Invalid tool_call_id format, missing 'functions.' prefix: "
          << tool_call_id;
      return "";
    }

    // Skip "functions." (10 characters)
    size_t start_pos = functions_pos + 10;

    // Find the position of the last colon
    size_t colon_pos = tool_call_id.find_last_of(':');
    if (colon_pos == std::string::npos || colon_pos <= start_pos) {
      LOG(WARNING) << "Invalid tool_call_id format, missing ':' separator: "
                   << tool_call_id;
      return "";
    }

    // Extract function name between "functions." and ":"
    return tool_call_id.substr(start_pos, colon_pos - start_pos);

  } catch (const std::exception& e) {
    LOG(ERROR) << "Error extracting function name from tool_call_id: "
               << tool_call_id << ", error: " << e.what();
    return "";
  }
}

int32_t KimiK2Detector::extract_function_index(
    const std::string& tool_call_id) const {
  // tool_call_id format: functions.{func_name}:{index}
  // Example: functions.get_weather:0

  try {
    // Find the position of the last colon
    size_t colon_pos = tool_call_id.find_last_of(':');
    if (colon_pos == std::string::npos) {
      LOG(WARNING) << "Invalid tool_call_id format, missing ':' separator: "
                   << tool_call_id;
      return 0;
    }

    // Extract index string after the colon
    std::string index_str = tool_call_id.substr(colon_pos + 1);

    // Convert to integer
    return std::stoi(index_str);

  } catch (const std::exception& e) {
    LOG(ERROR) << "Error extracting function index from tool_call_id: "
               << tool_call_id << ", error: " << e.what();
    return 0;
  }
}

StreamingParseResult KimiK2Detector::parse_streaming_increment(
    const std::string& new_text,
    const std::vector<JsonTool>& tools) {
  buffer_ += new_text;
  std::string current_text = buffer_;

  bool has_tool_call_marker =
      (current_text.find(bot_token_) != std::string::npos ||
       current_text.find(tool_call_start_token_) != std::string::npos);

  if (!has_tool_call_marker) {
    buffer_.clear();
    std::string cleaned_text = new_text;
    std::vector<std::string> end_tokens = {eot_token_, tool_call_end_token_};
    for (const auto& e_token : end_tokens) {
      size_t pos = cleaned_text.find(e_token);
      if (pos != std::string::npos) {
        cleaned_text.erase(pos, e_token.length());
      }
    }
    return StreamingParseResult(cleaned_text, {});
  }

  if (tool_indices_.empty()) {
    tool_indices_ = get_tool_indices(tools);
  }

  std::vector<ToolCallItem> calls;

  try {
    std::smatch match;
    if (std::regex_search(
            current_text, match, stream_tool_call_portion_regex_)) {
      std::string function_id = match[1].str();
      std::string function_args = match[2].str();

      std::string function_name = extract_function_name(function_id);

      if (current_tool_id_ == -1) {
        current_tool_id_ = 0;
        prev_tool_call_arr_.clear();
        streamed_args_for_tool_.clear();
        streamed_args_for_tool_.push_back("");
      }

      while (static_cast<int>(prev_tool_call_arr_.size()) <= current_tool_id_) {
        prev_tool_call_arr_.push_back(
            std::unordered_map<std::string, std::string>());
      }
      while (static_cast<int>(streamed_args_for_tool_.size()) <=
             current_tool_id_) {
        streamed_args_for_tool_.push_back("");
      }

      if (!current_tool_name_sent_) {
        calls.emplace_back(current_tool_id_, function_name, "");
        current_tool_name_sent_ = true;
        prev_tool_call_arr_[current_tool_id_]["name"] = function_name;
        prev_tool_call_arr_[current_tool_id_]["arguments"] = "{}";
      } else {
        std::string argument_diff;
        if (function_args.length() > last_arguments_.length() &&
            function_args.substr(0, last_arguments_.length()) ==
                last_arguments_) {
          argument_diff = function_args.substr(last_arguments_.length());
        } else {
          argument_diff = function_args;
        }

        size_t end_pos = argument_diff.find(tool_call_end_token_);
        if (end_pos != std::string::npos) {
          argument_diff = argument_diff.substr(0, end_pos);
        }

        if (!argument_diff.empty()) {
          calls.emplace_back(current_tool_id_, std::nullopt, argument_diff);
          last_arguments_ += argument_diff;
          streamed_args_for_tool_[current_tool_id_] += argument_diff;
        }

        std::string parsed_args = function_args;
        end_pos = parsed_args.find(tool_call_end_token_);
        if (end_pos != std::string::npos) {
          parsed_args = parsed_args.substr(0, end_pos);
        }

        if (is_complete_json(parsed_args)) {
          try {
            auto parsed_json = nlohmann::json::parse(parsed_args);
            prev_tool_call_arr_[current_tool_id_]["arguments"] =
                parsed_json.dump();
          } catch (const std::exception& e) {
            LOG(ERROR) << "Failed to parse JSON arguments: " << e.what();
          }

          std::regex tool_call_end_pattern(
              R"(<\|tool_call_begin\|>.*?<\|tool_call_end\|>)",
              std::regex_constants::ECMAScript);
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
    LOG(ERROR) << "Error in KimiK2 parse_streaming_increment: " << e.what();
    return StreamingParseResult(current_text, {});
  }
}

}  // namespace function_call
}  // namespace xllm

/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <string>
#include <string_view>
#include <utility>

#include "base_format_detector.h"

namespace xllm {
namespace function_call {

enum class StreamState {
  INIT,           // Initial state
  BETWEEN,        // Between key-value pairs
  IN_KEY,         // Reading key content
  WAITING_VALUE,  // Waiting for value tag
  IN_VALUE        // Reading value content
};

/**
 * Detector for GLM-4.7 and GLM-5 models function call format.
 *
 * Format Structure (compact, no newlines):
 * ```
 * <tool_call>function_name<arg_key>param1</arg_key><arg_value>value1</arg_value><arg_key>param2</arg_key><arg_value>value2</arg_value></tool_call>
 * ```
 *
 * Example:
 * ```
 * <tool_call>get_weather<arg_key>city</arg_key><arg_value>北京</arg_value><arg_key>date</arg_key><arg_value>2024-06-27</arg_value></tool_call>
 * ```
 */
class Glm47Detector : public BaseFormatDetector {
 public:
  Glm47Detector();

  virtual ~Glm47Detector() = default;

  bool has_tool_call(const std::string& text) override;

  StreamingParseResult detect_and_parse(
      const std::string& text,
      const std::vector<JsonTool>& tools) override;

  StreamingParseResult parse_streaming_increment(
      const std::string& new_text,
      const std::vector<JsonTool>& tools) override;

 private:
  // String-based parsing helpers (replaces regex to avoid stack overflow)
  std::vector<std::pair<size_t, size_t>> find_tool_call_ranges(
      const std::string& text) const;

  std::string extract_normal_text(
      const std::string& text,
      const std::vector<std::pair<size_t, size_t>>& ranges) const;

  std::pair<std::string, std::string> parse_tool_call_content(
      const std::string& content) const;

  std::vector<std::pair<std::string, std::string>> extract_argument_pairs(
      const std::string& args_raw) const;

  StreamState stream_state_;
  std::string current_key_;
  std::string current_value_;
  std::string xml_tag_buffer_;
  bool is_first_param_;
  bool value_started_;
  std::string cached_value_type_;
  std::string utf8_buffer_;  // Buffer for incomplete UTF-8 sequences
  std::string last_arguments_;
  size_t streamed_raw_length_;

  std::string trim_whitespace(std::string_view str) const;

  std::string get_argument_type(const std::string& func_name,
                                const std::string& arg_key,
                                const std::vector<JsonTool>& tools) const;

  nlohmann::json convert_to_number(const std::string& value) const;

  std::pair<nlohmann::json, bool> parse_arguments(
      const std::string& json_value,
      const std::string& arg_type) const;

  std::unordered_map<std::string, nlohmann::json> parse_argument_pairs(
      const std::vector<std::pair<std::string, std::string>>& pairs,
      const std::string& func_name,
      const std::vector<JsonTool>& tools) const;

  std::string get_value_type(const std::string& func_name,
                             const std::string& key,
                             const std::vector<JsonTool>& tools) const;

  std::string format_value_complete(const std::string& value,
                                    const std::string& value_type) const;

  std::string process_xml_to_json_streaming(const std::string& raw_increment,
                                            const std::string& func_name,
                                            const std::vector<JsonTool>& tools);

  void reset_streaming_state();

  // Helper to split string into complete UTF-8 part and incomplete tail
  // Returns: {complete_utf8_string, incomplete_tail}
  std::pair<std::string, std::string> split_incomplete_utf8(
      const std::string& str) const;
};

}  // namespace function_call
}  // namespace xllm

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

#pragma once

#include <regex>
#include <string>
#include <string_view>

#include "base_format_detector.h"

namespace xllm {
namespace function_call {

/**
 * Detector for GLM-4.5 model function call format.
 *
 * Format Structure:
 * ```
 * <tool_call>function_name
 * <arg_key>param1</arg_key>
 * <arg_value>value1</arg_value>
 * <arg_key>param2</arg_key>
 * <arg_value>value2</arg_value>
 * </tool_call>
 * ```
 *
 * Example:
 * ```
 * <tool_call>get_weather
 * <arg_key>city</arg_key>
 * <arg_value>北京</arg_value>
 * <arg_key>date</arg_key>
 * <arg_value>2024-06-27</arg_value>
 * </tool_call>
 * ```
 */
class Glm45Detector : public BaseFormatDetector {
 public:
  Glm45Detector();

  virtual ~Glm45Detector() = default;

  bool has_tool_call(const std::string& text) override;

  StreamingParseResult detect_and_parse(
      const std::string& text,
      const std::vector<JsonTool>& tools) override;

  StreamingParseResult parse_streaming_increment(
      const std::string& new_text,
      const std::vector<JsonTool>& tools) override;

 private:
  std::regex func_call_regex_;
  std::regex func_detail_regex_;
  std::regex func_arg_regex_;

  std::string trim_whitespace(std::string_view str) const;
};

}  // namespace function_call
}  // namespace xllm

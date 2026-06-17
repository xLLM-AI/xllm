/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <regex>
#include <string>
#include <string_view>
#include <utility>

#include "base_format_detector.h"

namespace xllm {
namespace function_call {

class DeepSeekV32Detector : public BaseFormatDetector {
 public:
  DeepSeekV32Detector();
  DeepSeekV32Detector(const std::string& tool_calls_start_token,
                      const std::string& tool_calls_end_token);

  virtual ~DeepSeekV32Detector() = default;

  bool has_tool_call(const std::string& text) override;

  StreamingParseResult detect_and_parse(
      const std::string& text,
      const std::vector<JsonTool>& tools) override;

  StreamingParseResult parse_streaming_increment(
      const std::string& new_text,
      const std::vector<JsonTool>& tools) override;

 private:
  std::regex function_calls_regex_;
  std::regex invoke_regex_;
  // For streaming: matches invoke with optional closing tag (group 3 =
  // "</｜DSML｜invoke>" or "")
  std::regex streaming_invoke_regex_;
  std::regex parameter_regex_;
  std::regex partial_parameter_regex_;

  std::string invoke_end_token_;
  std::vector<std::string> prefix_parameter_end_call_;
  std::string utf8_buffer_;

  std::string trim_whitespace(std::string_view str) const;

  std::pair<std::string, std::string> split_incomplete_utf8(
      const std::string& str) const;

  std::unordered_map<std::string, nlohmann::json> parse_parameters_from_xml(
      const std::string& invoke_content,
      bool allow_partial = false) const;

  std::vector<ToolCallItem> parse_json_tool_calls(
      const std::string& text,
      const std::vector<JsonTool>& tools);
};

class DeepSeekV4Detector final : public DeepSeekV32Detector {
 public:
  DeepSeekV4Detector();
};

}  // namespace function_call
}  // namespace xllm

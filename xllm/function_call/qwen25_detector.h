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

class Qwen25Detector : public BaseFormatDetector {
 public:
  Qwen25Detector();

  virtual ~Qwen25Detector() = default;

 private:
  std::string normal_text_buffer_;

  std::regex tool_call_regex_;

  std::string_view trim_whitespace(std::string_view str) const;

  std::vector<std::pair<size_t, size_t>> find_tool_call_ranges(
      const std::string& text) const;

 public:
  bool has_tool_call(const std::string& text) override;

  StreamingParseResult detect_and_parse(
      const std::string& text,
      const std::vector<JsonTool>& tools) override;

  // Streaming incremental parsing for Qwen 2.5/3 tool calls
  // parse_streaming_increment
  StreamingParseResult parse_streaming_increment(
      const std::string& new_text,
      const std::vector<JsonTool>& tools) override;
};

}  // namespace function_call
}  // namespace xllm

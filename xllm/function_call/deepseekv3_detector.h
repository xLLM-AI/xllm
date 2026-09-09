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

#include "base_format_detector.h"

namespace xllm {
namespace function_call {

class DeepSeekV3Detector : public BaseFormatDetector {
 public:
  DeepSeekV3Detector();

  virtual ~DeepSeekV3Detector() = default;

  bool has_tool_call(const std::string& text) override;

  StreamingParseResult detect_and_parse(
      const std::string& text,
      const std::vector<JsonTool>& tools) override;

  StreamingParseResult parse_streaming_increment(
      const std::string& new_text,
      const std::vector<JsonTool>& tools) override;

 private:
  std::string func_call_regex_;
  std::string func_detail_regex_;
  std::string last_arguments_;

  std::string_view trim_whitespace(std::string_view str) const;
  std::vector<std::pair<size_t, size_t>> find_tool_call_ranges(
      const std::string& text) const;
};

}  // namespace function_call
}  // namespace xllm

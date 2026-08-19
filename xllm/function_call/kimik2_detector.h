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

/**
 * Detector for Kimi K2 model function call format.
 *
 * Format Structure:
 * ```
 * <|tool_calls_section_begin|>
 * <|tool_call_begin|>functions.{func_name}:{index}
 * <|tool_call_argument_begin|>{json_args}<|tool_call_end|>
 * <|tool_calls_section_end|>
 * ```
 *
 * Reference:
 * https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/docs/tool_call_guidance.md
 */
class KimiK2Detector : public BaseFormatDetector {
 public:
  KimiK2Detector();

  virtual ~KimiK2Detector() = default;

 private:
  std::string tool_call_start_token_;
  std::string tool_call_end_token_;
  std::string tool_call_argument_begin_token_;

  std::regex tool_call_regex_;
  std::regex stream_tool_call_portion_regex_;

  std::string last_arguments_;

 public:
  bool has_tool_call(const std::string& text) override;

  StreamingParseResult detect_and_parse(
      const std::string& text,
      const std::vector<JsonTool>& tools) override;

  StreamingParseResult parse_streaming_increment(
      const std::string& new_text,
      const std::vector<JsonTool>& tools) override;

 private:
  std::string extract_function_name(const std::string& tool_call_id) const;

  int32_t extract_function_index(const std::string& tool_call_id) const;
};

}  // namespace function_call
}  // namespace xllm

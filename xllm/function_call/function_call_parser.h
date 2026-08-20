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

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "base_format_detector.h"
#include "core_types.h"

namespace xllm {
namespace function_call {

class FunctionCallParser {
 public:
  FunctionCallParser(const std::vector<JsonTool>& tools,
                     const std::string& tool_call_parser);

  ~FunctionCallParser() = default;

  FunctionCallParser(const FunctionCallParser&) = delete;
  FunctionCallParser& operator=(const FunctionCallParser&) = delete;

  bool has_tool_call(const std::string& text) const;

  std::tuple<std::string, std::vector<ToolCallItem>> parse_non_stream(
      const std::string& full_text);

  // Streaming incremental parsing method
  StreamingParseResult parse_streaming_increment(const std::string& new_text);

  // StructuralTagResponseFormat get_structure_tag();

  // std::tuple<std::string, std::any> get_structure_constraint(const
  // std::string& tool_choice);

  BaseFormatDetector* get_detector() const { return detector_.get(); }

  static std::string get_parser_auto(const std::string& parser,
                                     const std::string& model_type);

 private:
  std::unique_ptr<BaseFormatDetector> create_detector(
      const std::string& tool_call_parser);
  std::unique_ptr<BaseFormatDetector> detector_;
  std::vector<JsonTool> tools_;
};

namespace utils {

std::vector<ToolCallItem> parse_function_calls(
    const std::string& text,
    const std::vector<JsonTool>& tools,
    const std::string& parser_type = "qwen25");

bool has_function_calls(const std::string& text,
                        const std::string& parser_type = "qwen25");

// Streaming parsing utility function
StreamingParseResult parse_streaming_increment(
    const std::string& new_text,
    const std::vector<JsonTool>& tools,
    const std::string& parser_type = "qwen25");

std::string generate_tool_call_id();
}  // namespace utils

}  // namespace function_call
}  // namespace xllm

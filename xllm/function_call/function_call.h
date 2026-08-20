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

#include "base_format_detector.h"
#include "core_types.h"
#include "deepseekv3_detector.h"
#include "function_call_parser.h"
#include "glm45_detector.h"
#include "kimik2_detector.h"
#include "qwen25_detector.h"
#include "qwen3_coder_detector.h"

namespace xllm {
namespace function_call {

inline std::vector<ToolCallItem> parse(const std::string& text,
                                       const std::vector<JsonTool>& tools,
                                       const std::string& format = "qwen25") {
  return utils::parse_function_calls(text, tools, format);
}

inline bool has_calls(const std::string& text,
                      const std::string& format = "qwen25") {
  return utils::has_function_calls(text, format);
}

}  // namespace function_call
}  // namespace xllm

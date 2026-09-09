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

#include <functional>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

#include "core/common/types.h"

namespace xllm {
namespace function_call {

using JsonFunction = xllm::JsonFunction;
using JsonTool = xllm::JsonTool;

struct ToolCallItem {
  int32_t tool_index;
  std::optional<std::string> name;
  std::string parameters;  // JSON string

  ToolCallItem() : tool_index(-1), parameters("") {}

  ToolCallItem(int32_t index,
               const std::optional<std::string>& func_name,
               const std::string& params)
      : tool_index(index), name(func_name), parameters(params) {}
};

struct StreamingParseResult {
  std::string normal_text;
  std::vector<ToolCallItem> calls;

  StreamingParseResult() = default;

  explicit StreamingParseResult(std::string text)
      : normal_text(std::move(text)) {}

  explicit StreamingParseResult(std::vector<ToolCallItem> tool_calls)
      : calls(std::move(tool_calls)) {}

  StreamingParseResult(std::string text, std::vector<ToolCallItem> tool_calls)
      : normal_text(std::move(text)), calls(std::move(tool_calls)) {}

  bool has_calls() const { return !calls.empty(); }

  void clear() {
    normal_text.clear();
    calls.clear();
  }
};

struct StructureInfo {
  std::string begin;
  std::string end;
  std::string trigger;

  StructureInfo() = default;

  StructureInfo(const std::string& begin_str,
                const std::string& end_str,
                const std::string& trigger_str)
      : begin(begin_str), end(end_str), trigger(trigger_str) {}
};

using GetInfoFunc = std::function<StructureInfo(const std::string&)>;

}  // namespace function_call
}  // namespace xllm

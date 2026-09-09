/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "stream_output_parser.h"

namespace xllm {
StreamOutputParser::StreamOutputParser(
    const std::vector<function_call::JsonTool>& tools,
    const std::string& tool_call_parser_format,
    const std::string& reasoning_parser_format,
    bool force_reasoning)
    : tools_(tools),
      tool_call_parser_format_(tool_call_parser_format),
      reasoning_parser_format_(reasoning_parser_format),
      force_reasoning_(force_reasoning) {
  sequence_parsers_.resize(1);
  if (is_tool_call()) {
    sequence_parsers_[0].tool_call_parser =
        std::make_unique<function_call::FunctionCallParser>(
            tools_, tool_call_parser_format_);
  }
  if (is_reasoning()) {
    sequence_parsers_[0].reasoning_parser_ = std::make_unique<ReasoningParser>(
        reasoning_parser_format_, true, force_reasoning_);
  }
}

bool StreamOutputParser::is_tool_call() {
  return !tools_.empty() && !tool_call_parser_format_.empty();
}

bool StreamOutputParser::is_reasoning() {
  return !reasoning_parser_format_.empty();
}

void StreamOutputParser::check_resize_for_index(size_t index) {
  if (index >= sequence_parsers_.size()) {
    sequence_parsers_.resize(index + 1);
  }
}

function_call::FunctionCallParser* StreamOutputParser::get_tool_call_parser(
    size_t index) {
  if (!is_tool_call()) {
    return nullptr;
  }

  check_resize_for_index(index);

  if (!sequence_parsers_[index].tool_call_parser) {
    sequence_parsers_[index].tool_call_parser =
        std::make_unique<function_call::FunctionCallParser>(
            tools_, tool_call_parser_format_);
  }

  return sequence_parsers_[index].tool_call_parser.get();
}

ReasoningParser* StreamOutputParser::get_reasoning_parser(size_t index) {
  if (!is_reasoning()) {
    return nullptr;
  }

  check_resize_for_index(index);

  if (!sequence_parsers_[index].reasoning_parser_) {
    sequence_parsers_[index].reasoning_parser_ =
        std::make_unique<ReasoningParser>(
            reasoning_parser_format_, true, force_reasoning_);
  }

  return sequence_parsers_[index].reasoning_parser_.get();
}
}  // namespace xllm

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

#pragma once

#include "function_call/function_call.h"
#include "parser/reasoning_parser.h"

namespace xllm {

struct SequenceParser {
  std::unique_ptr<function_call::FunctionCallParser> tool_call_parser;
  bool has_tool_call = false;
  std::unique_ptr<ReasoningParser> reasoning_parser_;
};

class StreamOutputParser {
 public:
  StreamOutputParser(const std::vector<function_call::JsonTool>& tools,
                     const std::string& tool_call_parser_format,
                     const std::string& reasoning_parser_format,
                     bool force_reasoning = false);

  ~StreamOutputParser() = default;

  bool is_tool_call();

  bool is_reasoning();

  void check_resize_for_index(size_t index);

  function_call::FunctionCallParser* get_tool_call_parser(size_t index);

  ReasoningParser* get_reasoning_parser(size_t index);

  bool get_has_tool_call(size_t index) {
    check_resize_for_index(index);
    return sequence_parsers_[index].has_tool_call;
  }

  void set_has_tool_call(size_t index, bool has_tool_call) {
    check_resize_for_index(index);
    sequence_parsers_[index].has_tool_call = has_tool_call;
  }

 private:
  // candidate tools of requests
  std::vector<function_call::JsonTool> tools_;
  // list of parsers for each sequence
  std::vector<SequenceParser> sequence_parsers_;
  std::string tool_call_parser_format_;
  std::string reasoning_parser_format_;
  bool force_reasoning_;
};

}  // namespace xllm

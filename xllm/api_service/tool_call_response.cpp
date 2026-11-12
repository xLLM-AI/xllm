/* Copyright 2026 The xLLM Authors.

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

#include "api_service/tool_call_response.h"

#include <glog/logging.h>

#include <exception>
#include <utility>

#include "api_service/stream_output_parser.h"
#include "function_call/function_call_parser.h"

namespace xllm::api_service {

bool check_for_unstreamed_tool_args(
    std::shared_ptr<StreamOutputParser> stream_parser,
    size_t index,
    SendFunc send_func) {
  auto* parser = stream_parser->get_tool_call_parser(index);
  if (!parser) {
    return true;
  }

  auto* detector = parser->get_detector();
  if (!detector) {
    return true;
  }

  if (!detector->prev_tool_call_arr_.empty() &&
      !detector->streamed_args_for_tool_.empty()) {
    size_t tool_index = detector->prev_tool_call_arr_.size() - 1;
    if (tool_index < detector->streamed_args_for_tool_.size()) {
      const auto& expected_args = detector->prev_tool_call_arr_[tool_index];
      const std::string& actual_args =
          detector->streamed_args_for_tool_[tool_index];

      if (expected_args.find("arguments") != expected_args.end()) {
        const std::string& expected_call = expected_args.at("arguments");

        if (expected_call.length() > actual_args.length()) {
          std::string remaining_call =
              expected_call.substr(actual_args.length());

          if (!remaining_call.empty()) {
            return send_func(remaining_call, static_cast<int32_t>(tool_index));
          }
        }
      }
    }
  }

  return true;
}

ToolCallResult process_tool_calls(std::string text,
                                  const std::vector<xllm::JsonTool>& tools,
                                  const std::string& parser_format,
                                  std::string finish_reason,
                                  google::protobuf::Arena* arena) {
  ToolCallResult result;

  function_call::FunctionCallParser parser(tools, parser_format);

  if (!parser.has_tool_call(text)) {
    result.text = std::move(text);
    result.finish_reason = std::move(finish_reason);
    return result;
  }

  result.text = text;
  result.finish_reason = finish_reason;

  try {
    auto [parsed_text, call_info_list] = parser.parse_non_stream(text);
    result.text = std::move(parsed_text);

    if (call_info_list.empty()) {
      return result;
    }

    if (finish_reason == "stop") {
      result.finish_reason = "tool_calls";
    }

    google::protobuf::RepeatedPtrField<proto::ToolCall> tool_calls;

    for (const auto& call_info : call_info_list) {
      proto::ToolCall* tool_call =
          arena ? google::protobuf::Arena::CreateMessage<proto::ToolCall>(arena)
                : new proto::ToolCall();

      tool_call->set_id(function_call::utils::generate_tool_call_id());
      tool_call->set_type("function");

      auto* function = tool_call->mutable_function();
      if (call_info.name) {
        function->set_name(*call_info.name);
      }
      function->set_arguments(call_info.parameters);

      tool_calls.AddAllocated(tool_call);
    }

    result.tool_calls = std::move(tool_calls);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Tool call parsing error: " << e.what();
  }

  return result;
}

}  // namespace xllm::api_service

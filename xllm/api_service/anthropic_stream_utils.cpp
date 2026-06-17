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

#include "api_service/anthropic_stream_utils.h"

namespace xllm {
namespace api_service {

std::string convert_finish_reason_to_anthropic(
    const std::string& finish_reason) {
  if (finish_reason == "stop") {
    return "end_turn";
  }
  if (finish_reason == "length") {
    return "max_tokens";
  }
  if (finish_reason == "function_call" || finish_reason == "tool_calls") {
    return "tool_use";
  }
  return "end_turn";
}

std::string get_stream_stop_reason(bool finished,
                                   bool has_tool_call,
                                   const std::string& finish_reason) {
  if (!finished) {
    return "stop";
  }
  if (has_tool_call) {
    return "tool_use";
  }
  return convert_finish_reason_to_anthropic(finish_reason);
}

std::optional<proto::AnthropicStreamEvent> make_input_json_delta_event(
    int32_t content_block_index,
    const std::string& partial_json) {
  if (partial_json.empty()) {
    return std::nullopt;
  }

  proto::AnthropicStreamEvent chunk;
  chunk.set_index(content_block_index);
  chunk.set_type("content_block_delta");
  auto* delta = chunk.mutable_delta();
  delta->set_type("input_json_delta");
  delta->set_partial_json(partial_json);
  return chunk;
}

}  // namespace api_service
}  // namespace xllm

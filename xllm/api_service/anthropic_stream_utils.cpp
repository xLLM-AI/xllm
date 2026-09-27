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

#include "api_service/anthropic_stream_utils.h"

#include <google/protobuf/util/json_util.h>

namespace xllm {
namespace api_service {

std::optional<std::string> convert_anthropic_response(
    const proto::ChatResponse& chat_response,
    const std::string& thinking_signature,
    proto::AnthropicMessagesResponse& anthropic_response,
    bool include_text) {
  anthropic_response.Clear();

  // Set basic fields
  anthropic_response.set_id(chat_response.id());
  anthropic_response.set_type("message");
  anthropic_response.set_role("assistant");
  anthropic_response.set_model(chat_response.model());

  // Set usage
  if (chat_response.has_usage()) {
    auto* usage = anthropic_response.mutable_usage();
    usage->set_input_tokens(chat_response.usage().prompt_tokens());
    usage->set_output_tokens(chat_response.usage().completion_tokens());
  }

  // Process first choice
  if (chat_response.choices_size() > 0) {
    const auto& choice = chat_response.choices(0);

    // set stop_reason
    if (choice.has_finish_reason()) {
      anthropic_response.set_stop_reason(
          std::move(api_service::convert_finish_reason_to_anthropic(
              choice.finish_reason())));
    }

    const auto& message = choice.message();
    if (!message.reasoning_content().empty()) {
      auto* block = anthropic_response.add_content();
      block->set_type("thinking");
      block->set_thinking(message.reasoning_content());
      block->set_signature(thinking_signature);
    }
    if (include_text && !message.content().empty()) {
      auto* block = anthropic_response.add_content();
      block->set_type("text");
      block->set_text(message.content());
    }

    // Add tool_use blocks for each tool call
    if (choice.has_message()) {
      const auto& message = choice.message();
      for (const auto& tool_call : message.tool_calls()) {
        auto* tool_block = anthropic_response.add_content();
        tool_block->set_type("tool_use");
        tool_block->set_id(tool_call.id());
        tool_block->set_name(tool_call.function().name());

        // Parse arguments JSON string to Struct
        auto status = google::protobuf::util::JsonStringToMessage(
            tool_call.function().arguments().empty()
                ? "{}"
                : tool_call.function().arguments(),
            tool_block->mutable_input());
        if (!status.ok()) {
          return "Model returned invalid tool arguments";
        }
      }
    }
  }

  return std::nullopt;
}

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
                                   const std::string& finish_reason,
                                   bool named_tool_choice) {
  if (!finished) {
    return "end_turn";
  }
  if (has_tool_call && !named_tool_choice) {
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

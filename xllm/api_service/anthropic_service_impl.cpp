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

#include "api_service/anthropic_service_impl.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>

#include "api_service/anthropic_request_utils.h"
#include "api_service/anthropic_stream_utils.h"
#include "api_service/stream_output_parser.h"
#include "api_service/utils.h"
#include "core/common/types.h"
#include "core/distributed_runtime/llm_master.h"
#include "core/framework/request/request_params.h"
#include "core/util/uuid.h"
#include "function_call/function_call.h"

namespace xllm {
namespace {

LLMMaster* check_master(LLMMaster* master) {
  CHECK(master != nullptr);
  return master;
}

struct FunctionCallInfo {
  std::string id = "";
  std::string name = "";
  std::string arguments = "";
};

struct ContentBlockInfo {
  std::string normal_text = "";
  std::vector<FunctionCallInfo> function_calls;
};

// for non-streaming,
// generate chat response first and then convert to anthropic protobuf response.
void generate_chat_response(proto::ChatResponse& response,
                            const std::string& request_id,
                            const std::string& model,
                            const RequestOutput& req_output,
                            const std::string& tool_call_parser_format = "",
                            const std::string& reasoning_parser_format = "",
                            bool is_force_reasoning = false,
                            const std::vector<xllm::JsonTool>& tools = {},
                            bool named_tool_choice = false,
                            bool required_tool_choice = false) {
  response.set_object("chat.completion");
  response.set_id(request_id);
  response.set_model(model);

  response.mutable_choices()->Reserve(req_output.outputs.size());
  for (const auto& output : req_output.outputs) {
    auto* choice = response.add_choices();
    choice->set_index(output.index);
    auto* message = choice->mutable_message();
    message->set_role("assistant");

    // 1) handle reasoning output
    std::string cur_text = output.text;
    if (!reasoning_parser_format.empty()) {
      auto reasoning_parser = std::make_unique<ReasoningParser>(
          reasoning_parser_format, false, is_force_reasoning);
      auto result = reasoning_parser->parse_non_stream(cur_text);
      if (result.normal_text.has_value()) {
        cur_text = result.normal_text.value();
      } else {
        cur_text = "";
      }
      // set reasoning output
      if (result.reasoning_text.has_value()) {
        message->set_reasoning_content(result.reasoning_text.value());
      }
    }

    // 2) handle tool call output
    if (!tools.empty() && !tool_call_parser_format.empty() &&
        !cur_text.empty()) {
      auto* arena = response.GetArena();
      auto result =
          api_service::process_tool_calls(cur_text,
                                          tools,
                                          tool_call_parser_format,
                                          output.finish_reason.value_or(""),
                                          arena);

      // set tool call output
      message->mutable_content()->swap(result.text);
      // set tool calls
      if (result.tool_calls) {
        auto& source_tool_calls = *result.tool_calls;
        message->mutable_tool_calls()->Swap(&source_tool_calls);
      }
      // set finish reason
      if (!result.finish_reason.empty()) {
        choice->mutable_finish_reason()->swap(result.finish_reason);
      }
      if (named_tool_choice ||
          (required_tool_choice && output.finish_reason != "stop")) {
        choice->set_finish_reason(output.finish_reason.value_or("stop"));
      }
    } else {
      // 3) handle text output
      message->set_content(cur_text);
      if (output.finish_reason.has_value()) {
        choice->set_finish_reason(output.finish_reason.value());
      }
    }
  }

  // set usage
  if (req_output.usage.has_value()) {
    const auto& usage = req_output.usage.value();
    auto* proto_usage = response.mutable_usage();
    proto_usage->set_prompt_tokens(usage.num_prompt_tokens);
    proto_usage->set_completion_tokens(usage.num_generated_tokens);
    proto_usage->set_total_tokens(usage.num_total_tokens);
    proto_usage->mutable_prompt_tokens_details()->set_cached_tokens(
        usage.num_cached_tokens);
  }
}

// for non-streaming,
// convert chat response to anthropic protobuf response
template <typename AnthropicCall>
bool send_result_to_client(std::shared_ptr<AnthropicCall> call,
                           const proto::ChatResponse& chat_response,
                           bool include_text) {
  auto& response = call->response();
  std::optional<std::string> error = api_service::convert_anthropic_response(
      chat_response, ShortUUID().random(), response, include_text);
  if (error.has_value()) {
    return call->finish_with_error(StatusCode::UNKNOWN, error.value());
  }
  return call->write_and_finish(response);
}

bool stop_content_block(const std::shared_ptr<AnthropicCall>& call,
                        const std::string& type,
                        int32_t index) {
  if (type == "thinking") {
    proto::AnthropicStreamEvent signature;
    signature.set_type("content_block_delta");
    signature.set_index(index);
    signature.mutable_delta()->set_type("signature_delta");
    signature.mutable_delta()->set_signature(ShortUUID().random());
    if (!call->write(signature.type(), signature)) {
      return false;
    }
  }
  proto::AnthropicStreamEvent stop;
  stop.set_type("content_block_stop");
  stop.set_index(index);
  return call->write(stop.type(), stop);
}

// create a new content block like
// `<content_block_start>` ... `<content_block_stop>`
bool start_new_content_block(std::shared_ptr<AnthropicCall> call,
                             std::string& last_content_block_type,
                             const std::string& curr_content_block_type,
                             const ContentBlockInfo& content_block_info,
                             int32_t& content_block_index) {
  // if not the first content block,
  // we need to create a content_block_stop
  if (!last_content_block_type.empty() &&
      !stop_content_block(call, last_content_block_type, content_block_index)) {
    return false;
  }

  // update last_content_block_type
  last_content_block_type = curr_content_block_type;

  // create a new content block
  proto::AnthropicStreamEvent content_start_event;
  content_start_event.set_type("content_block_start");
  content_start_event.set_index(++content_block_index);
  auto* content_block = content_start_event.mutable_content_block();
  content_block->set_type(curr_content_block_type);
  if (curr_content_block_type == "text") {
    content_block->set_text("");
  } else if (curr_content_block_type == "thinking") {
    content_block->set_thinking("");
  } else if (curr_content_block_type == "tool_use") {
    content_block->set_id(content_block_info.function_calls[0].id);
    content_block->set_name(content_block_info.function_calls[0].name);
    content_block->mutable_input();
  } else {
    LOG(FATAL) << "Unknown content block type: " << curr_content_block_type;
  }
  if (!call->write(content_start_event.type(), content_start_event)) {
    LOG(ERROR) << "Failed to send content_block_start event";
    return false;
  }

  return true;
}

// send a content block delta content back
bool send_content_block_delta(std::shared_ptr<AnthropicCall> call,
                              std::string& last_content_block_type,
                              const std::string& curr_content_block_type,
                              const std::string& delta_type,
                              const ContentBlockInfo& content_block_info,
                              int32_t& content_block_index) {
  bool is_tool_use_delta = delta_type == "tool_use_delta";
  if (is_tool_use_delta && content_block_info.function_calls.empty()) {
    return true;
  }

  // counter new block or tool function call, we need a new content block
  // <content_block_start> ... <content_block_stop>
  if (last_content_block_type != curr_content_block_type ||
      (is_tool_use_delta &&
       !content_block_info.function_calls[0].name.empty())) {
    // try to create new content block
    if (!start_new_content_block(call,
                                 last_content_block_type,
                                 curr_content_block_type,
                                 content_block_info,
                                 content_block_index)) {
      return false;
    }
  }

  proto::AnthropicStreamEvent chunk;
  chunk.set_index(content_block_index);
  chunk.set_type("content_block_delta");
  auto* delta = chunk.mutable_delta();
  if (delta_type == "text_delta") {
    delta->set_type("text_delta");
    delta->set_text(content_block_info.normal_text);
  } else if (delta_type == "thinking_delta") {
    delta->set_type("thinking_delta");
    delta->set_thinking(content_block_info.normal_text);
  } else if (is_tool_use_delta) {
    std::optional<proto::AnthropicStreamEvent> event =
        api_service::make_input_json_delta_event(
            static_cast<int32_t>(content_block_index),
            content_block_info.function_calls[0].arguments);
    if (!event.has_value()) {
      return true;
    }
    if (!call->write(event->type(), event.value())) {
      LOG(ERROR) << "Failed to send content_block_delta event";
      return false;
    }
    return true;
  } else {
    LOG(FATAL) << "Unknown delta type: " << delta_type;
  }

  if (!call->write(chunk.type(), chunk)) {
    LOG(ERROR) << "Failed to send content_block_delta event";
    return false;
  }

  return true;
}

// for streaming,
// process tool call stream and send content block delta back
bool process_tool_call_stream(std::shared_ptr<AnthropicCall> call,
                              std::string& last_content_block_type,
                              int32_t& content_block_index,
                              std::shared_ptr<StreamOutputParser> stream_parser,
                              size_t index,
                              const std::string& delta) {
  auto* parser = stream_parser->get_tool_call_parser(index);
  if (!parser) {
    return true;
  }

  auto parse_result = parser->parse_streaming_increment(delta);
  if (!parse_result.normal_text.empty()) {
    ContentBlockInfo content_block_info;
    content_block_info.normal_text = parse_result.normal_text;
    if (!send_content_block_delta(call,
                                  last_content_block_type,
                                  "text",
                                  "text_delta",
                                  content_block_info,
                                  content_block_index)) {
      return false;
    }
  }

  for (const auto& call_item : parse_result.calls) {
    stream_parser->set_has_tool_call(index, true);
    std::string tool_call_id;
    std::string function_name;

    if (call_item.name.has_value()) {
      tool_call_id = function_call::utils::generate_tool_call_id();
      function_name = call_item.name.value();
    }

    ContentBlockInfo content_block_info;
    content_block_info.function_calls.emplace_back(FunctionCallInfo{
        .id = tool_call_id,
        .name = function_name,
        .arguments = call_item.parameters,
    });
    if (!send_content_block_delta(call,
                                  last_content_block_type,
                                  "tool_use",
                                  "tool_use_delta",
                                  content_block_info,
                                  content_block_index)) {
      return false;
    }
  }

  return true;
}

// for streaming,
// send stream delta content to client
bool send_delta_to_client(std::shared_ptr<AnthropicCall> call,
                          int32_t& content_block_index,
                          std::string& last_content_block_type,
                          std::string& finish_reason,
                          bool& has_tool_call,
                          const RequestOutput& output,
                          std::shared_ptr<StreamOutputParser> stream_parser,
                          bool named_tool_choice) {
  if (stream_parser && output.outputs.size() > 0) {
    stream_parser->check_resize_for_index(output.outputs.size() - 1);
  }

  for (const auto& seq_output : output.outputs) {
    const auto& index = seq_output.index;
    std::string cur_text = seq_output.text;

    // 1) Handle reasoning text
    if (!cur_text.empty() && stream_parser && stream_parser->is_reasoning()) {
      auto parser = stream_parser->get_reasoning_parser(index);
      auto result = parser->parse_stream_chunk(cur_text);
      if (result.normal_text.has_value()) {
        cur_text = result.normal_text.value();
      } else {
        cur_text = "";
      }
      if (result.reasoning_text.has_value() &&
          !result.reasoning_text->empty()) {
        ContentBlockInfo content_block_info;
        content_block_info.normal_text = result.reasoning_text.value();
        if (!send_content_block_delta(call,
                                      last_content_block_type,
                                      "thinking",
                                      "thinking_delta",
                                      content_block_info,
                                      content_block_index)) {
          return false;
        }
      }
    }

    if (!cur_text.empty()) {
      // 2) Handle tool call: text or tool_use
      if (stream_parser && stream_parser->is_tool_call()) {
        if (!process_tool_call_stream(call,
                                      last_content_block_type,
                                      content_block_index,
                                      stream_parser,
                                      index,
                                      cur_text)) {
          return false;
        }
      } else {
        // 3) Handle text output
        ContentBlockInfo content_block_info;
        content_block_info.normal_text = cur_text;
        if (!send_content_block_delta(call,
                                      last_content_block_type,
                                      "text",
                                      "text_delta",
                                      content_block_info,
                                      content_block_index)) {
          return false;
        }
      }
    }

    if (stream_parser && stream_parser->get_has_tool_call(index)) {
      has_tool_call = true;
    }

    // Handle finish reason
    if (seq_output.finish_reason.has_value()) {
      // Check for unstreamed tool args before sending finish reason
      if (stream_parser && stream_parser->get_has_tool_call(index)) {
        auto send_func = [&](const std::string& arguments,
                             int32_t /*tool_index*/) -> bool {
          ContentBlockInfo content_block_info;
          content_block_info.function_calls.emplace_back(FunctionCallInfo{
              .arguments = arguments,
          });
          return send_content_block_delta(call,
                                          last_content_block_type,
                                          "tool_use",
                                          "tool_use_delta",
                                          content_block_info,
                                          content_block_index);
        };
        if (!api_service::check_for_unstreamed_tool_args(
                stream_parser, index, send_func)) {
          return false;
        }
      }

      finish_reason = seq_output.finish_reason.value();
    }
  }

  // 4) finish request, we need to send the
  // last `content_block_stop` and `message_delta` event
  if (output.finished || output.cancelled) {
    finish_reason = api_service::get_stream_stop_reason(
        output.finished, has_tool_call, finish_reason, named_tool_choice);

    // if content_block_index < 0, means no content block started
    // so we don't need to send content_block_stop event
    if (content_block_index >= 0 &&
        !stop_content_block(
            call, last_content_block_type, content_block_index)) {
      return false;
    }

    // send message_delta event for the last message
    proto::AnthropicStreamEvent message_delta;
    message_delta.set_type("message_delta");
    auto* delta = message_delta.mutable_delta();
    delta->set_stop_reason(finish_reason);
    // Set usage information
    if (output.usage.has_value()) {
      auto* usage = message_delta.mutable_usage();
      const auto& stats = output.usage.value();
      usage->set_input_tokens(stats.num_prompt_tokens);
      usage->set_output_tokens(stats.num_generated_tokens);
    } else {
      auto* usage = message_delta.mutable_usage();
      usage->set_input_tokens(0);
      usage->set_output_tokens(0);
    }
    if (!call->write(message_delta.type(), message_delta)) {
      LOG(ERROR) << "Failed to send message_delta event";
      return false;
    }

    // send message_stop event
    proto::AnthropicStreamEvent stop_message;
    stop_message.set_type("message_stop");
    if (!call->write(stop_message.type(), stop_message)) {
      LOG(ERROR) << "Failed to send message_stop event";
      return false;
    }

    return call->finish();
  }

  return true;
}

}  // namespace

AnthropicServiceImpl::AnthropicServiceImpl(
    LLMMaster* master,
    const std::vector<std::string>& models)
    : APIServiceImpl(models),
      master_(check_master(master)),
      tool_call_parser_format_(
          master->options().tool_call_parser().value_or("")),
      reasoning_parser_format_(
          master->options().reasoning_parser().value_or("")) {}

void AnthropicServiceImpl::count_tokens(std::shared_ptr<AnthropicCall> call) {
  if (master_->get_rate_limiter()->is_limited()) {
    call->finish_with_error(
        StatusCode::RESOURCE_EXHAUSTED,
        "The number of concurrent requests has reached the limit.");
    return;
  }
  const auto& request = call->request();
  RequestParams params(
      request, call->get_x_request_id(), call->get_x_request_time());
  master_->count_chat_tokens(
      api_service::build_anthropic_messages(request),
      std::move(params),
      [call](Status status, int32_t input_tokens) {
        if (!status.ok()) {
          call->finish_with_error(status.code(), status.message());
          return;
        }
        proto::AnthropicCountTokensResponse response;
        response.set_input_tokens(input_tokens);
        response.mutable_context_management()->set_original_input_tokens(
            input_tokens);
        call->write_and_finish(response);
      });
}

void AnthropicServiceImpl::process_async_impl(
    std::shared_ptr<AnthropicCall> call) {
  const auto& rpc_request = call->request();
  const auto& model = rpc_request.model();
  // Check if model is supported
  if (!models_.contains(model)) {
    call->finish_with_error(StatusCode::UNKNOWN, "Model not supported");
    return;
  }

  // Check rate limit
  if (master_->get_rate_limiter()->is_limited()) {
    call->finish_with_error(
        StatusCode::RESOURCE_EXHAUSTED,
        "The number of concurrent requests has reached the limit.");
    return;
  }

  // Build request parameters
  RequestParams request_params(
      rpc_request, call->get_x_request_id(), call->get_x_request_time());

  // Build messages
  std::vector<Message> messages =
      api_service::build_anthropic_messages(rpc_request);

  const auto rendered = master_->chat_template().apply_with_generation_mode(
      messages, request_params.tools, request_params.chat_template_kwargs);
  if (!rendered.has_value()) {
    call->finish_with_error(StatusCode::INVALID_ARGUMENT,
                            "Failed to construct prompt from messages");
    return;
  }
  const bool force_reasoning =
      rendered->generation_mode == ChatTemplateGenerationMode::REASONING;
  // vLLM disables reasoning extraction even when the model's template still
  // ends in <think>, as GLM-5.3's template does.
  const bool reasoning_disabled =
      request_params.chat_template_kwargs.value("enable_thinking",
                                                nlohmann::json(true)) == false;
  const std::string reasoning_parser_format =
      reasoning_disabled ? "" : reasoning_parser_format_;
  auto parsing_tools = request_params.tool_choice == "none"
                           ? std::vector<JsonTool>{}
                           : request_params.tools;
  const bool named_tool_choice = rpc_request.has_tool_choice() &&
                                 rpc_request.tool_choice().type() == "tool";
  const bool required_tool_choice = request_params.tool_choice == "required";
  // Create stream parser if needed
  std::shared_ptr<StreamOutputParser> stream_parser;
  if (request_params.streaming &&
      (!tool_call_parser_format_.empty() || !reasoning_parser_format.empty())) {
    stream_parser =
        std::make_shared<StreamOutputParser>(parsing_tools,
                                             tool_call_parser_format_,
                                             reasoning_parser_format,
                                             force_reasoning);
  }

  const bool saved_streaming = request_params.streaming;
  std::string message_id = request_params.request_id;
  auto saved_tools = std::move(parsing_tools);

  std::optional<std::vector<int>> prompt_tokens = std::nullopt;
  if (rpc_request.has_routing()) {
    prompt_tokens = std::vector<int>{};
    prompt_tokens->reserve(rpc_request.token_ids_size());
    for (int32_t i = 0; i < rpc_request.token_ids_size(); ++i) {
      prompt_tokens->emplace_back(rpc_request.token_ids(i));
    }
    request_params.decode_address = rpc_request.routing().decode_name();
  }
  if (saved_streaming && !prompt_tokens.has_value()) {
    prompt_tokens.emplace();
    if (!master_->tokenizer().encode(rendered->prompt,
                                     &prompt_tokens.value(),
                                     request_params.add_special_tokens)) {
      master_->get_rate_limiter()->decrease_one_request();
      call->finish_with_error(StatusCode::INVALID_ARGUMENT,
                              "Failed to encode prompt");
      return;
    }
  }
  const int32_t input_token_count =
      prompt_tokens.has_value() ? static_cast<int32_t>(prompt_tokens->size())
                                : 0;

  // Handle request
  master_->handle_request(
      std::move(messages),
      std::move(prompt_tokens),
      std::move(request_params),
      call.get(),
      [call,
       model,
       stream = saved_streaming,
       force_reasoning,
       named_tool_choice,
       required_tool_choice,
       input_token_count,
       message_id = std::move(message_id),
       message_started = false,
       content_block_index = int32_t{-1},
       last_content_block_type = std::string{},
       finish_reason = std::string{},
       has_tool_call = false,
       tools = std::move(saved_tools),
       tool_call_parser_format = tool_call_parser_format_,
       reasoning_parser_format,
       stream_parser =
           stream_parser](const RequestOutput& req_output) mutable -> bool {
        // Handle errors
        if (req_output.status.has_value()) {
          const auto& status = req_output.status.value();
          if (!status.ok()) {
            return call->finish_with_error(status.code(), status.message());
          }
        }

        // Anthropic format:
        //
        // event: message_start
        // event: content_block_start
        // event: content_block_delta  (may multiple times)
        // event: content_block_stop
        // event: message_delta        (only once, at the end)
        // event: message_stop
        if (stream) {
          // 1. Send `message_start` event
          if (!message_started) {
            message_started = true;

            proto::AnthropicStreamEvent start_event;
            start_event.set_type("message_start");
            auto* start_message = start_event.mutable_message();
            start_message->set_id(message_id);
            start_message->set_type("message");
            start_message->set_role("assistant");
            start_message->set_model(model);
            auto* usage = start_message->mutable_usage();
            usage->set_input_tokens(req_output.usage.has_value()
                                        ? req_output.usage->num_prompt_tokens
                                        : input_token_count);
            usage->set_output_tokens(0);
            if (!call->write(start_event.type(), start_event)) {
              return false;
            }
          }

          return send_delta_to_client(call,
                                      content_block_index,
                                      last_content_block_type,
                                      finish_reason,
                                      has_tool_call,
                                      req_output,
                                      stream_parser,
                                      named_tool_choice);
        }

        // handle non-streaming response
        proto::ChatResponse chat_response;
        generate_chat_response(chat_response,
                               message_id,
                               model,
                               req_output,
                               tool_call_parser_format,
                               reasoning_parser_format,
                               force_reasoning,
                               tools,
                               named_tool_choice,
                               required_tool_choice);
        return send_result_to_client(
            call, chat_response, !named_tool_choice && !required_tool_choice);
      });
}

}  // namespace xllm

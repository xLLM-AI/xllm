/* Copyright 2025-2026 The xLLM Authors.

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

#include "anthropic_service_impl.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>
#include <google/protobuf/util/json_util.h>

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_set>

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

// Build messages from Anthropic protobuf request
std::vector<Message> build_messages(
    const proto::AnthropicMessagesRequest& request) {
  std::vector<Message> messages;

  // Add system message if provided
  if (request.has_system_string()) {
    messages.emplace_back("system", request.system_string());
  } else if (request.has_system_blocks()) {
    std::string system_text;
    for (const auto& block : request.system_blocks().blocks()) {
      if (block.type() == "text" && block.has_text()) {
        system_text += block.text();
      }
    }
    if (!system_text.empty()) {
      messages.emplace_back("system", system_text);
    }
  }

  // Convert Anthropic messages to internal format
  for (const auto& msg : request.messages()) {
    const std::string& role = msg.role();

    // Handle content - can be string or array of content blocks (oneof)
    switch (msg.message_content_case()) {
      case proto::AnthropicMessage::kContentString:
        // Simple string content
        messages.emplace_back(role, msg.content_string());
        break;

      case proto::AnthropicMessage::kContentBlocks: {
        // Handle complex content blocks
        std::vector<MMContent> content_parts;
        Message::ToolCallVec tool_calls;

        for (const auto& block : msg.content_blocks().blocks()) {
          if (block.type() == "text" && block.has_text()) {
            // Text content block
            content_parts.emplace_back("text", block.text());

          } else if (block.type() == "image" && block.has_source()) {
            // Image content block - convert source to image_url
            std::string image_url;
            auto source_json = api_service::struct_to_json(block.source());
            if (source_json.contains("data")) {
              image_url = source_json["data"].get<std::string>();
            }
            content_parts.emplace_back("image_url", ImageURL{image_url});

          } else if (block.type() == "tool_use") {
            // Tool use block - convert to function call format
            Message::ToolCall tool_call;
            tool_call.id =
                block.has_id()
                    ? block.id()
                    : ("call_" +
                       std::to_string(absl::ToUnixSeconds(absl::Now())));
            tool_call.type = "function";
            tool_call.function.name = block.has_name() ? block.name() : "";
            if (block.has_input()) {
              tool_call.function.arguments =
                  api_service::struct_to_json(block.input()).dump();
            } else {
              tool_call.function.arguments = "{}";
            }
            tool_calls.emplace_back(std::move(tool_call));

          } else if (block.type() == "tool_result") {
            // Tool result block
            if (role == "user") {
              // User's tool result becomes a separate tool message
              Message tool_msg("tool", "");
              tool_msg.tool_call_id = block.has_id() ? block.id() : "";
              if (block.has_content_string()) {
                tool_msg.content = block.content_string();
              }
              messages.emplace_back(std::move(tool_msg));
            } else {
              // Assistant tool result becomes regular text
              std::string tool_result_text = "Tool result: ";
              if (block.has_content_string()) {
                tool_result_text += block.content_string();
              }
              content_parts.emplace_back("text", tool_result_text);
            }
          }
        }

        if (!tool_calls.empty() || !content_parts.empty()) {
          Message new_msg(role, "");

          if (!tool_calls.empty()) {
            new_msg.tool_calls = std::move(tool_calls);
          }

          if (!content_parts.empty()) {
            if (content_parts.size() == 1 && content_parts[0].type == "text") {
              // Single text content - use string directly
              new_msg.content = content_parts[0].text;
            } else {
              // Multiple parts or non-text - use MMContentVec
              new_msg.content = std::move(content_parts);
            }
          }

          messages.emplace_back(std::move(new_msg));
        }
        break;
      }

      default:
        break;
    }
  }

  return messages;
}

// for non-streaming,
// generate chat response first and then convert to anthropic protobuf response.
void generate_chat_response(proto::ChatResponse& response,
                            const std::string& request_id,
                            const std::string& model,
                            const RequestOutput& req_output,
                            const std::string& tool_call_parser_format = "",
                            const std::string& reasoning_parser_format = "",
                            bool is_force_reasoning = false,
                            const std::vector<xllm::JsonTool>& tools = {}) {
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
  }
}

// for non-streaming,
// convert chat response to anthropic protobuf response
template <typename AnthropicCall>
bool send_result_to_client(std::shared_ptr<AnthropicCall> call,
                           const proto::ChatResponse& chat_response) {
  auto& anthropic_response = call->response();

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

    // Add text content block
    auto* text_block = anthropic_response.add_content();
    text_block->set_type("text");
    if (choice.has_message() && choice.message().has_content()) {
      text_block->set_text(choice.message().content());
    } else {
      text_block->set_text("");
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
        if (!tool_call.function().arguments().empty()) {
          google::protobuf::util::JsonStringToMessage(
              tool_call.function().arguments(), tool_block->mutable_input());
        }
      }
    }
  }

  return call->write_and_finish(anthropic_response);
}

// create a new content block like
// `<content_block_start>` ... `<content_block_stop>`
bool start_new_content_block(std::shared_ptr<AnthropicCall> call,
                             std::string& last_content_block_type,
                             const std::string& curr_content_block_type,
                             const ContentBlockInfo& content_block_info,
                             int& content_block_index) {
  // if not the first content block,
  // we need to create a content_block_stop
  if (!last_content_block_type.empty()) {
    proto::AnthropicStreamEvent stop_chunk;
    stop_chunk.set_index(content_block_index);
    stop_chunk.set_type("content_block_stop");
    if (!call->write(stop_chunk.type(), stop_chunk)) {
      LOG(ERROR) << "Failed to send content_block_stop event";
      return false;
    }
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
                              int& content_block_index) {
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
                              int& content_block_index,
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
bool send_delta_to_client(
    std::shared_ptr<AnthropicCall> call,
    ContentBlockInfo& content_block_info,
    bool& content_block_started,
    int& content_block_index,
    std::string& last_content_block_type,
    const std::string& request_id,
    const std::string& model,
    const RequestOutput& output,
    std::shared_ptr<StreamOutputParser> stream_parser = nullptr) {
  if (stream_parser && output.outputs.size() > 0) {
    stream_parser->check_resize_for_index(output.outputs.size() - 1);
  }

  std::string finish_reason = "";
  bool has_tool_call = false;
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
      if (result.reasoning_text.has_value()) {
        ContentBlockInfo content_block_info;
        content_block_info.normal_text = result.reasoning_text.value();
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
                             int tool_index) -> bool {
          ContentBlockInfo content_block_info;
          content_block_info.function_calls.push_back(FunctionCallInfo{
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
        output.finished, has_tool_call, finish_reason);

    // if content_block_index < 0, means no content block started
    // so we don't need to send content_block_stop event
    if (content_block_index >= 0) {
      proto::AnthropicStreamEvent stop_chunk;
      stop_chunk.set_index(content_block_index);
      stop_chunk.set_type("content_block_stop");
      if (!call->write(stop_chunk.type(), stop_chunk)) {
        LOG(ERROR) << "Failed to send content_block_stop event";
        return false;
      }
    }

    // send message_delta event for the last message
    proto::AnthropicStreamEvent message_delta;
    message_delta.set_type("message_delta");
    auto* delta = message_delta.mutable_delta();
    delta->set_stop_reason(finish_reason);
    // Set usage information
    if (output.usage.has_value()) {
      auto* usage = message_delta.mutable_usage();
      usage->set_input_tokens(output.usage.value().num_prompt_tokens);
      usage->set_output_tokens(output.usage.value().num_generated_tokens);
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
  std::vector<Message> messages = build_messages(rpc_request);

  // Create stream parser if needed
  std::shared_ptr<StreamOutputParser> stream_parser;
  if (request_params.streaming && (!tool_call_parser_format_.empty() ||
                                   !reasoning_parser_format_.empty())) {
    stream_parser =
        std::make_shared<StreamOutputParser>(request_params.tools,
                                             tool_call_parser_format_,
                                             reasoning_parser_format_,
                                             false /*is_force_reasoning_*/);
  }

  auto saved_streaming = request_params.streaming;
  auto message_id = request_params.request_id;
  auto saved_tools = request_params.tools;

  std::optional<std::vector<int>> prompt_tokens = std::nullopt;
  if (rpc_request.has_routing()) {
    prompt_tokens = std::vector<int>{};
    prompt_tokens->reserve(rpc_request.token_ids_size());
    for (int32_t i = 0; i < rpc_request.token_ids_size(); ++i) {
      prompt_tokens->emplace_back(rpc_request.token_ids(i));
    }
    request_params.decode_address = rpc_request.routing().decode_name();
  }

  // Handle request
  master_->handle_request(
      std::move(messages),
      std::move(prompt_tokens),
      std::move(request_params),
      call.get(),
      [call,
       model,
       master = master_,
       stream = saved_streaming,
       message_id = std::move(message_id),
       message_started = false,
       content_block_started = false,
       content_block_index = -1,
       last_content_block_type = std::string{},
       tools = std::move(saved_tools),
       tool_call_parser_format = tool_call_parser_format_,
       reasoning_parser_format = reasoning_parser_format_,
       stream_parser =
           stream_parser](const RequestOutput& req_output) mutable -> bool {
        // Handle errors
        if (req_output.status.has_value()) {
          const auto& status = req_output.status.value();
          if (!status.ok()) {
            master->get_rate_limiter()->decrease_one_request();
            return call->finish_with_error(status.code(), status.message());
          }
        }

        // Decrease rate limiter on completion
        if (req_output.finished || req_output.cancelled ||
            req_output.finished_on_prefill_instance) {
          master->get_rate_limiter()->decrease_one_request();
        }

        // Anthropic format:
        //
        // event: message_start
        // event: content_block_start
        // event: content_block_delta  (may multiple times)
        // event: content_block_stop
        // event: message_delta        (only once, at the end)
        // event: message_stop
        // data: [DONE]
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
            usage->set_input_tokens(0);
            usage->set_output_tokens(0);
            if (!call->write(start_event.type(), start_event)) {
              return false;
            }
          }

          ContentBlockInfo content_block_info;
          return send_delta_to_client(call,
                                      content_block_info,
                                      content_block_started,
                                      content_block_index,
                                      last_content_block_type,
                                      message_id,
                                      model,
                                      req_output,
                                      stream_parser);
        }

        // handle non-streaming response
        proto::ChatResponse chat_response;
        generate_chat_response(chat_response,
                               message_id,
                               model,
                               req_output,
                               tool_call_parser_format,
                               reasoning_parser_format,
                               false /*is_force_reasoning_*/,
                               tools);
        return send_result_to_client(call, chat_response);
      });
}

}  // namespace xllm

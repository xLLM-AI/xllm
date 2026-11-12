/* Copyright 2026 The xLLM Authors.
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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "api_service/openai_json.h"
#include "api_service/response_usage.h"
#include "api_service/stream_output_parser.h"
#include "api_service/tool_call_response.h"
#include "chat.pb.h"
#include "core/framework/request/request_output.h"
#include "function_call/function_call_parser.h"

namespace xllm::api_service {

inline void set_logprobs(proto::ChatChoice* choice,
                         const std::optional<std::vector<LogProb>>& logprobs) {
  if (!logprobs.has_value() || logprobs.value().empty()) {
    return;
  }

  auto* proto_logprobs = choice->mutable_logprobs();
  proto_logprobs->mutable_content()->Reserve(logprobs.value().size());
  for (const auto& logprob : logprobs.value()) {
    auto* logprob_proto = proto_logprobs->add_content();
    logprob_proto->set_token(logprob.token);
    logprob_proto->set_token_id(logprob.token_id);
    logprob_proto->set_logprob(logprob.logprob);

    if (logprob.top_logprobs.has_value()) {
      for (const auto& top_logprob : logprob.top_logprobs.value()) {
        auto* top_logprob_proto = logprob_proto->add_top_logprobs();
        top_logprob_proto->set_token(top_logprob.token);
        top_logprob_proto->set_token_id(top_logprob.token_id);
        top_logprob_proto->set_logprob(top_logprob.logprob);
      }
    }
  }
}

template <typename Emit>
bool build_tool_call_chunk(proto::ChatResponse& response,
                           const Emit& emit,
                           size_t index,
                           const std::string& tool_call_id,
                           const std::string& function_name,
                           const std::string& arguments,
                           int32_t tool_index,
                           const std::string& request_id,
                           int64_t created_time,
                           const std::string& model) {
  response.Clear();
  response.set_object("chat.completion.chunk");
  response.set_id(request_id);
  response.set_created(created_time);
  response.set_model(model);

  auto* choice = response.add_choices();
  choice->set_index(index);
  auto* delta = choice->mutable_delta();

  auto* tool_call = delta->add_tool_calls();
  if (!tool_call_id.empty()) {
    tool_call->set_id(tool_call_id);
  }
  tool_call->set_index(tool_index);
  tool_call->set_type("function");

  auto* function = tool_call->mutable_function();
  if (!function_name.empty()) {
    function->set_name(function_name);
  }
  if (!arguments.empty()) {
    function->set_arguments(arguments);
  }

  return emit(response);
}

template <typename Emit>
bool build_normal_text_chunk(proto::ChatResponse& response,
                             const Emit& emit,
                             size_t index,
                             const std::string& content,
                             const std::string& request_id,
                             int64_t created_time,
                             const std::string& model) {
  response.Clear();
  response.set_object("chat.completion.chunk");
  response.set_id(request_id);
  response.set_created(created_time);
  response.set_model(model);

  auto* choice = response.add_choices();
  choice->set_index(index);
  auto* delta = choice->mutable_delta();
  delta->set_content(content);

  return emit(response);
}

template <typename Emit>
bool build_reasoning_text_chunk(proto::ChatResponse& response,
                                const Emit& emit,
                                size_t index,
                                const std::string& content,
                                const std::string& request_id,
                                int64_t created_time,
                                const std::string& model) {
  response.Clear();
  response.set_object("chat.completion.chunk");
  response.set_id(request_id);
  response.set_created(created_time);
  response.set_model(model);

  auto* choice = response.add_choices();
  choice->set_index(index);
  auto* delta = choice->mutable_delta();
  delta->set_reasoning_content(content);

  return emit(response);
}

template <typename Emit>
bool process_tool_call_stream(proto::ChatResponse& response,
                              const Emit& emit,
                              std::shared_ptr<StreamOutputParser> stream_parser,
                              size_t index,
                              const std::string& delta,
                              const std::string& request_id,
                              int64_t created_time,
                              const std::string& model) {
  auto* parser = stream_parser->get_tool_call_parser(index);
  if (!parser) {
    return true;
  }

  auto parse_result = parser->parse_streaming_increment(delta);

  if (!parse_result.normal_text.empty()) {
    if (!build_normal_text_chunk(response,
                                 emit,
                                 index,
                                 parse_result.normal_text,
                                 request_id,
                                 created_time,
                                 model)) {
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

    if (!build_tool_call_chunk(response,
                               emit,
                               index,
                               tool_call_id,
                               function_name,
                               call_item.parameters,
                               call_item.tool_index,
                               request_id,
                               created_time,
                               model)) {
      return false;
    }
  }

  return true;
}

inline bool get_enable_thinking_from_request(
    const nlohmann::json& chat_template_kwargs,
    const std::string& reasoning_parser_format) {
  // Default to true if reasoning_parser is configured
  // This matches chat template defaults for models like glm47 that enable
  // thinking by default
  bool default_value = !reasoning_parser_format.empty();

  if (chat_template_kwargs.empty()) {
    return default_value;
  }

  // Check for explicit enable_thinking or thinking setting
  // qwen3 and glm45/glm47 use enable_thinking and deepseek-v3 uses thinking
  if (chat_template_kwargs.contains("enable_thinking") ||
      chat_template_kwargs.contains("thinking")) {
    auto get_bool_val = [&](const char* key) {
      auto it = chat_template_kwargs.find(key);
      if (it != chat_template_kwargs.end() && it->is_boolean()) {
        return it->get<bool>();
      }
      return false;
    };
    return get_bool_val("enable_thinking") || get_bool_val("thinking");
  }

  return default_value;
}

// Emit each constructed protobuf frame without retaining a connection or a
// frame list.
template <typename Emit>
bool build_chat_delta(proto::ChatResponse& response,
                      bool include_usage,
                      std::unordered_set<size_t>* first_message_sent,
                      const std::string& request_id,
                      int64_t created_time,
                      const std::string& model,
                      const RequestOutput& output,
                      std::shared_ptr<StreamOutputParser> stream_parser,
                      const Emit& emit) {
  if (stream_parser && output.outputs.size() > 0) {
    stream_parser->check_resize_for_index(output.outputs.size() - 1);
  }
  // Build delta frames.
  for (const auto& seq_output : output.outputs) {
    const auto& index = seq_output.index;
    std::string cur_text = seq_output.text;
    bool logprobs_sent = false;

    if (first_message_sent->find(index) == first_message_sent->end()) {
      response.Clear();
      response.set_object("chat.completion.chunk");
      response.set_id(request_id);
      response.set_created(created_time);
      response.set_model(model);
      auto* choice = response.add_choices();
      choice->set_index(index);
      auto* message = choice->mutable_delta();
      message->set_role("assistant");
      message->set_content("");
      first_message_sent->insert(index);
      if (!emit(response)) {
        return false;
      }
    }

    // Handle reasoning text
    if (!cur_text.empty()) {
      if (stream_parser && stream_parser->is_reasoning()) {
        auto parser = stream_parser->get_reasoning_parser(index);
        auto result = parser->parse_stream_chunk(cur_text);
        if (result.normal_text.has_value()) {
          cur_text = result.normal_text.value();
        } else {
          cur_text = "";
        }
        if (result.reasoning_text.has_value()) {
          if (!build_reasoning_text_chunk(response,
                                          emit,
                                          index,
                                          result.reasoning_text.value(),
                                          request_id,
                                          created_time,
                                          model)) {
            return false;
          }
        }
      }
    }

    if (!cur_text.empty()) {
      // Handle tool call text
      if (stream_parser && stream_parser->is_tool_call()) {
        if (!process_tool_call_stream(response,
                                      emit,
                                      stream_parser,
                                      index,
                                      cur_text,
                                      request_id,
                                      created_time,
                                      model)) {
          return false;
        }
      } else {
        response.Clear();
        response.set_object("chat.completion.chunk");
        response.set_id(request_id);
        response.set_created(created_time);
        response.set_model(model);
        auto* choice = response.add_choices();
        choice->set_index(index);
        set_logprobs(choice, seq_output.logprobs);
        logprobs_sent = true;
        auto* message = choice->mutable_delta();
        message->set_content(cur_text);
        if (!emit(response)) {
          return false;
        }
      }
    }

    // Control tokens and parser-buffered text still carry token logprobs.
    if (!logprobs_sent && seq_output.logprobs.has_value() &&
        !seq_output.logprobs->empty()) {
      response.Clear();
      response.set_object("chat.completion.chunk");
      response.set_id(request_id);
      response.set_created(created_time);
      response.set_model(model);
      auto* choice = response.add_choices();
      choice->set_index(index);
      choice->mutable_delta();
      set_logprobs(choice, seq_output.logprobs);
      if (!emit(response)) {
        return false;
      }
    }

    // Handle finish reason
    if (seq_output.finish_reason.has_value()) {
      // Check for unstreamed tool args before sending finish reason
      if (stream_parser && stream_parser->get_has_tool_call(index)) {
        auto send_func = [&](const std::string& arguments, int32_t tool_index) {
          return build_tool_call_chunk(response,
                                       emit,
                                       index,
                                       "",
                                       "",
                                       arguments,
                                       tool_index,
                                       request_id,
                                       created_time,
                                       model);
        };
        if (!api_service::check_for_unstreamed_tool_args(
                stream_parser, index, send_func)) {
          return false;
        }
      }

      response.Clear();
      response.set_object("chat.completion.chunk");
      response.set_id(request_id);
      response.set_created(created_time);
      response.set_model(model);
      auto* choice = response.add_choices();
      choice->set_index(index);
      choice->mutable_delta();
      api_service::set_proto_stop_reason(seq_output.stop_reason,
                                         choice->mutable_stop_reason());

      if (stream_parser && stream_parser->get_has_tool_call(index) &&
          seq_output.finish_reason.value() == "stop") {
        choice->set_finish_reason("tool_calls");
      } else {
        choice->set_finish_reason(seq_output.finish_reason.value());
      }

      if (!emit(response)) {
        return false;
      }
    }
  }

  if (include_usage && output.usage.has_value() &&
      (output.finished || output.cancelled)) {
    response.Clear();
    const auto& usage = output.usage.value();
    response.set_object("chat.completion.chunk");
    response.set_id(request_id);
    response.set_created(created_time);
    response.set_model(model);
    auto* proto_usage = response.mutable_usage();
    api_service::set_proto_usage(proto_usage, usage);
    if (!emit(response)) {
      return false;
    }
  }

  return true;
}

inline void build_chat_result(proto::ChatResponse& response,
                              const std::string& request_id,
                              int64_t created_time,
                              const std::string& model,
                              const RequestOutput& req_output,
                              const std::string& tool_call_parser_format = "",
                              const std::string& reasoning_parser_format = "",
                              bool is_force_reasoning = false,
                              const std::vector<xllm::JsonTool>& tools = {}) {
  response.Clear();
  response.set_object("chat.completion");
  response.set_id(request_id);
  response.set_created(created_time);
  response.set_model(model);

  response.mutable_choices()->Reserve(req_output.outputs.size());
  for (const auto& output : req_output.outputs) {
    auto* choice = response.add_choices();
    choice->set_index(output.index);
    api_service::set_proto_stop_reason(output.stop_reason,
                                       choice->mutable_stop_reason());
    set_logprobs(choice, output.logprobs);
    auto* message = choice->mutable_message();
    message->set_role("assistant");

    // handle reasoning output
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
      if (result.reasoning_text.has_value()) {
        message->set_reasoning_content(result.reasoning_text.value());
      }
    }

    // handle tool call output
    if (!tools.empty() && !tool_call_parser_format.empty() &&
        !cur_text.empty()) {
      auto* arena = response.GetArena();
      auto result =
          api_service::process_tool_calls(cur_text,
                                          tools,
                                          tool_call_parser_format,
                                          output.finish_reason.value_or(""),
                                          arena);

      message->mutable_content()->swap(result.text);

      if (result.tool_calls) {
        auto& source_tool_calls = *result.tool_calls;
        message->mutable_tool_calls()->Swap(&source_tool_calls);
      }

      if (!result.finish_reason.empty()) {
        choice->mutable_finish_reason()->swap(result.finish_reason);
      }
    } else {
      message->set_content(cur_text);
      if (output.finish_reason.has_value()) {
        choice->set_finish_reason(output.finish_reason.value());
      }
    }
  }

  if (req_output.usage.has_value()) {
    const auto& usage = req_output.usage.value();
    auto* proto_usage = response.mutable_usage();
    api_service::set_proto_usage(proto_usage, usage);
  }
}

}  // namespace xllm::api_service

/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "chat_service_impl.h"

#include <absl/strings/escaping.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>
#include <google/protobuf/util/json_util.h>
#include <torch/torch.h>

#include <boost/algorithm/string.hpp>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_set>

#include "api_service/stream_output_parser.h"
#include "api_service/utils.h"
#include "core/common/instance_name.h"
#include "core/common/types.h"
#include "core/distributed_runtime/llm_master.h"
#include "core/distributed_runtime/rec_master.h"
#include "core/distributed_runtime/vlm_master.h"
#include "core/framework/request/rec_type.h"
#include "core/framework/request/request_params.h"
#include "core/util/utils.h"
#include "core/util/uuid.h"
#include "mm_service_utils.h"

namespace xllm {
namespace {

void set_logprobs(proto::ChatChoice* choice,
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

template <typename ChatCall>
bool send_tool_call_chunk(std::shared_ptr<ChatCall> call,
                          size_t index,
                          const std::string& tool_call_id,
                          const std::string& function_name,
                          const std::string& arguments,
                          int tool_index,
                          const std::string& request_id,
                          int64_t created_time,
                          const std::string& model) {
  auto& response = call->response();
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

  return call->write(response);
}

template <typename ChatCall>
bool send_normal_text_chunk(std::shared_ptr<ChatCall> call,
                            size_t index,
                            const std::string& content,
                            const std::string& request_id,
                            int64_t created_time,
                            const std::string& model) {
  auto& response = call->response();
  response.Clear();
  response.set_object("chat.completion.chunk");
  response.set_id(request_id);
  response.set_created(created_time);
  response.set_model(model);

  auto* choice = response.add_choices();
  choice->set_index(index);
  auto* delta = choice->mutable_delta();
  delta->set_content(content);

  return call->write(response);
}

template <typename ChatCall>
bool send_reasoning_text_chunk(std::shared_ptr<ChatCall> call,
                               size_t index,
                               const std::string& content,
                               const std::string& request_id,
                               int64_t created_time,
                               const std::string& model) {
  auto& response = call->response();
  response.Clear();
  response.set_object("chat.completion.chunk");
  response.set_id(request_id);
  response.set_created(created_time);
  response.set_model(model);

  auto* choice = response.add_choices();
  choice->set_index(index);
  auto* delta = choice->mutable_delta();
  delta->set_reasoning_content(content);

  return call->write(response);
}

template <typename ChatCall>
bool process_tool_call_stream(std::shared_ptr<ChatCall> call,
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
    if (!send_normal_text_chunk(call,
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

    if (!send_tool_call_chunk(call,
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

bool get_enable_thinking_from_request(
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

template <typename ChatCall>
bool send_delta_to_client_brpc(
    std::shared_ptr<ChatCall> call,
    bool include_usage,
    std::unordered_set<size_t>* first_message_sent,
    const std::string& request_id,
    int64_t created_time,
    const std::string& model,
    const RequestOutput& output,
    std::shared_ptr<StreamOutputParser> stream_parser = nullptr) {
  auto& response = call->response();

  if (stream_parser && output.outputs.size() > 0) {
    stream_parser->check_resize_for_index(output.outputs.size() - 1);
  }
  // send delta to client
  for (const auto& seq_output : output.outputs) {
    const auto& index = seq_output.index;
    std::string cur_text = seq_output.text;

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
      if (!call->write(response)) {
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
          send_reasoning_text_chunk(call,
                                    index,
                                    result.reasoning_text.value(),
                                    request_id,
                                    created_time,
                                    model);
        }
      }
    }

    if (!cur_text.empty()) {
      // Handle tool call text
      if (stream_parser && stream_parser->is_tool_call()) {
        if (!process_tool_call_stream(call,
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
        auto* message = choice->mutable_delta();
        message->set_content(cur_text);
        if (!call->write(response)) {
          return false;
        }
      }
    }

    // Handle finish reason
    if (seq_output.finish_reason.has_value()) {
      // Check for unstreamed tool args before sending finish reason
      if (stream_parser && stream_parser->get_has_tool_call(index)) {
        auto send_func = [&](const std::string& arguments, int tool_index) {
          return send_tool_call_chunk(call,
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

      if (stream_parser && stream_parser->get_has_tool_call(index) &&
          seq_output.finish_reason.value() == "stop") {
        choice->set_finish_reason("tool_calls");
      } else {
        choice->set_finish_reason(std::move(seq_output.finish_reason.value()));
      }

      if (!call->write(response)) {
        return false;
      }
    }
  }

  if (include_usage && output.usage.has_value()) {
    response.Clear();
    const auto& usage = output.usage.value();
    response.set_object("chat.completion.chunk");
    response.set_id(request_id);
    response.set_created(created_time);
    response.set_model(model);
    auto* proto_usage = response.mutable_usage();
    api_service::set_proto_usage(proto_usage, usage);
    if (!call->write(response)) {
      return false;
    }
  }

  if (output.finished || output.cancelled) {
    response.Clear();
    return call->finish();
  }
  return true;
}

template <typename ChatCall>
bool send_result_to_client_brpc(std::shared_ptr<ChatCall> call,
                                const std::string& request_id,
                                int64_t created_time,
                                const std::string& model,
                                const RequestOutput& req_output,
                                const std::string& tool_call_parser_format = "",
                                const std::string& reasoning_parser_format = "",
                                bool is_force_reasoning = false,
                                const std::vector<xllm::JsonTool>& tools = {}) {
  auto& response = call->response();
  response.set_object("chat.completion");
  response.set_id(request_id);
  response.set_created(created_time);
  response.set_model(model);

  response.mutable_choices()->Reserve(req_output.outputs.size());
  for (const auto& output : req_output.outputs) {
    auto* choice = response.add_choices();
    choice->set_index(output.index);
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

  return call->write_and_finish(response);
}

}  // namespace

ChatServiceImpl::ChatServiceImpl(LLMMaster* master,
                                 const std::vector<std::string>& models)
    : APIServiceImpl(models),
      master_(master),
      tool_call_parser_format_(
          master_->options().tool_call_parser().value_or("")),
      reasoning_parser_format_(
          master_->options().reasoning_parser().value_or("")) {
  CHECK(master_ != nullptr);
  add_model_master(models[0], master);
}

ChatServiceImpl::ChatServiceImpl(RecMaster* master,
                                 const std::vector<std::string>& models)
    : APIServiceImpl(models),
      rec_master_(master),
      // RecMaster does not expose tool_call_parser/reasoning_parser options,
      // and the current Rec scenario does not require these features.
      tool_call_parser_format_(""),
      reasoning_parser_format_("") {
  CHECK(rec_master_ != nullptr);
}

void ChatServiceImpl::add_model_master(const std::string& model,
                                       LLMMaster* master) {
  CHECK(master != nullptr);
  std::unique_lock<std::shared_mutex> lock(llm_model_to_master_mutex_);
  llm_model_to_master_.insert_or_assign(model, master);
  models_.insert(model);
}

LLMMaster* ChatServiceImpl::get_model_master(const std::string& model) const {
  std::shared_lock<std::shared_mutex> lock(llm_model_to_master_mutex_);
  auto it = llm_model_to_master_.find(model);
  if (it == llm_model_to_master_.end()) {
    return nullptr;
  }
  return it->second;
}

void ChatServiceImpl::process_rec_chat_request(std::shared_ptr<ChatCall> call) {
  CHECK(rec_master_ != nullptr);
  const auto& rpc_request = call->request();
  const auto& model = rpc_request.model();

  if (rec_master_->rec_type() != RecType::kLlmRec) {
    call->finish_with_error(StatusCode::INVALID_ARGUMENT,
                            "Chat is only supported for LLMRec models");
    return;
  }

  if (rec_master_->get_rate_limiter()->is_limited()) {
    call->finish_with_error(
        StatusCode::RESOURCE_EXHAUSTED,
        "The number of concurrent requests has reached the limit.");
    return;
  }

  RequestParams request_params(
      rpc_request, call->get_x_request_id(), call->get_x_request_time());

  // Build messages from rpc request (inline code, not extracted to helper)
  std::vector<Message> messages;
  messages.reserve(rpc_request.messages_size());
  for (const auto& message : rpc_request.messages()) {
    messages.emplace_back(message.role(), message.content());
    auto& msg = messages.back();

    if (message.has_tool_call_id()) {
      msg.tool_call_id = message.tool_call_id();
    }

    if (message.has_reasoning_content()) {
      msg.reasoning_content = message.reasoning_content();
    }

    if (message.tool_calls_size() > 0) {
      Message::ToolCallVec tool_calls;
      tool_calls.reserve(message.tool_calls_size());
      for (const auto& tool_call : message.tool_calls()) {
        tool_calls.emplace_back();
        auto& tc = tool_calls.back();
        tc.id = tool_call.id();
        tc.type = tool_call.type();
        tc.function.name = tool_call.function().name();
        tc.function.arguments = tool_call.function().arguments();
      }
      msg.tool_calls = std::move(tool_calls);
    }
  }

  bool include_usage = false;
  if (rpc_request.has_stream_options()) {
    include_usage = rpc_request.stream_options().include_usage();
  }

  // Parse prompt tokens from routing (inline code, not extracted to helper)
  std::optional<std::vector<int>> prompt_tokens = std::nullopt;
  if (rpc_request.has_routing()) {
    prompt_tokens = std::vector<int>{};
    prompt_tokens->reserve(rpc_request.token_ids_size());
    for (int i = 0; i < rpc_request.token_ids_size(); i++) {
      prompt_tokens->emplace_back(rpc_request.token_ids(i));
    }
    request_params.decode_address = rpc_request.routing().decode_name();
  }

  auto saved_streaming = request_params.streaming;
  auto saved_request_id = request_params.request_id;

  rec_master_->handle_request(
      std::move(messages),
      std::move(prompt_tokens),
      std::nullopt,
      std::move(request_params),
      [call,
       model,
       master = rec_master_,
       stream = std::move(saved_streaming),
       include_usage = include_usage,
       first_message_sent = std::unordered_set<size_t>(),
       request_id = std::move(saved_request_id),
       created_time = absl::ToUnixSeconds(absl::Now())](
          const RequestOutput& req_output) mutable -> bool {
        if (req_output.status.has_value()) {
          const auto& status = req_output.status.value();
          if (!status.ok()) {
            master->get_rate_limiter()->decrease_one_request();
            return call->finish_with_error(status.code(), status.message());
          }
        }

        if (req_output.finished || req_output.cancelled ||
            req_output.finished_on_prefill_instance) {
          master->get_rate_limiter()->decrease_one_request();
        }

        if (stream) {
          return send_delta_to_client_brpc(call,
                                           include_usage,
                                           &first_message_sent,
                                           request_id,
                                           created_time,
                                           model,
                                           req_output);
        }
        return send_result_to_client_brpc(
            call, request_id, created_time, model, req_output);
      });
}

// chat_async for brpc with xllm_serice
void ChatServiceImpl::process_async_rpc_impl(
    const proto::ChatRequest* request) {
  const auto& service_request_id = request->service_request_id();
  const auto& target_xservice_addr = request->source_xservice_addr();
  auto callback = [master = master_](const RequestOutput& req_output) -> bool {
    req_output.log_request_status();
    if (req_output.status.has_value()) {
      const auto& status = req_output.status.value();
      if (!status.ok()) {
        // Reduce the number of concurrent requests when a request is
        // finished with error.
        master->get_rate_limiter()->decrease_one_request();
        return master->handle_rpc_response(req_output);
      }
    }
    // Reduce the number of concurrent requests when a request is finished
    // or canceled.
    if (req_output.finished || req_output.cancelled ||
        req_output.finished_on_prefill_instance) {
      master->get_rate_limiter()->decrease_one_request();
    }
    return master->handle_rpc_response(req_output);
  };

  // LLMMaster path (existing logic)
  // Check if the request is being rate-limited.
  CHECK(master_ != nullptr);
  if (master_->get_rate_limiter()->is_limited()) {
    CALLBACK_WITH_ERROR(
        StatusCode::RESOURCE_EXHAUSTED,
        "The number of concurrent requests has reached the limit.",
        service_request_id,
        target_xservice_addr);
    return;
  }

  // check if model is supported
  const auto& rpc_request = *request;
  const auto& model = rpc_request.model();
  if (unlikely(!models_.contains(model))) {
    CALLBACK_WITH_ERROR(StatusCode::UNKNOWN,
                        "Model not supported",
                        service_request_id,
                        target_xservice_addr);
    return;
  }

  RequestParams request_params(rpc_request, "", "");
  std::vector<Message> messages;
  messages.reserve(rpc_request.messages_size());
  for (const auto& message : rpc_request.messages()) {
    messages.emplace_back(message.role(), message.content());
    auto& msg = messages.back();

    if (message.has_tool_call_id()) {
      msg.tool_call_id = message.tool_call_id();
    }

    if (message.has_reasoning_content()) {
      msg.reasoning_content = message.reasoning_content();
    }

    if (message.tool_calls_size() > 0) {
      Message::ToolCallVec tool_calls;
      tool_calls.reserve(message.tool_calls_size());
      for (const auto& tool_call : message.tool_calls()) {
        tool_calls.emplace_back();
        auto& tc = tool_calls.back();
        tc.id = tool_call.id();
        tc.type = tool_call.type();
        tc.function.name = tool_call.function().name();
        tc.function.arguments = tool_call.function().arguments();
      }
      msg.tool_calls = std::move(tool_calls);
    }
  }

  std::optional<std::vector<int>> prompt_tokens = std::nullopt;
  if (rpc_request.has_routing()) {
    prompt_tokens = std::vector<int>{};
    prompt_tokens->reserve(rpc_request.token_ids_size());
    for (int i = 0; i < rpc_request.token_ids_size(); i++) {
      prompt_tokens->emplace_back(rpc_request.token_ids(i));
    }

    request_params.decode_address = rpc_request.routing().decode_name();
  }
  // Preserve parser-relevant special tokens in decoded output
  // so tool call detectors can match their control markers.
  // Aligns with vLLM parser adjust_request behavior.
  if (!tool_call_parser_format_.empty() && !request_params.tools.empty()) {
    request_params.skip_special_tokens = false;
  }

  master_->handle_request(std::move(messages),
                          std::move(prompt_tokens),
                          std::move(request_params),
                          std::nullopt,
                          callback);
}

// chat_async for brpc
void ChatServiceImpl::process_async_impl(std::shared_ptr<ChatCall> call) {
  const auto& rpc_request = call->request();
  const auto& model = rpc_request.model();

  // Route to RecMaster if configured
  if (rec_master_) {
    if (unlikely(!models_.contains(model))) {
      call->finish_with_error(StatusCode::UNKNOWN, "Model not supported");
      return;
    }
    process_rec_chat_request(call);
    return;
  }

  LLMMaster* master = get_model_master(model);
  if (unlikely(master == nullptr)) {
    call->finish_with_error(StatusCode::UNKNOWN, "Model not supported");
    return;
  }
  // LLMMaster path (existing logic)
  // Check if the request is being rate-limited or model is sleeping.
  // is_limited() returns true if sleeping or rate-limited.
  if (unlikely(master->get_rate_limiter()->is_limited())) {
    if (master->get_rate_limiter()->is_sleeping()) {
      call->finish_with_error(StatusCode::UNAVAILABLE,
                              "Model is currently in sleep state.");
    } else {
      call->finish_with_error(
          StatusCode::RESOURCE_EXHAUSTED,
          "The number of concurrent requests has reached the limit.");
    }
    return;
  }

  RequestParams request_params(
      rpc_request, call->get_x_request_id(), call->get_x_request_time());
  std::vector<Message> messages;
  messages.reserve(rpc_request.messages_size());
  for (const auto& message : rpc_request.messages()) {
    messages.emplace_back(message.role(), message.content());
    auto& msg = messages.back();

    if (message.has_tool_call_id()) {
      msg.tool_call_id = message.tool_call_id();
    }

    if (message.has_reasoning_content()) {
      msg.reasoning_content = message.reasoning_content();
    }

    if (message.tool_calls_size() > 0) {
      Message::ToolCallVec tool_calls;
      tool_calls.reserve(message.tool_calls_size());
      for (const auto& tool_call : message.tool_calls()) {
        tool_calls.emplace_back();
        auto& tc = tool_calls.back();
        tc.id = tool_call.id();
        tc.type = tool_call.type();
        tc.function.name = tool_call.function().name();
        tc.function.arguments = tool_call.function().arguments();
      }
      msg.tool_calls = std::move(tool_calls);
    }
  }

  bool include_usage = false;
  if (rpc_request.has_stream_options()) {
    include_usage = rpc_request.stream_options().include_usage();
  }
  std::optional<std::vector<int>> prompt_tokens = std::nullopt;
  if (rpc_request.has_routing()) {
    prompt_tokens = std::vector<int>{};
    prompt_tokens->reserve(rpc_request.token_ids_size());
    for (int i = 0; i < rpc_request.token_ids_size(); i++) {
      prompt_tokens->emplace_back(rpc_request.token_ids(i));
    }

    request_params.decode_address = rpc_request.routing().decode_name();
  }

  if (!tool_call_parser_format_.empty() && !request_params.tools.empty()) {
    request_params.skip_special_tokens = false;
  }

  const bool is_force_reasoning = get_enable_thinking_from_request(
      request_params.chat_template_kwargs, reasoning_parser_format_);

  std::shared_ptr<StreamOutputParser> stream_parser;
  if (request_params.streaming && (!tool_call_parser_format_.empty() ||
                                   !reasoning_parser_format_.empty())) {
    stream_parser =
        std::make_shared<StreamOutputParser>(request_params.tools,
                                             tool_call_parser_format_,
                                             reasoning_parser_format_,
                                             is_force_reasoning);
    CHECK(stream_parser != nullptr) << "create StreamOutputParser failed!";
  }

  auto saved_tools = request_params.tools;
  auto saved_streaming = request_params.streaming;
  auto saved_request_id = request_params.request_id;

  master->handle_request(
      std::move(messages),
      std::move(prompt_tokens),
      std::move(request_params),
      call.get(),
      [call,
       model,
       master = master,
       stream = std::move(saved_streaming),
       include_usage = include_usage,
       first_message_sent = std::unordered_set<size_t>(),
       request_id = std::move(saved_request_id),
       created_time = absl::ToUnixSeconds(absl::Now()),
       json_tools = std::move(saved_tools),
       tool_call_parser_format = tool_call_parser_format_,
       reasoning_parser_format = reasoning_parser_format_,
       is_force_reasoning = is_force_reasoning,
       stream_parser =
           stream_parser](const RequestOutput& req_output) mutable -> bool {
        req_output.log_request_status();
        if (req_output.status.has_value()) {
          const auto& status = req_output.status.value();
          if (!status.ok()) {
            // Reduce the number of concurrent requests when a
            // request is finished with error.
            master->get_rate_limiter()->decrease_one_request();

            return call->finish_with_error(status.code(), status.message());
          }
        }

        // Reduce the number of concurrent requests when a request
        // is finished or canceled.
        if (req_output.finished || req_output.cancelled ||
            req_output.finished_on_prefill_instance) {
          master->get_rate_limiter()->decrease_one_request();
        }

        if (stream) {
          return send_delta_to_client_brpc(call,
                                           include_usage,
                                           &first_message_sent,
                                           request_id,
                                           created_time,
                                           model,
                                           req_output,
                                           stream_parser);
        }
        return send_result_to_client_brpc(call,
                                          request_id,
                                          created_time,
                                          model,
                                          req_output,
                                          tool_call_parser_format,
                                          reasoning_parser_format,
                                          is_force_reasoning,
                                          json_tools);
      });
}

MMChatServiceImpl::MMChatServiceImpl(VLMMaster* master,
                                     const std::vector<std::string>& models)
    : APIServiceImpl(models),
      master_(master),
      tool_call_parser_format_(
          master_->options().tool_call_parser().value_or("")),
      reasoning_parser_format_(
          master_->options().reasoning_parser().value_or("")) {
  CHECK(master != nullptr);
}

void MMChatServiceImpl::process_async_impl(std::shared_ptr<MMChatCall> call) {
  const auto& rpc_request = call->request();
  const auto& req_messages = rpc_request.messages();
  const auto& model = rpc_request.model();
  if (!models_.contains(model)) {
    call->finish_with_error(StatusCode::UNKNOWN, "Model not supported");
    return;
  }

  // Check if the request is being rate-limited.
  if (master_->get_rate_limiter()->is_limited()) {
    call->finish_with_error(
        StatusCode::RESOURCE_EXHAUSTED,
        "The number of concurrent requests has reached the limit.");
    return;
  }

  RequestParams request_params(
      rpc_request, call->get_x_request_id(), call->get_x_request_time());

  std::vector<Message> messages;
  if (!mm_service_utils::build_messages<MMChatCall>(
          req_messages, messages, call, master_->get_image_limit())) {
    return;
  }

  bool include_usage = false;
  if (rpc_request.has_stream_options()) {
    include_usage = rpc_request.stream_options().include_usage();
  }

  const bool is_force_reasoning = get_enable_thinking_from_request(
      request_params.chat_template_kwargs, reasoning_parser_format_);

  std::shared_ptr<StreamOutputParser> stream_parser;
  if (request_params.streaming && (!tool_call_parser_format_.empty() ||
                                   !reasoning_parser_format_.empty())) {
    stream_parser =
        std::make_shared<StreamOutputParser>(request_params.tools,
                                             tool_call_parser_format_,
                                             reasoning_parser_format_,
                                             is_force_reasoning);
    CHECK(stream_parser != nullptr) << "create StreamOutputParser failed!";
  }

  auto saved_streaming = request_params.streaming;
  auto saved_request_id = request_params.request_id;
  auto saved_tools = request_params.tools;

  auto payload = call->take_request_payload();

  // schedule the request
  master_->handle_request(
      std::move(messages),
      std::move(request_params),
      std::move(payload),
      [call,
       model,
       master = master_,
       stream = std::move(saved_streaming),
       include_usage = include_usage,
       first_message_sent = std::unordered_set<size_t>(),
       request_id = std::move(saved_request_id),
       created_time = absl::ToUnixSeconds(absl::Now()),
       json_tools = std::move(saved_tools),
       tool_call_parser_format = tool_call_parser_format_,
       reasoning_parser_format = reasoning_parser_format_,
       is_force_reasoning = is_force_reasoning,
       stream_parser =
           stream_parser](const RequestOutput& req_output) mutable -> bool {
        req_output.log_request_status();
        if (req_output.status.has_value()) {
          const auto& status = req_output.status.value();
          if (!status.ok()) {
            // Reduce the number of concurrent requests when a request is
            // finished with error.
            master->get_rate_limiter()->decrease_one_request();

            return call->finish_with_error(status.code(), status.message());
          }
        }

        // Reduce the number of concurrent requests when a request is finished
        // or canceled.
        if (req_output.finished || req_output.cancelled ||
            req_output.finished_on_prefill_instance) {
          master->get_rate_limiter()->decrease_one_request();
        }

        if (stream) {
          // send delta to client
          return send_delta_to_client_brpc(call,
                                           include_usage,
                                           &first_message_sent,
                                           request_id,
                                           created_time,
                                           model,
                                           req_output,
                                           stream_parser);
        }
        return send_result_to_client_brpc(call,
                                          request_id,
                                          created_time,
                                          model,
                                          req_output,
                                          tool_call_parser_format,
                                          reasoning_parser_format,
                                          is_force_reasoning,
                                          json_tools);
      });
}

}  // namespace xllm

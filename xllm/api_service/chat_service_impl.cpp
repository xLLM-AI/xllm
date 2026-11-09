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

#include "chat_service_impl.h"

#include <absl/strings/escaping.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>
#include <google/protobuf/util/json_util.h>
#include <torch/torch.h>

#include <boost/algorithm/string.hpp>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <utility>

#include "api_service/chat_response_builder.h"
#include "api_service/openai_http.h"
#include "api_service/response_usage.h"
#include "api_service/stream_output_parser.h"
#include "core/common/instance_name.h"
#include "core/common/types.h"
#include "core/distributed_runtime/llm_master.h"
#include "core/distributed_runtime/master_manager.h"
#include "core/distributed_runtime/vlm_master.h"
#include "core/distributed_runtime/xservice_client.h"
#include "core/framework/config/distributed_config.h"
#include "core/framework/request/request_params.h"
#include "core/util/utils.h"
#include "core/util/uuid.h"
#include "mm_service_utils.h"

#ifdef unlikely
#undef unlikely
#endif
#define unlikely(x) __builtin_expect(!!(x), 0)

namespace xllm {
namespace {

using api_service::get_enable_thinking_from_request;
template <typename ChatCall>
bool send_delta_to_client_brpc(
    const std::shared_ptr<ChatCall>& call,
    bool include_usage,
    std::unordered_set<size_t>* first_message_sent,
    const std::string& request_id,
    int64_t created_time,
    const std::string& model,
    const RequestOutput& output,
    std::shared_ptr<StreamOutputParser> parser = nullptr) {
  if (output.usage.has_value()) {
    proto::Usage usage;
    api_service::set_proto_usage(&usage, output.usage.value());
    call->set_stream_usage(usage);
  }
  const bool written = api_service::build_chat_delta(
      call->response(),
      include_usage,
      first_message_sent,
      request_id,
      created_time,
      model,
      output,
      std::move(parser),
      [&call](proto::ChatResponse& frame) { return call->write(frame); });
  if (!written) {
    return false;
  }
  if (output.finished || output.cancelled) {
    call->response().Clear();
    return call->finish();
  }
  return true;
}

template <typename ChatCall>
bool send_result_to_client_brpc(const std::shared_ptr<ChatCall>& call,
                                const std::string& request_id,
                                int64_t created_time,
                                const std::string& model,
                                const RequestOutput& output,
                                const std::string& tool_parser = "",
                                const std::string& reasoning_parser = "",
                                bool force_reasoning = false,
                                const std::vector<JsonTool>& tools = {}) {
  api_service::build_chat_result(call->response(),
                                 request_id,
                                 created_time,
                                 model,
                                 output,
                                 tool_parser,
                                 reasoning_parser,
                                 force_reasoning,
                                 tools);
  return call->write_and_finish(call->response());
}

LLMMaster* check_master(LLMMaster* master) {
  CHECK(master != nullptr);
  return master;
}

}  // namespace

ChatServiceImpl::ChatServiceImpl(LLMMaster* master,
                                 const std::vector<std::string>& models,
                                 std::shared_ptr<MasterManager> master_manager)
    : APIServiceImpl(models),
      master_manager_(std::move(master_manager)),
      tool_call_parser_format_(
          check_master(master)->options().tool_call_parser().value_or("")),
      reasoning_parser_format_(
          master->options().reasoning_parser().value_or("")) {
  CHECK(master_manager_ != nullptr);
}

std::shared_ptr<LLMMaster> ChatServiceImpl::get_model_master(
    const std::string& model) const {
  return std::dynamic_pointer_cast<LLMMaster>(
      master_manager_->find_master(model));
}

void ChatServiceImpl::process_async_rpc_impl(
    const proto::ChatRequest* request) {
  process_request_from_service(*request);
}

void ChatServiceImpl::process_request_from_service(
    const proto::ChatRequest& rpc_request,
    const std::string& x_request_id,
    const std::string& x_request_time) {
  const auto& service_request_id = rpc_request.service_request_id();
  const auto& target_xservice_addr = rpc_request.source_xservice_addr();
  OutputCallback callback = [service_request_id,
                             target_xservice_addr,
                             stream =
                                 rpc_request.stream()](RequestOutput output) {
    // Validation failures may contain only a Status. Bind their owner here too.
    output.service_request_id = service_request_id;
    output.target_xservice_addr = target_xservice_addr;
    output.log_request_status();
    output.stream = stream;
    const auto results = XServiceClient::get_instance()->generations({output});
    return results.size() == 1 && results.front();
  };
  const auto& model = rpc_request.model();
  auto master = get_model_master(model);
  if (unlikely(master == nullptr)) {
    CALLBACK_WITH_ERROR(StatusCode::UNKNOWN,
                        "Model not supported",
                        service_request_id,
                        target_xservice_addr);
    return;
  }
  if (rpc_request.has_routing() &&
      (!master->options().enable_disagg_pd() ||
       rpc_request.routing().decode_name().empty() ||
       rpc_request.token_ids().empty())) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "PD service requests require PD mode, a Decode route "
                        "and prompt token IDs",
                        service_request_id,
                        target_xservice_addr);
    return;
  }

  // LLMMaster path (existing logic)
  // Check if the request is being rate-limited.
  const Status admission = master->get_rate_limiter()->acquire();
  if (!admission.ok()) {
    CALLBACK_WITH_ERROR(admission.code(),
                        admission.message(),
                        service_request_id,
                        target_xservice_addr);
    return;
  }

  RequestParams request_params(rpc_request, x_request_id, x_request_time);
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
    for (int32_t i = 0; i < rpc_request.token_ids_size(); ++i) {
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

  master->handle_request(std::move(messages),
                         std::move(prompt_tokens),
                         std::move(request_params),
                         std::nullopt,
                         std::move(callback));
}

// Ordinary HTTP admission and response handling.
void ChatServiceImpl::process_async_impl(std::shared_ptr<ChatCall> call) {
  const auto& rpc_request = call->request();
  if (!rpc_request.source_xservice_addr().empty() &&
      rpc_request.has_routing()) {
    // Without an attempt ID, the service cannot correlate a result callback.
    if (rpc_request.service_request_id().empty()) {
      call->finish_with_error(StatusCode::INVALID_ARGUMENT,
                              "Missing service callback request ID",
                              "service_request_id");
      return;
    }
    process_request_from_service(
        rpc_request, call->get_x_request_id(), call->get_x_request_time());
    // This acknowledges the result-channel protocol, including business errors.
    call->accept_async();
    return;
  }
  const auto& model = rpc_request.model();

  std::shared_ptr<LLMMaster> master = get_model_master(model);
  if (unlikely(master == nullptr)) {
    call->finish_with_error(StatusCode::NOT_FOUND,
                            "The model `" + model + "` does not exist.",
                            "model");
    return;
  }
  RequestParams request_params(
      rpc_request, call->get_x_request_id(), call->get_x_request_time());
  const Status admission = master->get_rate_limiter()->acquire();
  if (unlikely(!admission.ok())) {
    call->finish_with_error(admission.code(), admission.message());
    return;
  }
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
            return call->finish_with_error(status.code(), status.message());
          }
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
    call->finish_with_error(StatusCode::NOT_FOUND,
                            "The model `" + model + "` does not exist.",
                            "model");
    return;
  }

  const Status admission = master_->get_rate_limiter()->acquire();
  if (!admission.ok()) {
    call->finish_with_error(admission.code(), admission.message());
    return;
  }

  RequestParams request_params(
      rpc_request, call->get_x_request_id(), call->get_x_request_time());

  std::vector<Message> messages;
  if (!mm_service_utils::build_messages<MMChatCall>(
          req_messages, messages, call, master_->get_image_limit())) {
    master_->get_rate_limiter()->decrease_one_request();
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
  }

  auto saved_streaming = request_params.streaming;
  auto saved_request_id = request_params.request_id;
  auto saved_tools = request_params.tools;

  MMInput mm_input(call->take_request_payload());

  // schedule the request
  master_->handle_request(
      std::move(messages),
      std::move(request_params),
      std::move(mm_input),
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
            return call->finish_with_error(status.code(), status.message());
          }
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

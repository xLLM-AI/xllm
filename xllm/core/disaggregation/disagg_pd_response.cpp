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

#include "core/disaggregation/disagg_pd_response.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "api_service/chat_response_builder.h"
#include "api_service/completion_response_builder.h"
#include "api_service/openai_json.h"
#include "api_service/response_usage.h"
#include "chat.pb.h"
#include "completion.pb.h"
#include "core/framework/request/request_output.h"
#include "core/framework/request/request_params.h"
#include "disagg_pd.pb.h"

namespace xllm {
namespace {

using api_service::get_enable_thinking_from_request;

// Own only response construction/encoding state; service delivery stays in
// XServiceClient.
template <typename Request, typename Response, typename Build>
std::shared_ptr<ServiceOutputFormatter> make_output_formatter(
    Request request,
    std::string system_fingerprint,
    Build build) {
  if (request.source_xservice_addr().empty() ||
      request.service_request_id().empty()) {
    LOG(ERROR) << "Service output requires a service address and request ID";
    return {};
  }
  const bool stream = request.stream();
  ServiceOutputFormatter format =
      [request = std::move(request),
       response = Response{},
       usage = proto::Usage{},
       stream,
       system_fingerprint = std::move(system_fingerprint),
       build = std::move(build),
       sequence = uint64_t{0},
       closed = false](const RequestOutput& output,
                       bool http,
                       std::optional<ServiceGenerationOutput>* result) mutable {
        CHECK(result != nullptr);
        result->reset();
        if (closed) {
          return false;
        }
        const bool failed = output.status.has_value() && !output.status->ok();
        const bool finished = failed || output.finished || output.cancelled;
        std::string payload;
        std::shared_ptr<proto::StructuredChatOutput> chat_output;
        if constexpr (std::is_same_v<Response, proto::ChatResponse>) {
          if (!http) {
            chat_output = std::make_shared<proto::StructuredChatOutput>();
          }
        }
        if (failed && http) {
          payload = api_service::openai_error_json(output.status->code(),
                                                   output.status->message());
          if (stream) {
            payload = "data: " + payload + "\n\n";
          }
        } else if (!failed) {
          if (output.usage.has_value()) {
            api_service::set_proto_usage(&usage, output.usage.value());
          }
          auto encode = [&](const Response& frame) {
            if (!http) {
              if constexpr (std::is_same_v<Response, proto::ChatResponse>) {
                auto* parsed_frame = chat_output->add_frames();
                *parsed_frame = frame;
                const auto& options = request.stream_options();
                if (stream && options.include_usage() &&
                    options.continuous_usage_stats() && !frame.has_usage()) {
                  *parsed_frame->mutable_usage() = usage;
                }
              }
              return true;
            }
            std::string json = api_service::serialize_openai_response(
                request, frame, usage, system_fingerprint, /*delta=*/stream);
            if (stream) {
              payload += "data: ";
              payload += json;
              payload += "\n\n";
            } else {
              payload = std::move(json);
            }
            return true;
          };
          if (http || std::is_same_v<Response, proto::ChatResponse>) {
            if (!build(response, output, encode)) {
              closed = true;
              return false;
            }
          }
        }
        if (http && stream && finished) {
          payload += "data: [DONE]\n\n";
        }
        if (http && payload.empty()) {
          return true;
        }
        result->emplace();
        (*result)->payload = std::move(payload);
        (*result)->chat_output = std::move(chat_output);
        (*result)->sequence = sequence++;
        (*result)->stream = stream;
        (*result)->finished = finished;
        (*result)->status_code =
            failed ? api_service::openai_http_status(output.status->code())
                   : 200;
        closed = finished;
        return !failed;
      };
  return std::make_shared<ServiceOutputFormatter>(std::move(format));
}

proto::ChatRequest response_options(const proto::DisaggRequest& request) {
  proto::ChatRequest response;
  response.set_source_xservice_addr(request.source_xservice_addr());
  response.set_model(request.model());
  response.set_stream(request.stream());
  *response.mutable_tools() = request.tools();
  response.set_tool_choice(request.tool_choice());
  *response.mutable_chat_template_kwargs() = request.chat_template_kwargs();
  *response.mutable_stream_options() = request.stream_options();
  return response;
}

std::shared_ptr<ServiceOutputFormatter> make_completion_output_formatter(
    const proto::CompletionRequest& request,
    std::string system_fingerprint) {
  const bool stream = request.stream();
  const bool include_usage = request.stream_options().include_usage();
  const int64_t created = absl::ToUnixSeconds(absl::Now());
  proto::CompletionRequest response;
  response.set_request_id(request.request_id());
  response.set_service_request_id(request.service_request_id());
  response.set_source_xservice_addr(request.source_xservice_addr());
  response.set_model(request.model());
  response.set_stream(stream);
  *response.mutable_stream_options() = request.stream_options();
  auto format = [stream,
                 include_usage,
                 request_id = request.request_id(),
                 model = request.model(),
                 created,
                 offsets = std::unordered_map<size_t, int32_t>()](
                    proto::CompletionResponse& response,
                    const RequestOutput& output,
                    const auto& emit) mutable {
    if (stream) {
      return api_service::build_completion_delta(response,
                                                 include_usage,
                                                 request_id,
                                                 created,
                                                 model,
                                                 output,
                                                 &offsets,
                                                 emit);
    }
    api_service::build_completion_result(
        response, request_id, created, model, output);
    return emit(response);
  };
  return make_output_formatter<proto::CompletionRequest,
                               proto::CompletionResponse>(
      std::move(response), std::move(system_fingerprint), std::move(format));
}

std::shared_ptr<ServiceOutputFormatter> make_chat_output_formatter(
    const proto::ChatRequest& request,
    const std::string& tool_parser_format,
    const std::string& reasoning_parser_format,
    std::string system_fingerprint) {
  const bool stream = request.stream();
  const bool include_usage = request.stream_options().include_usage();
  const int64_t created = absl::ToUnixSeconds(absl::Now());
  proto::ChatRequest response;
  response.set_request_id(request.request_id());
  response.set_service_request_id(request.service_request_id());
  response.set_source_xservice_addr(request.source_xservice_addr());
  response.set_model(request.model());
  response.set_stream(stream);
  *response.mutable_tools() = request.tools();
  response.set_tool_choice(request.tool_choice());
  *response.mutable_chat_template_kwargs() = request.chat_template_kwargs();
  *response.mutable_stream_options() = request.stream_options();
  RequestParams params(response, /*x_rid=*/"", /*x_rtime=*/"");
  const bool force_reasoning = get_enable_thinking_from_request(
      params.chat_template_kwargs, reasoning_parser_format);
  std::shared_ptr<StreamOutputParser> parser;
  if (stream &&
      (!tool_parser_format.empty() || !reasoning_parser_format.empty())) {
    parser = std::make_shared<StreamOutputParser>(params.tools,
                                                  tool_parser_format,
                                                  reasoning_parser_format,
                                                  force_reasoning);
  }
  auto format = [parser = std::move(parser),
                 tools = std::move(params.tools),
                 stream,
                 include_usage,
                 request_id = request.request_id(),
                 model = request.model(),
                 created,
                 tool_parser_format,
                 reasoning_parser_format,
                 force_reasoning,
                 first_message_sent = std::unordered_set<size_t>()](
                    proto::ChatResponse& response,
                    const RequestOutput& output,
                    const auto& emit) mutable {
    if (stream) {
      return api_service::build_chat_delta(response,
                                           include_usage,
                                           &first_message_sent,
                                           request_id,
                                           created,
                                           model,
                                           output,
                                           parser,
                                           emit);
    }
    api_service::build_chat_result(response,
                                   request_id,
                                   created,
                                   model,
                                   output,
                                   tool_parser_format,
                                   reasoning_parser_format,
                                   force_reasoning,
                                   tools);
    return emit(response);
  };
  return make_output_formatter<proto::ChatRequest, proto::ChatResponse>(
      std::move(response), std::move(system_fingerprint), std::move(format));
}

}  // namespace

std::shared_ptr<ServiceOutputFormatter> make_decode_output_formatter(
    const proto::DisaggRequest& request,
    const std::string& tool_parser_format,
    const std::string& reasoning_parser_format,
    std::string system_fingerprint) {
  if (request.response_type() == proto::TEXT_COMPLETION) {
    proto::CompletionRequest response;
    response.set_request_id(request.req_id());
    response.set_service_request_id(request.service_req_id());
    response.set_source_xservice_addr(request.source_xservice_addr());
    response.set_model(request.model());
    response.set_stream(request.stream());
    *response.mutable_stream_options() = request.stream_options();
    return make_completion_output_formatter(response,
                                            std::move(system_fingerprint));
  }
  if (request.response_type() != proto::CHAT_COMPLETION) {
    LOG(ERROR) << "Unsupported PD response type: " << request.response_type();
    return {};
  }
  auto response = response_options(request);
  response.set_request_id(request.req_id());
  response.set_service_request_id(request.service_req_id());
  return make_chat_output_formatter(response,
                                    tool_parser_format,
                                    reasoning_parser_format,
                                    std::move(system_fingerprint));
}

}  // namespace xllm

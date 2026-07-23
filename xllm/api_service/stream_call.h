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

#pragma once

#include <brpc/controller.h>
#include <butil/iobuf.h>
#include <glog/logging.h>
#include <json2pb/pb_to_json.h>

#include <atomic>
#include <memory>
#include <optional>
#include <string>

#include "anthropic.pb.h"
#include "api_service/anthropic_json.h"
#include "api_service/api_error.h"
#include "api_service/call.h"
#include "core/common/types.h"
#include "core/util/verbose_trace_logger.h"

namespace xllm {

template <typename Request, typename Response>
class StreamCall : public Call {
 public:
  using ReqType = Request;
  using ResType = Response;

  StreamCall(brpc::Controller* controller,
             ::google::protobuf::Closure* done,
             Request* request,
             Response* response,
             bool use_arena = false)
      : Call(controller, request_body_x_request_id(request)),
        done_(done),
        request_(request),
        response_(response),
        use_arena_(use_arena) {
    stream_ = request_->stream();
    if (stream_) {
      pa_ = controller_->CreateProgressiveAttachment();

      // Send the first SSE response
      controller_->http_response().set_content_type(
          "text/event-stream; charset=utf-8");
      controller_->http_response().set_status_code(200);
      controller_->http_response().SetHeader("Connection", "keep-alive");
      controller_->http_response().SetHeader("Cache-Control", "no-cache");
      // Done Run first for steam response
      done_->Run();

    } else {
      controller_->http_response().set_content_type("application/json");
    }

    json_options_.bytes_to_base64 = false;
    json_options_.jsonify_empty_array = true;
  }

  ~StreamCall() override {
    // For non stream response, call brpc done Run
    if (!stream_) {
      done_->Run();
    }
    if (!use_arena_) {
      delete request_;
      delete response_;
    }
  }

  bool write_and_finish(Response& response) {
    butil::IOBufAsZeroCopyOutputStream json_output(
        &controller_->response_attachment());
    std::string err_msg;
    if (!json2pb::ProtoMessageToJson(
            response, &json_output, json_options_, &err_msg)) {
      return finish_with_error(StatusCode::UNKNOWN, err_msg);
    }
    XLLM_VERBOSE_TRACE() << "event=request_completed http=200 x-request-id="
                         << x_request_id();
    return true;
  }

  virtual bool finish_with_error(const StatusCode& code,
                                 const std::string& error_message) {
    if (!stream_) {
      api_service::write_openai_error(
          controller_, code, error_message, x_request_id());
    } else {
      // The stream has already committed a 200 header, so the error can only be
      // surfaced as a terminal SSE event carrying the OpenAI error envelope.
      io_buf_.clear();
      io_buf_.append("data: ");
      io_buf_.append(api_service::make_openai_error_json(code, error_message));
      io_buf_.append("\n\n");
      pa_->Write(io_buf_);
      api_service::log_request_error(code, error_message, x_request_id());
    }

    return true;
  }

  // For stream response
  bool write(Response& response) {
    io_buf_.clear();
    io_buf_.append("data: ");
    butil::IOBufAsZeroCopyOutputStream json_output(&io_buf_);
    std::string err_msg;
    if (!json2pb::ProtoMessageToJson(
            response, &json_output, json_options_, &err_msg)) {
      LOG(ERROR) << "Failed to convert proto to json: " << err_msg;
      return false;
    }
    io_buf_.append("\n\n");

    connection_status_ |= pa_->Write(io_buf_);
    return true;
  }

  // For stream response
  bool finish() {
    io_buf_.clear();
    io_buf_.append("data: [DONE]\n\n");

    pa_->Write(io_buf_);
    XLLM_VERBOSE_TRACE() << "event=request_completed http=200 x-request-id="
                         << x_request_id();
    return true;
  }

  bool is_disconnected() const override {
    if (stream_) {
      return connection_status_ != 0;
    } else {
      if (controller_) {
        return controller_->IsCanceled();
      }
      return true;
    }
  }

  const Request& request() const { return *request_; }
  Response& response() { return *response_; }
  ::google::protobuf::Closure* done() { return done_; }

 protected:
  ::google::protobuf::Closure* done_;

  Request* request_ = nullptr;
  Response* response_ = nullptr;

  bool stream_ = false;
  bool use_arena_ = false;
  butil::intrusive_ptr<brpc::ProgressiveAttachment> pa_;
  butil::IOBuf io_buf_;

  json2pb::Pb2JsonOptions json_options_;

  int connection_status_ = 0;
};

// Anthropic SSE stream call with custom event formatting
class AnthropicCall : public StreamCall<proto::AnthropicMessagesRequest,
                                        proto::AnthropicMessagesResponse> {
 public:
  AnthropicCall(brpc::Controller* controller,
                ::google::protobuf::Closure* done,
                proto::AnthropicMessagesRequest* request,
                proto::AnthropicMessagesResponse* response,
                bool use_arena = false)
      : StreamCall<proto::AnthropicMessagesRequest,
                   proto::AnthropicMessagesResponse>(controller,
                                                     done,
                                                     request,
                                                     response,
                                                     use_arena) {}

  ~AnthropicCall() override = default;

  // Anthropic uses its own error envelope: a JSON object
  // {"type":"error","error":{"type":...,"message":...}} for non-stream and an
  // `error` SSE event for streaming responses.
  bool finish_with_error(const StatusCode& code,
                         const std::string& error_message) override {
    if (!this->stream_) {
      api_service::write_anthropic_error(
          this->controller_, code, error_message, this->x_request_id());
    } else {
      this->io_buf_.clear();
      this->io_buf_.append("event: error\ndata: ");
      this->io_buf_.append(
          api_service::make_anthropic_error_json(code, error_message));
      this->io_buf_.append("\n\n");
      this->pa_->Write(this->io_buf_);
      api_service::log_request_error(code, error_message, this->x_request_id());
    }
    return true;
  }

  bool write_and_finish(proto::AnthropicMessagesResponse& response) {
    std::string json;
    std::string err_msg;
    if (!api_service::proto_to_anthropic_json(
            response, this->json_options_, &json, &err_msg)) {
      return this->finish_with_error(StatusCode::UNKNOWN, err_msg);
    }
    this->controller_->response_attachment().append(json);
    XLLM_VERBOSE_TRACE() << "event=request_completed http=200 x-request-id="
                         << this->x_request_id();
    return true;
  }

  // Write SSE event with Anthropic format: event: <type>\ndata: <json>\n\n
  bool write(const std::string& event_type, const std::string& json_data) {
    this->io_buf_.clear();
    this->io_buf_.append("event: ");
    this->io_buf_.append(event_type);
    this->io_buf_.append("\ndata: ");
    this->io_buf_.append(json_data);
    this->io_buf_.append("\n\n");

    this->connection_status_ |= this->pa_->Write(this->io_buf_);
    return this->connection_status_ == 0;
  }

  // Write SSE event with proto message
  template <typename ProtoMessage>
  bool write(const std::string& event_type, const ProtoMessage& message) {
    this->io_buf_.clear();
    this->io_buf_.append("event: ");
    this->io_buf_.append(event_type);
    this->io_buf_.append("\ndata: ");
    std::string json;
    std::string err_msg;
    if (!api_service::proto_to_anthropic_json(
            message, this->json_options_, &json, &err_msg)) {
      LOG(ERROR) << "Failed to convert proto to json: " << err_msg;
      return false;
    }
    this->io_buf_.append(json);
    this->io_buf_.append("\n\n");
    this->connection_status_ |= this->pa_->Write(this->io_buf_);
    return this->connection_status_ == 0;
  }
};

template <typename T>
struct is_stream_call : std::false_type {};

template <typename... Args>
struct is_stream_call<StreamCall<Args...>> : std::true_type {};

template <typename T>
inline constexpr bool is_stream_call_v = is_stream_call<T>::value;

}  // namespace xllm

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

#pragma once

#include <brpc/controller.h>
#include <butil/iobuf.h>
#include <glog/logging.h>
#include <json2pb/pb_to_json.h>

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>

#include "anthropic.pb.h"
#include "api_service/anthropic_json.h"
#include "api_service/call.h"
#include "api_service/openai_http.h"
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
             bool use_arena = false,
             bool is_http_request = false,
             bool defer_stream_start = false)
      : Call(controller, request_body_x_request_id(request), is_http_request),
        done_(done),
        request_(request),
        response_(response),
        use_arena_(use_arena) {
    openai_http_ = is_http_request &&
                   (std::is_same_v<Response, proto::ChatResponse> ||
                    std::is_same_v<Response, proto::CompletionResponse>);
    stream_ = request_->stream();
    if (stream_ && !openai_http_ && !defer_stream_start) {
      start_stream();
    } else {
      controller_->http_response().set_content_type("application/json");
    }

    json_options_.bytes_to_base64 = false;
    json_options_.jsonify_empty_array = false;
  }

  ~StreamCall() override {
    // For non stream response, call brpc done Run
    if (!stream_started_) {
      done_->Run();
    }
    if (!use_arena_) {
      delete request_;
      delete response_;
    }
  }

  bool write_and_finish(Response& response) {
    if constexpr (std::is_same_v<Response, proto::ChatResponse> ||
                  std::is_same_v<Response, proto::CompletionResponse>) {
      if (openai_http_ && use_openai_json(response)) {
        controller_->http_response().set_content_type("application/json");
        controller_->response_attachment().append(response_json(response).dump(
            -1, ' ', false, nlohmann::json::error_handler_t::replace));
        return true;
      }
    }
    butil::IOBufAsZeroCopyOutputStream json_output(
        &controller_->response_attachment());
    std::string err_msg;
    if (!json2pb::ProtoMessageToJson(
            response, &json_output, json_options_, &err_msg)) {
      return finish_with_error(StatusCode::UNKNOWN, err_msg);
    }
    XLLM_VERBOSE_TRACE() << "event=response_serialized x-request-id="
                         << x_request_id_;
    return true;
  }

  bool finish_with_error(const StatusCode& code,
                         const std::string& error_message,
                         const std::string& param = "") {
    XLLM_VERBOSE_TRACE() << "event=request_error x-request-id=" << x_request_id_
                         << " message=" << error_message;
    if (openai_http_) {
      if (stream_finished_.load(std::memory_order_acquire)) {
        return false;
      }
      if (!stream_started_) {
        api_service::write_openai_error(
            controller_, code, error_message, param);
        stream_finished_.store(true, std::memory_order_release);
      } else {
        io_buf_.clear();
        io_buf_.append("data: ");
        io_buf_.append(
            api_service::openai_error_json(code, error_message, param));
        io_buf_.append("\n\n");
        connection_status_ |= pa_->Write(io_buf_);
        finish();
      }
      return false;
    }
    if (!stream_) {
      controller_->SetFailed(error_message);

    } else {
      io_buf_.clear();
      io_buf_.append(error_message);
      pa_->Write(io_buf_);
    }

    return true;
  }

  // For stream response
  bool write(Response& response) {
    if (stream_finished_.load(std::memory_order_acquire)) {
      return false;
    }
    if (openai_http_ && !stream_started_) {
      start_stream();
    }
    if (pa_ == nullptr) {
      return false;
    }

    io_buf_.clear();
    io_buf_.append("data: ");
    if constexpr (std::is_same_v<Response, proto::ChatResponse> ||
                  std::is_same_v<Response, proto::CompletionResponse>) {
      if (openai_http_ && use_openai_json(response)) {
        auto json = response_json(response);
        const auto& options = request_->stream_options();
        if (options.include_usage() && options.continuous_usage_stats() &&
            !response.has_usage()) {
          json["usage"] = api_service::openai_usage_json(stream_usage_);
        }
        io_buf_.append(json.dump(
            -1, ' ', false, nlohmann::json::error_handler_t::replace));
        io_buf_.append("\n\n");
        connection_status_ |= pa_->Write(io_buf_);
        return connection_status_ == 0;
      }
    }
    butil::IOBufAsZeroCopyOutputStream json_output(&io_buf_);
    std::string err_msg;
    if (!json2pb::ProtoMessageToJson(
            response, &json_output, json_options_, &err_msg)) {
      LOG(ERROR) << "Failed to convert proto to json: " << err_msg;
      return false;
    }
    io_buf_.append("\n\n");

    connection_status_ |= pa_->Write(io_buf_);
    return connection_status_ == 0;
  }

  void set_stream_usage(const proto::Usage& usage) { stream_usage_ = usage; }

  // For stream response
  bool finish() {
    if (stream_finished_.exchange(true, std::memory_order_acq_rel)) {
      return true;
    }

    if (openai_http_ && !stream_started_) {
      start_stream();
    }
    io_buf_.clear();
    io_buf_.append("data: [DONE]\n\n");

    if (pa_ != nullptr) {
      pa_->Write(io_buf_);
      // ProgressiveAttachment sends the HTTP chunked-response terminator from
      // its destructor. Release it here instead of waiting for StreamCall to
      // be destroyed by the scheduler.
      pa_.reset();
    }
    XLLM_VERBOSE_TRACE() << "event=stream_closed x-request-id="
                         << x_request_id_;
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

  void set_system_fingerprint(std::string fingerprint) {
    system_fingerprint_ = std::move(fingerprint);
  }

  const Request& request() const { return *request_; }
  Response& response() { return *response_; }
  ::google::protobuf::Closure* done() { return done_; }

 private:
  nlohmann::json response_json(const Response& response) const {
    nlohmann::json json;
    if constexpr (std::is_same_v<Response, proto::ChatResponse>) {
      const std::string& choice = request_->tool_choice();
      const bool named = !choice.empty() && choice != "none" &&
                         choice != "auto" && choice != "required";
      json = api_service::openai_response_json(
          response, named, choice == "required");
    } else {
      json = api_service::openai_response_json(response, stream_);
    }
    api_service::set_openai_system_fingerprint(
        json,
        system_fingerprint_,
        stream_,
        request_->stream_options().include_usage());
    return json;
  }

  static bool use_openai_json(const Response& response) {
    if constexpr (std::is_same_v<Response, proto::CompletionResponse>) {
      return response.output_tensors().empty();
    }
    return true;
  }

  std::string system_fingerprint_;
  bool openai_http_ = false;
  proto::Usage stream_usage_;

 protected:
  void start_stream() {
    if (stream_started_) {
      return;
    }
    pa_ = controller_->CreateProgressiveAttachment();
    controller_->http_response().set_content_type(
        "text/event-stream; charset=utf-8");
    controller_->http_response().set_status_code(200);
    controller_->http_response().SetHeader("Connection", "keep-alive");
    controller_->http_response().SetHeader("Cache-Control", "no-cache");
    stream_started_ = true;
    done_->Run();
  }

  ::google::protobuf::Closure* done_;

  Request* request_ = nullptr;
  Response* response_ = nullptr;

  bool stream_ = false;
  bool stream_started_ = false;
  bool use_arena_ = false;
  std::atomic<bool> stream_finished_{false};
  butil::intrusive_ptr<brpc::ProgressiveAttachment> pa_;
  butil::IOBuf io_buf_;

  json2pb::Pb2JsonOptions json_options_;

  int connection_status_ = 0;
};

// Anthropic SSE stream call with custom event formatting
class AnthropicCall final
    : public StreamCall<proto::AnthropicMessagesRequest,
                        proto::AnthropicMessagesResponse> {
 public:
  AnthropicCall(brpc::Controller* controller,
                ::google::protobuf::Closure* done,
                proto::AnthropicMessagesRequest* request,
                proto::AnthropicMessagesResponse* response,
                bool use_arena = false,
                bool is_http_request = false)
      : StreamCall<proto::AnthropicMessagesRequest,
                   proto::AnthropicMessagesResponse>(
            controller,
            done,
            request,
            response,
            use_arena,
            is_http_request,
            /*defer_stream_start=*/true) {
    // Anthropic responses require empty content arrays to remain visible.
    this->json_options_.jsonify_empty_array = true;
  }

  ~AnthropicCall() override = default;

  bool finish_with_error(const StatusCode& code, const std::string& message) {
    int32_t http_status = 500;
    std::string type = "internal_error";
    if (code == StatusCode::INVALID_ARGUMENT) {
      http_status = 400;
      type = "BadRequestError";
    } else if (code == StatusCode::RESOURCE_EXHAUSTED) {
      http_status = 429;
      type = "rate_limit_error";
    } else if (code == StatusCode::UNAVAILABLE) {
      http_status = 503;
      type = "overloaded_error";
    }
    const nlohmann::json error = {
        {"type", "error"}, {"error", {{"type", type}, {"message", message}}}};
    if (!this->stream_started_) {
      this->controller_->http_response().set_status_code(http_status);
      this->controller_->response_attachment().clear();
      this->controller_->response_attachment().append(error.dump());
      return true;
    }
    bool written = write("error", error.dump());
    finish();
    return written;
  }

  bool finish() {
    if (this->stream_finished_.exchange(true, std::memory_order_acq_rel)) {
      return true;
    }
    this->pa_.reset();
    return true;
  }

  template <typename ProtoMessage>
  bool write_and_finish(ProtoMessage& response) {
    std::string json;
    std::string err_msg;
    if (!api_service::proto_to_anthropic_json(
            response, this->json_options_, &json, &err_msg)) {
      return this->finish_with_error(StatusCode::UNKNOWN, err_msg);
    }
    this->controller_->response_attachment().append(json);
    XLLM_VERBOSE_TRACE() << "event=response_serialized x-request-id="
                         << this->x_request_id_;
    return true;
  }

  // Write SSE event with Anthropic format: event: <type>\ndata: <json>\n\n
  bool write(const std::string& event_type, const std::string& json_data) {
    if (this->stream_finished_.load(std::memory_order_acquire)) {
      return false;
    }
    this->start_stream();
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
    std::string json;
    std::string err_msg;
    if (!api_service::proto_to_anthropic_json(
            message, this->json_options_, &json, &err_msg)) {
      return finish_with_error(StatusCode::UNKNOWN, err_msg);
    }
    return write(event_type, json);
  }
};

template <typename T>
struct is_stream_call : std::false_type {};

template <typename... Args>
struct is_stream_call<StreamCall<Args...>> : std::true_type {};

template <typename T>
inline constexpr bool is_stream_call_v = is_stream_call<T>::value;

}  // namespace xllm

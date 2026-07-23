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

#include "api_service/api_error.h"
#include "call.h"
#include "core/common/types.h"
#include "core/util/verbose_trace_logger.h"

namespace xllm {

template <typename Request, typename Response>
class NonStreamCall : public Call {
 public:
  using ReqType = Request;
  using ResType = Response;
  NonStreamCall(brpc::Controller* controller,
                ::google::protobuf::Closure* done,
                Request* request,
                Response* response,
                bool use_arena = false)
      : Call(controller, request_body_x_request_id(request)),
        done_(done),
        request_(request),
        response_(response),
        use_arena_(use_arena) {
    controller_->http_response().set_content_type("application/json");

    json_options_.bytes_to_base64 = false;
    json_options_.jsonify_empty_array = true;
    json_options_.always_print_primitive_fields = true;
  }

  ~NonStreamCall() override {
    done_->Run();

    if (!use_arena_) {
      delete request_;
      delete response_;
    }
  }

  void set_bytes_to_base64(bool bytes_to_base64) {
    json_options_.bytes_to_base64 = bytes_to_base64;
  }

  // For non stream response
  bool write_and_finish(Response& response) {
    butil::IOBufAsZeroCopyOutputStream json_output(
        &controller_->response_attachment());
    std::string err_msg;
    if (!json2pb::ProtoMessageToJson(
            response, &json_output, json_options_, &err_msg)) {
      finish_with_error(StatusCode::UNKNOWN, err_msg);
      return false;
    }

    XLLM_VERBOSE_TRACE() << "event=request_completed http=200 x-request-id="
                         << x_request_id();
    return true;
  }

  // For non stream response with json and binary payload
  bool write_and_finish(Response& response, const std::string& binary_payload) {
    if (!write_and_finish(response)) {
      return false;
    }
    if (binary_payload.size() == 0) {
      return true;
    }
    size_t json_len = controller_->response_attachment().size();

    // Overwrite the header to indicate that the response contains JSON + binary
    controller_->http_response().SetHeader("Content-Type",
                                           "application/octet-stream");
    // Add additional header to indicate the json payload length
    controller_->http_response().SetHeader("X-Json-Length",
                                           std::to_string(json_len));
    controller_->response_attachment().append(binary_payload);

    return true;
  }

  // For non stream response
  bool finish_with_error(const StatusCode& code,
                         const std::string& error_message) {
    api_service::write_openai_error(
        controller_, code, error_message, x_request_id());
    return true;
  }

  bool is_disconnected() const override { return controller_->IsCanceled(); }

  const Request& request() const { return *request_; }
  Response& response() { return *response_; }
  ::google::protobuf::Closure* done() { return done_; }

 private:
  ::google::protobuf::Closure* done_;

  Request* request_ = nullptr;
  Response* response_ = nullptr;

  bool use_arena_ = false;
  json2pb::Pb2JsonOptions json_options_;
};

};  // namespace xllm

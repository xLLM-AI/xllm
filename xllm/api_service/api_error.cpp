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

#include "api_service/api_error.h"

#include <glog/logging.h>

#include <nlohmann/json.hpp>
#include <string>
#include <string_view>

#include "core/common/instance_name.h"
#include "core/util/uuid.h"
#include "core/util/verbose_trace_logger.h"

namespace xllm {
namespace api_service {

namespace {

// HTTP status codes used for error responses. Kept local to avoid pulling in
// brpc's http_status_code.h across the api_service headers.
constexpr int32_t kHttpBadRequest = 400;
constexpr int32_t kHttpClientClosedRequest = 499;
constexpr int32_t kHttpInternalServerError = 500;
constexpr int32_t kHttpServiceUnavailable = 503;
constexpr int32_t kHttpGatewayTimeout = 504;
constexpr int32_t kHttpTooManyRequests = 429;

}  // namespace

int32_t status_code_to_http_status(StatusCode code) {
  switch (code) {
    case StatusCode::OK:
      return 200;
    case StatusCode::CANCELLED:
      return kHttpClientClosedRequest;
    case StatusCode::INVALID_ARGUMENT:
      return kHttpBadRequest;
    case StatusCode::DEADLINE_EXCEEDED:
      return kHttpGatewayTimeout;
    case StatusCode::RESOURCE_EXHAUSTED:
      return kHttpTooManyRequests;
    case StatusCode::UNAVAILABLE:
      return kHttpServiceUnavailable;
    case StatusCode::UNKNOWN:
    default:
      return kHttpInternalServerError;
  }
}

const char* status_code_to_openai_error_type(StatusCode code) {
  switch (code) {
    case StatusCode::INVALID_ARGUMENT:
      return "invalid_request_error";
    case StatusCode::RESOURCE_EXHAUSTED:
      return "rate_limit_error";
    case StatusCode::UNAVAILABLE:
      return "service_unavailable_error";
    case StatusCode::DEADLINE_EXCEEDED:
      return "timeout_error";
    case StatusCode::CANCELLED:
      return "request_cancelled";
    case StatusCode::OK:
    case StatusCode::UNKNOWN:
    default:
      return "internal_error";
  }
}

const char* status_code_to_anthropic_error_type(StatusCode code) {
  switch (code) {
    case StatusCode::INVALID_ARGUMENT:
      return "invalid_request_error";
    case StatusCode::RESOURCE_EXHAUSTED:
      return "rate_limit_error";
    case StatusCode::UNAVAILABLE:
      return "overloaded_error";
    case StatusCode::DEADLINE_EXCEEDED:
      return "timeout_error";
    case StatusCode::CANCELLED:
      return "request_cancelled";
    case StatusCode::OK:
    case StatusCode::UNKNOWN:
    default:
      return "api_error";
  }
}

const char* status_code_to_string(StatusCode code) {
  switch (code) {
    case StatusCode::OK:
      return "ok";
    case StatusCode::CANCELLED:
      return "cancelled";
    case StatusCode::INVALID_ARGUMENT:
      return "invalid_argument";
    case StatusCode::DEADLINE_EXCEEDED:
      return "deadline_exceeded";
    case StatusCode::RESOURCE_EXHAUSTED:
      return "resource_exhausted";
    case StatusCode::UNAVAILABLE:
      return "unavailable";
    case StatusCode::UNKNOWN:
    default:
      return "unknown";
  }
}

std::string make_openai_error_json(StatusCode code, std::string_view message) {
  nlohmann::json body;
  auto& error = body["error"];
  error["message"] = message;
  error["type"] = status_code_to_openai_error_type(code);
  error["code"] = status_code_to_string(code);
  error["param"] = nullptr;
  return body.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

std::string make_anthropic_error_json(StatusCode code,
                                      std::string_view message) {
  nlohmann::json body;
  body["type"] = "error";
  auto& error = body["error"];
  error["type"] = status_code_to_anthropic_error_type(code);
  error["message"] = message;
  return body.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

std::string get_header_x_request_id(const brpc::Controller* controller) {
  if (controller == nullptr) {
    return "";
  }
  const std::string* request_id =
      controller->http_request().GetHeader("x-request-id");
  if (request_id != nullptr) {
    return *request_id;
  }
  const std::string* ms_request_id =
      controller->http_request().GetHeader("x-ms-client-request-id");
  if (ms_request_id != nullptr) {
    return *ms_request_id;
  }
  return "";
}

std::string generate_x_request_id() {
  thread_local ShortUUID short_uuid;
  return "req-" + InstanceName::name()->get_name_hash() + "-" +
         short_uuid.random();
}

std::string ensure_x_request_id(const brpc::Controller* controller) {
  std::string x_request_id = get_header_x_request_id(controller);
  if (x_request_id.empty()) {
    x_request_id = generate_x_request_id();
  }
  return x_request_id;
}

void log_request_error(StatusCode code,
                       std::string_view message,
                       std::string_view x_request_id) {
  std::string_view id = x_request_id.empty() ? "-" : x_request_id;
  LOG(ERROR) << "[x-request-id=" << id
             << "] request failed (http=" << status_code_to_http_status(code)
             << ", status=" << status_code_to_string(code) << "): " << message;
  // Mirror the failure to the asynchronous verbose trace log (no-op when the
  // feature is disabled). This keeps the detailed off-hot-path record in sync
  // with every error response we emit, regardless of the entry point.
  XLLM_VERBOSE_TRACE() << "event=request_error x-request-id=" << id
                       << " http=" << status_code_to_http_status(code)
                       << " status=" << status_code_to_string(code)
                       << " message=" << message;
}

namespace {

void write_error_body(brpc::Controller* controller,
                      StatusCode code,
                      std::string_view message,
                      std::string_view x_request_id,
                      const std::string& body) {
  const int32_t http_status = status_code_to_http_status(code);
  log_request_error(code, message, x_request_id);

  controller->http_response().set_content_type("application/json");
  controller->http_response().set_status_code(http_status);
  if (!x_request_id.empty()) {
    controller->http_response().SetHeader("x-request-id",
                                          std::string(x_request_id));
  }
  // Drop any partially serialized payload before emitting the error body.
  controller->response_attachment().clear();
  controller->response_attachment().append(body);
}

}  // namespace

void write_openai_error(brpc::Controller* controller,
                        StatusCode code,
                        std::string_view message,
                        std::string_view x_request_id) {
  if (controller == nullptr) {
    return;
  }
  write_error_body(controller,
                   code,
                   message,
                   x_request_id,
                   make_openai_error_json(code, message));
}

void write_anthropic_error(brpc::Controller* controller,
                           StatusCode code,
                           std::string_view message,
                           std::string_view x_request_id) {
  if (controller == nullptr) {
    return;
  }
  write_error_body(controller,
                   code,
                   message,
                   x_request_id,
                   make_anthropic_error_json(code, message));
}

}  // namespace api_service
}  // namespace xllm

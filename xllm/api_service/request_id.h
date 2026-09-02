/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <string>
#include <string_view>

namespace xllm::api_service {

template <typename Request>
std::string request_body_x_request_id(const Request* request) {
  if (request == nullptr) {
    return "";
  }
  if constexpr (requires(const Request& value) {
                  value.has_x_request_id();
                  value.x_request_id();
                }) {
    if (request->has_x_request_id() && !request->x_request_id().empty()) {
      return request->x_request_id();
    }
  }
  if constexpr (requires(const Request& value) {
                  value.has_request_id();
                  value.request_id();
                }) {
    if (request->has_request_id() && !request->request_id().empty()) {
      return request->request_id();
    }
  }
  return "";
}

bool is_valid_x_request_id(std::string_view x_request_id);

std::string get_header_x_request_id(const brpc::Controller* controller);

std::string get_header_x_request_time(const brpc::Controller* controller);

std::string generate_x_request_id();

std::string resolve_x_request_id(
    const brpc::Controller* controller,
    std::string_view body_x_request_id = std::string_view{});

// Sets the HTTP response x-request-id if it is not already a valid value.
// Reuses a header already written by Call, otherwise resolves inbound headers
// and generates an id only when none is available.
std::string ensure_http_x_request_id(brpc::Controller* controller);

// Ensures the HTTP response has an x-request-id when the handler returns
// without constructing a Call (parse errors, missing services, etc.).
class HttpXRequestIdGuard final {
 public:
  explicit HttpXRequestIdGuard(brpc::Controller* controller)
      : controller_(controller) {}

  ~HttpXRequestIdGuard() { ensure_http_x_request_id(controller_); }

  HttpXRequestIdGuard(const HttpXRequestIdGuard&) = delete;
  HttpXRequestIdGuard& operator=(const HttpXRequestIdGuard&) = delete;
  HttpXRequestIdGuard(HttpXRequestIdGuard&&) = delete;
  HttpXRequestIdGuard& operator=(HttpXRequestIdGuard&&) = delete;

 private:
  brpc::Controller* controller_;
};

}  // namespace xllm::api_service

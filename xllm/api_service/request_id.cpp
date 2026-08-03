/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "api_service/request_id.h"

#include <cstddef>
#include <string>
#include <string_view>

#include "core/common/instance_name.h"
#include "core/util/uuid.h"

namespace xllm::api_service {

namespace {

constexpr size_t kMaxXRequestIdLength = 256;

std::string get_valid_header(const brpc::HttpHeader& header, const char* name) {
  const std::string* value = header.GetHeader(name);
  if (value != nullptr && is_valid_x_request_id(*value)) {
    return *value;
  }
  return "";
}

}  // namespace

bool is_valid_x_request_id(std::string_view x_request_id) {
  if (x_request_id.empty() || x_request_id.size() > kMaxXRequestIdLength) {
    return false;
  }
  for (char character : x_request_id) {
    const unsigned char value = static_cast<unsigned char>(character);
    if (value < 0x20 || value == 0x7f) {
      return false;
    }
  }
  return true;
}

std::string get_header_x_request_id(const brpc::Controller* controller) {
  if (controller == nullptr) {
    return "";
  }

  std::string x_request_id =
      get_valid_header(controller->http_request(), "x-request-id");
  if (x_request_id.empty()) {
    x_request_id =
        get_valid_header(controller->http_request(), "x-ms-client-request-id");
  }
  return x_request_id;
}

std::string generate_x_request_id() {
  thread_local ShortUUID short_uuid;
  return "req-" + InstanceName::name()->get_name_hash() + "-" +
         short_uuid.random();
}

std::string resolve_x_request_id(const brpc::Controller* controller,
                                 std::string_view body_x_request_id) {
  std::string x_request_id = get_header_x_request_id(controller);
  if (x_request_id.empty() && is_valid_x_request_id(body_x_request_id)) {
    x_request_id = body_x_request_id;
  }
  if (x_request_id.empty() && controller != nullptr) {
    x_request_id =
        get_valid_header(controller->http_response(), "x-request-id");
  }
  if (x_request_id.empty()) {
    x_request_id = generate_x_request_id();
  }
  return x_request_id;
}

std::string ensure_http_x_request_id(brpc::Controller* controller) {
  if (controller == nullptr) {
    return generate_x_request_id();
  }
  std::string x_request_id = resolve_x_request_id(controller);
  controller->http_response().SetHeader("x-request-id", x_request_id);
  return x_request_id;
}

}  // namespace xllm::api_service

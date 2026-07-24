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

#include <functional>
#include <string>
#include <utility>

namespace xllm {

template <typename Request>
std::string request_body_x_request_id(const Request* request) {
  if constexpr (requires(const Request& value) {
                  value.has_x_request_id();
                  value.x_request_id();
                }) {
    if (request != nullptr && request->has_x_request_id()) {
      return request->x_request_id();
    }
  }
  return "";
}

class Call {
 public:
  struct CompletionStatus {
    bool failed = false;
  };

  using CompletionCallback = std::function<void(void*)>;

  Call(brpc::Controller* controller,
       std::string body_x_request_id = "",
       bool is_http_request = false,
       CompletionCallback completion_callback = {});
  virtual ~Call() = default;

  const std::string& x_request_time() const { return x_request_time_; }

  // The request-scoped x-request-id: the client header when present, then the
  // request body value, otherwise a server-generated id. Always non-empty
  // after construction and shared across logs, the response and the engine.
  const std::string& x_request_id() const { return x_request_id_; }

  std::string take_request_payload() { return std::move(request_payload_); }
  void init_request_payload();

  virtual bool is_disconnected() const = 0;

 protected:
  void init(std::string body_x_request_id);
  void complete_request();
  void mark_request_failed() { completion_status_.failed = true; }
  bool is_http_request() const { return is_http_request_; }

 protected:
  brpc::Controller* controller_;
  bool is_http_request_ = false;
  CompletionCallback completion_callback_;
  CompletionStatus completion_status_;

  std::string x_request_time_;
  std::string x_request_id_;

  std::string request_payload_;
};

}  // namespace xllm

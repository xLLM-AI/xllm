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

#include <string>

namespace xllm {

class Call {
 public:
  Call(brpc::Controller* controller);
  virtual ~Call() = default;

  const std::string& raw_header_x_request_id() const {
    return raw_header_x_request_id_;
  }
  const std::string& x_request_time() const { return x_request_time_; }

  // The request-scoped x-request-id: the client-supplied value when present,
  // otherwise a server-generated id. Always non-empty after construction and
  // shared across logs, the verbose trace and the engine.
  const std::string& x_request_id() const { return x_request_id_; }

  std::string take_request_payload() { return std::move(request_payload_); }
  void init_request_payload();

  virtual bool is_disconnected() const = 0;

 protected:
  void init();

 protected:
  brpc::Controller* controller_;

  std::string raw_header_x_request_id_;
  std::string x_request_time_;
  std::string x_request_id_;

  std::string request_payload_;
};

}  // namespace xllm

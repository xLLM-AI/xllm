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

  std::string get_x_request_id() { return x_request_id_; }
  std::string get_x_request_time() { return x_request_time_; }

  std::string take_request_payload() { return std::move(request_payload_); }
  void init_request_payload();

  void set_raw_request_body(std::string body) {
    raw_request_body_ = std::move(body);
  }
  void set_request_endpoint(std::string path) {
    request_endpoint_ = std::move(path);
  }
  const std::string& raw_request_body() const { return raw_request_body_; }
  const std::string& request_endpoint() const { return request_endpoint_; }

  virtual bool is_disconnected() const = 0;

 protected:
  void init();

 protected:
  brpc::Controller* controller_;

  std::string x_request_id_;
  std::string x_request_time_;

  std::string request_payload_;
  std::string raw_request_body_;
  std::string request_endpoint_;
};

}  // namespace xllm

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

#include "api_service/rpc_request_metrics.h"

#include "core/common/metrics.h"

namespace xllm {

RpcRequestMetrics::RpcRequestMetrics(brpc::Controller* controller)
    : controller_(controller), active_(controller != nullptr) {
  if (active_) {
    COUNTER_INC(server_request_in_total);
  }
}

RpcRequestMetrics::~RpcRequestMetrics() { record_out(); }

RpcRequestMetrics::RpcRequestMetrics(RpcRequestMetrics&& other) noexcept
    : controller_(other.controller_),
      active_(other.active_),
      failed_(other.failed_),
      error_code_(other.error_code_) {
  other.controller_ = nullptr;
  other.active_ = false;
}

RpcRequestMetrics& RpcRequestMetrics::operator=(
    RpcRequestMetrics&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  record_out();
  controller_ = other.controller_;
  active_ = other.active_;
  failed_ = other.failed_;
  error_code_ = other.error_code_;
  other.controller_ = nullptr;
  other.active_ = false;
  return *this;
}

void RpcRequestMetrics::finish(const brpc::Controller* controller) {
  if (!active_) {
    return;
  }
  if (controller != nullptr) {
    failed_ = controller->Failed();
    error_code_ = controller->ErrorCode();
  }
  controller_ = nullptr;
  record_out();
}

void RpcRequestMetrics::detach() { controller_ = nullptr; }

void RpcRequestMetrics::mark_failed(int32_t error_code) {
  failed_ = true;
  error_code_ = error_code;
}

void RpcRequestMetrics::record_out() {
  if (!active_) {
    return;
  }
  if (controller_ != nullptr) {
    failed_ = controller_->Failed();
    error_code_ = controller_->ErrorCode();
    controller_ = nullptr;
  }
  active_ = false;

  if (!failed_) {
    COUNTER_INC(server_request_total_ok);
    return;
  }

  COUNTER_INC(server_request_total_fail);
  if (error_code_ == brpc::ELIMIT) {
    COUNTER_INC(server_request_total_limit);
  }
}

}  // namespace xllm

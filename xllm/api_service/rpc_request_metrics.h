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

#include <cstdint>

namespace xllm {

// RAII counter for inference RPC admission and completion.
// Ctor increments server_request_in_total. finish() or the destructor
// records ok / fail / ELIMIT from a snapshot. Call must finish() before
// done->Run(); after that the controller may already be recycled.
// Default-constructed instances are inactive (used by tests that build a Call
// without going through APIService).
class RpcRequestMetrics final {
 public:
  RpcRequestMetrics() = default;
  explicit RpcRequestMetrics(brpc::Controller* controller);

  ~RpcRequestMetrics();

  RpcRequestMetrics(const RpcRequestMetrics&) = delete;
  RpcRequestMetrics& operator=(const RpcRequestMetrics&) = delete;

  RpcRequestMetrics(RpcRequestMetrics&& other) noexcept;
  RpcRequestMetrics& operator=(RpcRequestMetrics&& other) noexcept;

  // Snapshot Failed()/ErrorCode() and record immediately. Must be called
  // before done->Run() recycles the controller. Idempotent.
  void finish(const brpc::Controller* controller);

  // Drop the controller pointer without recording. Call this before
  // done->Run() on stream handshake so later mark_failed()/dtor only
  // use the snapshot.
  void detach();

  void mark_failed(int32_t error_code = 0);

 private:
  void record_out();

  const brpc::Controller* controller_ = nullptr;
  bool active_ = false;
  bool failed_ = false;
  int32_t error_code_ = 0;
};

}  // namespace xllm

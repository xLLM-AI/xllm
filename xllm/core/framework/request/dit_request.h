/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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
#include <absl/time/clock.h>
#include <absl/time/time.h>

#include <atomic>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "common.pb.h"
#include "dit_request_output.h"
#include "dit_request_state.h"
#include "request.h"
#include "runtime/dit_forward_params.h"

namespace xllm {

class DiTRequest : public RequestBase {
 public:
  DiTRequest(const std::string& request_id,
             const std::string& x_request_id,
             const std::string& x_request_time,
             const DiTRequestState& state,
             const std::string& service_request_id = "",
             const std::string& source_xservice_addr = "",
             RateLimiter* rate_limiter = nullptr);

  bool finished() const;

  void handle_forward_output(torch::Tensor output);

  void handle_forward_text_output(const std::string& text);

  const DiTRequestOutput generate_output();

  void log_statistic(double total_latency);

  // Mark the request as cancelled (e.g. client disconnected).
  void set_cancel() { cancelled_.store(true, std::memory_order_relaxed); }

  // Whether the request has been cancelled.
  bool cancelled() const { return cancelled_.load(std::memory_order_relaxed); }

  // Check if the client connection is still alive; if disconnected, mark
  // the request as cancelled.  Mirrors Request::update_connection_status().
  void update_connection_status();

  DiTRequestState& state() { return state_; }

 private:
  DiTRequestState state_;
  DiTForwardOutput output_;
  std::atomic<bool> cancelled_{false};
};

}  // namespace xllm

/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <absl/time/clock.h>
#include <absl/time/time.h>

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "common.pb.h"
#include "request_state.h"
#include "sequences_group.h"
#include "stopping_checker.h"

namespace xllm {

class RateLimiter;

class RequestBase {
 public:
  RequestBase(const std::string& request_id,
              const std::string& x_request_id,
              const std::string& x_request_time,
              const std::string& service_request_id = "",
              const std::string& source_xservice_addr = "",
              RateLimiter* rate_limiter = nullptr);

  virtual ~RequestBase();

  RequestBase(const RequestBase&) = delete;
  RequestBase& operator=(const RequestBase&) = delete;
  RequestBase(RequestBase&&) = delete;
  RequestBase& operator=(RequestBase&&) = delete;

  absl::Time created_time() const { return created_time_; }

  // Get the elapsed time since the request was created.
  double elapsed_seconds() const {
    return absl::ToDoubleSeconds(absl::Now() - created_time_);
  }

  const std::string& request_id() const { return request_id_; }

  const std::string& service_request_id() const { return service_request_id_; }

  const std::string& source_xservice_addr() const {
    return source_xservice_addr_;
  }

  const std::string& x_request_id() const { return x_request_id_; }

  const std::string& x_request_time() const { return x_request_time_; }

 protected:
  // request create time
  absl::Time created_time_;

  std::string request_id_;

  std::string service_request_id_;

  std::string source_xservice_addr_;

  // x-request-id header value from client
  std::string x_request_id_;

  // x-request-time header value from client
  std::string x_request_time_;

  // Non-owning; nullptr for profile/warmup requests and PD-decode-side
  // forwarded requests that don't count against the concurrency budget. The
  // slot is owned by whoever constructed the Request (either a scoped
  // ScopeGuard on the failure path, or the Request itself on success). The
  // destructor decrements exactly once when non-null.
  RateLimiter* rate_limiter_ = nullptr;
};

}  // namespace xllm

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

#include "request_base.h"

#include "common/rate_limiter.h"

namespace xllm {

RequestBase::RequestBase(const std::string& request_id,
                         const std::string& x_request_id,
                         const std::string& x_request_time,
                         const std::string& service_request_id,
                         const std::string& source_xservice_addr,
                         RateLimiter* rate_limiter)
    : created_time_(absl::Now()),
      request_id_(request_id),
      service_request_id_(service_request_id),
      source_xservice_addr_(source_xservice_addr),
      x_request_id_(x_request_id),
      x_request_time_(x_request_time),
      rate_limiter_(rate_limiter) {
  // The caller (a service_impl entry) already incremented the counter via
  // RateLimiter::is_limited() returning false. This constructor takes over
  // ownership of that slot; the destructor releases it. Pass nullptr for
  // requests that don't count against the concurrency budget (profile /
  // warmup requests, PD-decode-side forwarded requests).
}

RequestBase::~RequestBase() {
  if (rate_limiter_ != nullptr) {
    rate_limiter_->decrease_one_request();
  }
}

}  // namespace xllm

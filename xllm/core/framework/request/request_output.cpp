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

#include "request_output.h"

#include "core/common/metrics.h"

namespace xllm {

void RequestOutput::log_request_status() const {
  if (!status.has_value()) {
    return;
  }

  auto code = status.value().code();
  switch (code) {
    case StatusCode::OK:
      COUNTER_INC(request_status_total_ok);
      break;
    case StatusCode::CANCELLED:
      COUNTER_INC(request_status_total_cancelled);
      break;
    case StatusCode::UNKNOWN:
      COUNTER_INC(request_status_total_unknown);
      break;
    case StatusCode::INVALID_ARGUMENT:
      COUNTER_INC(request_status_total_invalid_argument);
      break;
    case StatusCode::DEADLINE_EXCEEDED:
      COUNTER_INC(request_status_total_deadline_exceeded);
      break;
    case StatusCode::RESOURCE_EXHAUSTED:
      COUNTER_INC(request_status_total_resource_exhausted);
      break;
    default:
      COUNTER_INC(request_status_total_unknown);
      break;
  }
}

}  // namespace xllm

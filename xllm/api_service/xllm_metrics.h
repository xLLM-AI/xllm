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

#include <brpc/server.h>
#include <google/protobuf/service.h>

#include "core/common/metrics.h"

namespace xllm {

static bool is_failed_request(const brpc::Controller* ctrl) {
  return ctrl->Failed() || ctrl->http_response().status_code() >= 400;
}

static void request_in_metric(void* context) {
  COUNTER_INC(server_request_in_total);
}

static void request_out_metric(void* context) {
  auto ctrl = reinterpret_cast<brpc::Controller*>(context);
  if (ctrl == nullptr) {
    LOG(ERROR) << "ctrl is nullptr";
    return;
  }

  if (!is_failed_request(ctrl)) {
    COUNTER_INC(server_request_total_ok);
  } else {
    COUNTER_INC(server_request_total_fail);
    if (ctrl->ErrorCode() == brpc::ELIMIT) {
      COUNTER_INC(server_request_total_limit);
    }
  }
}

static void device_info_metric() {
  // TODO: get cpu device info
  GAUGE_SET(xllm_cpu_num, 0);
  GAUGE_SET(xllm_cpu_utilization, 0);

  // TODO: get gpu device info
  GAUGE_SET(xllm_gpu_num, 0);
  GAUGE_SET(xllm_gpu_utilization, 0);
}

}  // namespace xllm

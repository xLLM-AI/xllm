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

#include <brpc/health_reporter.h>

#include <nlohmann/json.hpp>

#include "core/common/health_check_manager.h"

namespace xllm {

// Custom health reporter for brpc server.
// Reports cluster health status based on worker connections.
class HealthReporter : public brpc::HealthReporter {
 public:
  static HealthReporter& instance() {
    static HealthReporter reporter;
    return reporter;
  }

  void GenerateReport(brpc::Controller* cntl,
                      google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);

    bool is_healthy = HealthCheckManager::instance().is_healthy();

    nlohmann::ordered_json report;
    if (!is_healthy) {
      std::string reason = HealthCheckManager::instance().unhealthy_reason();
      LOG(ERROR) << "HealthReporter: cluster is unhealthy, reason: " << reason;
      report["status"] = "unhealthy";
      report["reason"] = reason;
      cntl->http_response().set_status_code(503);  // Service Unavailable
    } else {
      report["status"] = "healthy";
      cntl->http_response().set_status_code(200);  // OK
    }

    cntl->http_response().set_content_type("application/json");
    cntl->response_attachment().append(report.dump());
  }
};

}  // namespace xllm

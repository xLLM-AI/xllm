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

#include <cstdint>
#include <nlohmann/json_fwd.hpp>
#include <string>

#include "core/common/macros.h"
#include "core/framework/config/option_category.h"

namespace xllm {

class JsonReader;

class ServiceConfig final {
 public:
  ServiceConfig() = default;
  ~ServiceConfig() = default;

  static ServiceConfig& get_instance();

  void from_flags();
  void from_json(const JsonReader& json);
  void append_config_json(nlohmann::ordered_json& config_json) const;
  void initialize();

  [[nodiscard]] static const OptionCategory& option_category() {
    static const OptionCategory kOptionCategory = {
        "SERVICE OPTIONS",
        {"host",
         "port",
         "rpc_idle_timeout_s",
         "rpc_channel_timeout_ms",
         "max_reconnect_count",
         "num_threads",
         "max_concurrent_requests",
         "num_request_handling_threads",
         "num_response_handling_threads",
         "health_check_interval_ms",
         "enable_verbose_trace_log",
         "verbose_trace_log_path",
         "verbose_trace_log_max_size_mb",
         "verbose_trace_log_max_files"}};
    return kOptionCategory;
  }

  PROPERTY(std::string, host);

  PROPERTY(int32_t, port) = 8010;

  PROPERTY(int32_t, rpc_idle_timeout_s) = -1;

  PROPERTY(int32_t, rpc_channel_timeout_ms) = -1;

  PROPERTY(int32_t, max_reconnect_count) = 40;

  PROPERTY(int32_t, num_threads) = 8;

  PROPERTY(int32_t, max_concurrent_requests) = 200;

  PROPERTY(int32_t, num_request_handling_threads) = 4;

  PROPERTY(int32_t, num_response_handling_threads) = 4;

  PROPERTY(int32_t, health_check_interval_ms) = 3000;

  PROPERTY(bool, enable_verbose_trace_log) = false;

  PROPERTY(std::string, verbose_trace_log_path);

  PROPERTY(int32_t, verbose_trace_log_max_size_mb) = 1024;

  PROPERTY(int32_t, verbose_trace_log_max_files) = 1000;
};

}  // namespace xllm

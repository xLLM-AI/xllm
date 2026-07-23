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

#include "core/framework/config/service_config.h"

#include "core/common/global_flags.h"
#include "core/framework/config/config_utils.h"

DEFINE_string(host, "", "Host name for brpc server.");

DEFINE_int32(port, 8010, "Port for brpc server.");

DEFINE_int32(
    rpc_idle_timeout_s,
    -1,
    "Connection will be closed if there is no read/write operations "
    "during the last `rpc_idle_timeout_s`. -1 means wait indefinitely.");

DEFINE_int32(rpc_channel_timeout_ms,
             -1,
             "Max duration of bRPC Channel. -1 means wait indefinitely.");

DEFINE_int32(max_reconnect_count,
             40,
             "The max count for worker try to connect to server.");

DEFINE_int32(num_threads, 8, "Number of threads to process requests.");

DEFINE_int32(max_concurrent_requests,
             200,
             "Maximum number of concurrent requests the xllm service can "
             "handle. If set to 0, there is no limit.");

DEFINE_int32(num_request_handling_threads,
             4,
             "Number of threads for handling input requests.");

DEFINE_int32(num_response_handling_threads,
             4,
             "Number of threads for handling responses.");

DEFINE_int32(health_check_interval_ms,
             3000,
             "Worker health check interval in milliseconds.");

// --- request input/output trace(dump) config ---
DEFINE_bool(enable_request_trace,
            false,
            "Whether to dump the input and output of each request to a file. "
            "For streaming requests, the complete output is saved after all "
            "the streaming outputs are finished.");

DEFINE_string(request_trace_path,
              "./request_trace.jsonl",
              "The path to dump the request input/output trace. When "
              "request_trace_per_file is false, this is a file that JSON "
              "records are appended to (one record per line). When "
              "request_trace_per_file is true, this is a directory and each "
              "request is written to <dir>/<request_id>.json. Only used when "
              "enable_request_trace is true.");

DEFINE_bool(request_trace_per_file,
            false,
            "Whether to dump each request to a separate file named "
            "<request_trace_path>/<request_id>.json instead of appending all "
            "requests to a single JSON-lines file.");

namespace {
// brpc's /flags endpoint refuses to modify a gflag at runtime unless it has a
// registered validator (it returns "A reloadable gflag must have validator").
// Register a no-op validator so enable_request_trace can be toggled without
// restarting the service, as documented for the request trace feature.
bool validate_enable_request_trace(const char* /*unused*/, bool /*unused*/) {
  return true;
}
const bool kEnableRequestTraceValidatorDummy =
    google::RegisterFlagValidator(&FLAGS_enable_request_trace,
                                  &validate_enable_request_trace);
}  // namespace

namespace xllm {

void ServiceConfig::from_flags() {
  XLLM_CONFIG_ASSIGN_FROM_FLAG(host);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(port);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(rpc_idle_timeout_s);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(rpc_channel_timeout_ms);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(max_reconnect_count);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(num_threads);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(max_concurrent_requests);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(num_request_handling_threads);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(num_response_handling_threads);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(health_check_interval_ms);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_request_trace);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(request_trace_path);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(request_trace_per_file);
}

void ServiceConfig::from_json(const JsonReader& json) {
  XLLM_CONFIG_ASSIGN_FROM_JSON(host);
  // don't read rank-related config
  // XLLM_CONFIG_ASSIGN_FROM_JSON(port);
  XLLM_CONFIG_ASSIGN_FROM_JSON(rpc_idle_timeout_s);
  XLLM_CONFIG_ASSIGN_FROM_JSON(rpc_channel_timeout_ms);
  XLLM_CONFIG_ASSIGN_FROM_JSON(max_reconnect_count);
  XLLM_CONFIG_ASSIGN_FROM_JSON(num_threads);
  XLLM_CONFIG_ASSIGN_FROM_JSON(max_concurrent_requests);
  XLLM_CONFIG_ASSIGN_FROM_JSON(num_request_handling_threads);
  XLLM_CONFIG_ASSIGN_FROM_JSON(num_response_handling_threads);
  XLLM_CONFIG_ASSIGN_FROM_JSON(health_check_interval_ms);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_request_trace);
  XLLM_CONFIG_ASSIGN_FROM_JSON(request_trace_path);
  XLLM_CONFIG_ASSIGN_FROM_JSON(request_trace_per_file);
}

void ServiceConfig::append_config_json(
    nlohmann::ordered_json& config_json) const {
  const ServiceConfig default_config;
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(config_json, default_config, host);
  // only dump rank-0 port
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(config_json, default_config, port);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, rpc_idle_timeout_s);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, rpc_channel_timeout_ms);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, max_reconnect_count);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, num_threads);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, max_concurrent_requests);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, num_request_handling_threads);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, num_response_handling_threads);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, health_check_interval_ms);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_request_trace);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, request_trace_path);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, request_trace_per_file);
}

ServiceConfig& ServiceConfig::get_instance() {
  static ServiceConfig config;
  return config;
}

void ServiceConfig::initialize() {
  from_flags();
  if (const auto& json_config = config::get_parsed_json_config()) {
    from_json(*json_config);
  }
}

}  // namespace xllm

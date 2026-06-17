/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "env_var.h"

#include <glog/logging.h>

#include <climits>
#include <cstdlib>
#include <cstring>

namespace xllm {
namespace util {

bool get_bool_env(const std::string& key, bool defaultValue) {
  const char* val = std::getenv(key.c_str());
  if (val == nullptr) {
    return defaultValue;
  }
  std::string strVal(val);
  return (strVal == "1" || strVal == "true" || strVal == "TRUE" ||
          strVal == "True");
}

int64_t get_int_env(const std::string& key, int64_t defaultValue) {
  const char* val = std::getenv(key.c_str());
  if (val == nullptr) {
    return defaultValue;
  }
  // Use strtol for proper error handling
  char* endptr;
  int64_t result = std::strtol(val, &endptr, 10);
  // Check if conversion was successful (endptr points to end of string or valid
  // terminator)
  if (endptr == val || *endptr != '\0') {
    return defaultValue;
  }
  // Check for overflow/underflow
  if (result < INT64_MIN || result > INT64_MAX) {
    return defaultValue;
  }
  return result;
}

std::string get_string_env(const std::string& name) {
  const char* val = std::getenv(name.c_str());
  if (val == nullptr) {
    LOG(FATAL) << "Environment variable " << name.c_str() << " is not set";
  }
  return std::string(val);
}

std::optional<std::string> get_optional_string_env(const std::string& name) {
  const char* val = std::getenv(name.c_str());
  if (val == nullptr) {
    return std::nullopt;
  }
  return std::string(val);
}

double get_double_env(const std::string& key, double defaultValue = -1) {
  const char* val = std::getenv(key.c_str());
  if (val == nullptr) {
    return defaultValue;
  }
  char* endptr = nullptr;
  double result = std::strtod(val, &endptr);
  // Check if conversion was successful (endptr points to end of string or valid
  // terminator)
  if (endptr == val || *endptr != '\0') {
    LOG(WARNING) << "Invalid value for " << key << ": " << val
                 << ". Must be a valid double. Using default " << defaultValue;
    return defaultValue;
  }
  return result;
}

int64_t get_process_group_test_timeout_seconds() {
  // Default timeout is 4 seconds, but can be overridden via environment
  // variable to accommodate multi-node multi-device communication scenarios
  // where network latency may require a longer timeout period.
  constexpr int64_t kDefaultTimeoutSeconds = 4;
  constexpr const char* kTimeoutEnvVar =
      "XLLM_PROCESS_GROUP_ASYNC_TIMEOUT_SECONDS";
  return get_int_env(kTimeoutEnvVar, kDefaultTimeoutSeconds);
}

}  // namespace util
}  // namespace xllm

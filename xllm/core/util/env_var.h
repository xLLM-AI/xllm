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

#pragma once

#include <optional>
#include <string>

namespace xllm {
namespace util {

bool get_bool_env(const std::string& key, bool defaultValue);

// Get an integer value from an environment variable.
// Returns the default value if the environment variable is not set or cannot be
// parsed.
int64_t get_int_env(const std::string& key, int64_t defaultValue);

std::string get_string_env(const std::string& name);
std::optional<std::string> get_optional_string_env(const std::string& name);

// Get the timeout in seconds for process group test operations.
// This timeout is used when waiting for process group initialization tests
// to complete in multi-device/multi-node scenarios. The default value is 4
// seconds, but can be overridden by setting the
// XLLM_PROCESS_GROUP_ASYNC_TIMEOUT_SECONDS environment variable. This is
// particularly useful in multi-node multi-device communication scenarios
// where network latency may cause the default 4-second timeout to be
// insufficient.
int64_t get_process_group_test_timeout_seconds();

}  // namespace util
}  // namespace xllm

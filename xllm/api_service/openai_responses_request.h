/* Copyright 2026 The xLLM Authors.

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

#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/common/message.h"
#include "core/common/types.h"
#include "core/framework/request/request_params.h"

namespace xllm::api_service {

struct ResponsesRequest {
  std::string model;
  std::vector<Message> messages;
  RequestParams params;
  nlohmann::json instructions = nullptr;
  nlohmann::json metadata = nlohmann::json::object();
  nlohmann::json tools = nlohmann::json::array();
  nlohmann::json reasoning = nullptr;
  nlohmann::json max_output_tokens = nullptr;
};

// Decode the public Responses schema directly into the common engine request.
// Unsupported semantics are errors, not fields discarded by proto conversion.
std::pair<Status, ResponsesRequest> parse_responses_request(
    std::string_view body,
    const std::string& default_model,
    std::string* error_param);

nlohmann::json responses_initial_response(const ResponsesRequest& request);

}  // namespace xllm::api_service

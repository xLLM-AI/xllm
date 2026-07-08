/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "api_service/completion_json_parser.h"

#include <glog/logging.h>

#include <cstdint>
#include <limits>
#include <nlohmann/json.hpp>

namespace xllm {

std::pair<Status, std::string> preprocess_completion_prompt(
    std::string json_str) {
  try {
    auto json = nlohmann::json::parse(json_str);
    if (!json.contains("prompt") || !json["prompt"].is_array()) {
      // string / scalar / missing prompt: leave bytes untouched.
      return {Status(), std::move(json_str)};
    }

    if (json.contains("token_ids")) {
      return {Status(StatusCode::INVALID_ARGUMENT,
                     "specify prompt as a token array or token_ids, not both"),
              ""};
    }

    const auto& prompt = json["prompt"];
    if (prompt.empty()) {
      return {Status(StatusCode::INVALID_ARGUMENT, "prompt array is empty"),
              ""};
    }

    const auto& first = prompt.front();
    if (first.is_string()) {
      return {Status(StatusCode::INVALID_ARGUMENT,
                     "batch prompts (array of strings) are not supported"),
              ""};
    }
    if (first.is_array()) {
      return {Status(StatusCode::INVALID_ARGUMENT,
                     "batch prompts (nested arrays) are not supported"),
              ""};
    }

    std::vector<int32_t> token_ids;
    token_ids.reserve(prompt.size());
    for (const auto& elem : prompt) {
      if (!elem.is_number_integer()) {
        return {Status(StatusCode::INVALID_ARGUMENT,
                       "prompt array must contain only integer token ids"),
                ""};
      }
      if (elem.is_number_unsigned()) {
        uint64_t value = elem.get<uint64_t>();
        if (value >
            static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
          return {Status(StatusCode::INVALID_ARGUMENT,
                         "token id exceeds int32 range"),
                  ""};
        }
        token_ids.emplace_back(static_cast<int32_t>(value));
      } else {
        int64_t value = elem.get<int64_t>();
        if (value < std::numeric_limits<int32_t>::min() ||
            value > std::numeric_limits<int32_t>::max()) {
          return {Status(StatusCode::INVALID_ARGUMENT,
                         "token id exceeds int32 range"),
                  ""};
        }
        token_ids.emplace_back(static_cast<int32_t>(value));
      }
    }

    json["token_ids"] = token_ids;
    json["prompt"] = "";
    return {Status(), json.dump()};
  } catch (const nlohmann::json::exception& e) {
    return {Status(StatusCode::INVALID_ARGUMENT,
                   "Invalid JSON format: " + std::string(e.what())),
            ""};
  } catch (const std::exception& e) {
    LOG(ERROR) << "Exception during completion JSON preprocessing: "
               << e.what();
    return {Status(StatusCode::UNKNOWN,
                   "Internal server error during JSON processing."),
            ""};
  }
}

}  // namespace xllm

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
#include <type_traits>

#include "chat.pb.h"
#include "completion.pb.h"
#include "core/common/types.h"
#include "core/framework/request/finish_reason.h"
#include "embedding.pb.h"
#include "models.pb.h"

namespace xllm {
class Options;
class ModelArgs;
}  // namespace xllm

namespace xllm::api_service {

int32_t openai_http_status(StatusCode code);
std::string openai_error_json(StatusCode code,
                              const std::string& message,
                              const std::string& param = "",
                              bool schema_error = false);
nlohmann::json openai_response_json(const proto::ChatResponse& response,
                                    bool named_tool_choice = false,
                                    bool required_tool_choice = false);
nlohmann::json openai_response_json(const proto::CompletionResponse& response,
                                    bool stream = false);
nlohmann::json openai_embedding_json(const proto::EmbeddingResponse& response,
                                     const std::string& encoding_format);
nlohmann::json openai_usage_json(const proto::Usage& usage, bool stream = true);
nlohmann::json openai_models_json(const proto::ModelListResponse& response);
std::string openai_system_fingerprint(const std::string& configuration);
std::string openai_system_fingerprint(const Options& options,
                                      const ModelArgs* model_args);
void set_openai_system_fingerprint(nlohmann::json& response,
                                   const std::string& fingerprint,
                                   bool stream,
                                   bool include_usage);
void set_proto_stop_reason(const StopReason& reason, proto::StopReason* output);

// Shared JSON encoding for connection responses and asynchronous service
// results.
template <typename Request, typename Response>
std::string serialize_openai_response(const Request& request,
                                      const Response& response,
                                      const proto::Usage& stream_usage,
                                      const std::string& fingerprint,
                                      bool delta) {
  nlohmann::json json;
  if constexpr (std::is_same_v<Response, proto::ChatResponse>) {
    const std::string& choice = request.tool_choice();
    const bool named = !choice.empty() && choice != "none" &&
                       choice != "auto" && choice != "required";
    json = openai_response_json(response, named, choice == "required");
  } else {
    static_assert(std::is_same_v<Response, proto::CompletionResponse>);
    json = openai_response_json(response, request.stream());
  }
  const auto& options = request.stream_options();
  set_openai_system_fingerprint(
      json, fingerprint, request.stream(), options.include_usage());
  if (delta && options.include_usage() && options.continuous_usage_stats() &&
      !response.has_usage()) {
    json["usage"] = openai_usage_json(stream_usage);
  }
  return json.dump(-1,
                   ' ',
                   /*ensure_ascii=*/false,
                   nlohmann::json::error_handler_t::replace);
}

}  // namespace xllm::api_service

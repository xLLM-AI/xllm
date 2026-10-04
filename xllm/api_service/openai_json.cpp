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

#include "api_service/openai_json.h"

#include <absl/strings/escaping.h>
#include <absl/strings/str_cat.h>
#include <xxHash/xxhash.h>

#include <algorithm>
#include <bit>

#include "core/common/xllm_build_info.h"
#include "core/util/uuid.h"

namespace xllm::api_service {
namespace {

nlohmann::json stop_reason_json(const proto::StopReason& reason) {
  if (reason.has_token_id()) {
    return reason.token_id();
  }
  if (reason.has_stop_string()) {
    return reason.stop_string();
  }
  return nullptr;
}

nlohmann::json envelope(const std::string& id,
                        const std::string& object,
                        uint32_t created,
                        const std::string& model) {
  return {{"id", id},
          {"object", object},
          {"created", created},
          {"model", model},
          {"choices", nlohmann::json::array()}};
}

template <typename LogProb>
nlohmann::json chat_logprob(const LogProb& value) {
  nlohmann::json bytes = nlohmann::json::array();
  for (unsigned char byte : value.token()) {
    bytes.push_back(byte);
  }
  return {{"token", value.token()},
          {"logprob", std::max(-9999.0f, value.logprob())},
          {"bytes", std::move(bytes)}};
}

nlohmann::json chat_message(const proto::ChatMessage& message,
                            bool delta,
                            bool forced_tool_choice) {
  nlohmann::json json = nlohmann::json::object();
  if (!delta) {
    json = {{"content", nullptr},
            {"refusal", nullptr},
            {"annotations", nullptr},
            {"audio", nullptr},
            {"function_call", nullptr},
            {"reasoning", nullptr}};
  }
  if (message.has_role()) {
    json["role"] = message.role();
  }
  if (message.has_content()) {
    json["content"] = !delta && message.content().empty() &&
                              (!message.reasoning_content().empty() ||
                               !message.tool_calls().empty()) &&
                              !forced_tool_choice
                          ? nlohmann::json(nullptr)
                          : nlohmann::json(message.content());
  }
  if (message.has_reasoning_content() &&
      (!delta || !message.reasoning_content().empty())) {
    json["reasoning"] = message.reasoning_content();
  }
  if (!message.tool_calls().empty()) {
    if (delta && message.content().empty()) {
      json["content"] = nullptr;
    }
    auto calls = nlohmann::json::array();
    for (const auto& call : message.tool_calls()) {
      nlohmann::json item = nlohmann::json::object();
      if (!delta || call.has_id() || !call.function().name().empty()) {
        item["type"] = "function";
      }
      if (call.has_id()) {
        item["id"] = call.id();
      }
      if (delta && call.has_index()) {
        item["index"] = call.index();
      }
      auto function = nlohmann::json::object();
      if (!delta || !call.function().name().empty()) {
        function["name"] = call.function().name();
      }
      function["arguments"] = call.function().arguments();
      item["function"] = std::move(function);
      calls.push_back(std::move(item));
    }
    json["tool_calls"] = std::move(calls);
  }
  return json;
}

}  // namespace

int32_t openai_http_status(StatusCode code) {
  switch (code) {
    case StatusCode::INVALID_ARGUMENT:
      return 400;
    case StatusCode::NOT_FOUND:
      return 404;
    case StatusCode::RATE_LIMITED:
      return 429;
    case StatusCode::UNAVAILABLE:
      return 503;
    case StatusCode::DEADLINE_EXCEEDED:
      return 504;
    default:
      return 500;
  }
}

std::string openai_error_json(StatusCode code,
                              const std::string& message,
                              const std::string& param,
                              bool schema_error) {
  const int32_t http_status = openai_http_status(code);
  std::string type = "InternalServerError";
  if (http_status == 400) {
    type = schema_error ? "Bad Request" : "BadRequestError";
  }
  if (http_status == 404) {
    type = "NotFoundError";
  }
  if (http_status == 429) {
    type = "RateLimitError";
  }
  if (http_status == 503) {
    type = "ServiceUnavailableError";
  }
  if (http_status == 504) {
    type = "TimeoutError";
  }
  return nlohmann::json({{"error",
                          {{"message", message},
                           {"type", type},
                           {"param",
                            param.empty() ? nlohmann::json(nullptr)
                                          : nlohmann::json(param)},
                           {"code", http_status}}}})
      .dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

nlohmann::json openai_usage_json(const proto::Usage& usage, bool stream) {
  nlohmann::json json = {{"prompt_tokens", usage.prompt_tokens()},
                         {"completion_tokens", usage.completion_tokens()},
                         {"total_tokens", usage.total_tokens()}};
  if (!stream) {
    json["prompt_tokens_details"] = nullptr;
  }
  if (usage.has_prompt_tokens_details() &&
      usage.prompt_tokens_details().cached_tokens() > 0) {
    json["prompt_tokens_details"] = {
        {"cached_tokens", usage.prompt_tokens_details().cached_tokens()}};
  }
  return json;
}

nlohmann::json openai_response_json(const proto::ChatResponse& response,
                                    bool named_tool_choice,
                                    bool required_tool_choice) {
  auto json = envelope(
      response.id(), response.object(), response.created(), response.model());
  const bool stream = response.object() == "chat.completion.chunk";
  if (!stream) {
    json.update({{"system_fingerprint", nullptr},
                 {"service_tier", nullptr},
                 {"prompt_logprobs", nullptr},
                 {"prompt_token_ids", nullptr},
                 {"prompt_text", nullptr},
                 {"kv_transfer_params", nullptr}});
  }
  for (const auto& choice : response.choices()) {
    nlohmann::json item = {{"index", choice.index()},
                           {"logprobs", nullptr},
                           {"finish_reason", nullptr}};
    item[stream ? "delta" : "message"] =
        chat_message(stream ? choice.delta() : choice.message(),
                     stream,
                     named_tool_choice || required_tool_choice);
    if (!stream || !choice.delta().has_role()) {
      item["token_ids"] = nullptr;
    }
    if (!stream || choice.has_finish_reason()) {
      item["stop_reason"] = stop_reason_json(choice.stop_reason());
    }
    if (!stream) {
      item["routed_experts"] = nullptr;
    } else if (choice.delta().has_role()) {
      json["prompt_token_ids"] = nullptr;
      json["prompt_text"] = nullptr;
    }
    if (choice.has_finish_reason()) {
      item["finish_reason"] =
          named_tool_choice && choice.finish_reason() == "tool_calls"
              ? "stop"
              : choice.finish_reason();
    }
    if (choice.has_logprobs()) {
      auto content = nlohmann::json::array();
      for (const auto& logprob : choice.logprobs().content()) {
        auto entry = chat_logprob(logprob);
        entry["top_logprobs"] = nlohmann::json::array();
        for (const auto& top : logprob.top_logprobs()) {
          entry["top_logprobs"].push_back(chat_logprob(top));
        }
        content.push_back(std::move(entry));
      }
      item["logprobs"] = {{"content", std::move(content)}};
    }
    json["choices"].push_back(std::move(item));
  }
  if (response.has_usage()) {
    json["usage"] = openai_usage_json(response.usage(), stream);
    if (!stream) {
      json["usage"]["completion_tokens_details"] = nullptr;
    }
  }
  return json;
}

nlohmann::json openai_response_json(const proto::CompletionResponse& response,
                                    bool stream) {
  auto json = envelope(
      response.id(), response.object(), response.created(), response.model());
  if (!stream) {
    json.update({{"system_fingerprint", nullptr},
                 {"service_tier", nullptr},
                 {"kv_transfer_params", nullptr}});
  }
  for (const auto& choice : response.choices()) {
    nlohmann::json item = {
        {"index", choice.index()},
        {"text", choice.text()},
        {"logprobs", nullptr},
        {"finish_reason", nullptr},
        {"stop_reason", stop_reason_json(choice.stop_reason())},
        {"token_ids", nullptr},
        {"prompt_token_ids", nullptr}};
    if (!stream) {
      item["prompt_logprobs"] = nullptr;
      item["routed_experts"] = nullptr;
    }
    if (choice.has_finish_reason()) {
      item["finish_reason"] = choice.finish_reason();
    }
    if (choice.has_logprobs()) {
      const auto& values = choice.logprobs();
      nlohmann::json logprobs = {{"tokens", values.tokens()},
                                 {"token_logprobs", values.token_logprobs()},
                                 {"text_offset", values.text_offset()},
                                 {"top_logprobs", nlohmann::json::array()}};
      for (const auto& top : values.top_logprobs()) {
        auto entry = nlohmann::json::object();
        for (const auto& [token, value] : top.values()) {
          entry[token] = std::max(-9999.0f, value);
        }
        logprobs["top_logprobs"].push_back(std::move(entry));
      }
      item["logprobs"] = std::move(logprobs);
    }
    json["choices"].push_back(std::move(item));
  }
  if (response.has_usage()) {
    json["usage"] = openai_usage_json(response.usage(), stream);
  }
  return json;
}

nlohmann::json openai_embedding_json(const proto::EmbeddingResponse& response,
                                     const std::string& encoding_format) {
  nlohmann::json json = {{"id", response.id()},
                         {"object", "list"},
                         {"created", response.created()},
                         {"model", response.model()},
                         {"data", nlohmann::json::array()}};
  for (const auto& data : response.data()) {
    nlohmann::json embedding = data.embedding();
    if (encoding_format == "base64") {
      std::string bytes;
      bytes.reserve(data.embedding_size() * sizeof(float));
      for (float value : data.embedding()) {
        const uint32_t bits = std::bit_cast<uint32_t>(value);
        for (uint32_t shift = 0; shift < 32; shift += 8) {
          bytes += static_cast<char>((bits >> shift) & 0xff);
        }
      }
      std::string encoded;
      absl::Base64Escape(bytes, &encoded);
      embedding = std::move(encoded);
    }
    json["data"].push_back({{"index", data.index()},
                            {"object", "embedding"},
                            {"embedding", std::move(embedding)}});
  }
  json["usage"] = {{"prompt_tokens", response.usage().prompt_tokens()},
                   {"total_tokens", response.usage().total_tokens()}};
  return json;
}

void set_proto_stop_reason(const StopReason& reason,
                           proto::StopReason* output) {
  output->Clear();
  if (const auto* token = std::get_if<int32_t>(&reason)) {
    output->set_token_id(*token);
  } else if (const auto* text = std::get_if<std::string>(&reason)) {
    output->set_stop_string(*text);
  }
}

std::string openai_system_fingerprint(const std::string& configuration) {
  return absl::StrCat(
      "xllm-",
      XLLM_BUILD_VERSION,
      "-",
      absl::Hex(XXH3_64bits(configuration.data(), configuration.size()),
                absl::kZeroPad16));
}

void set_openai_system_fingerprint(nlohmann::json& response,
                                   const std::string& fingerprint,
                                   bool stream,
                                   bool include_usage) {
  if (fingerprint.empty()) {
    return;
  }
  const auto& choices = response["choices"];
  const bool terminal =
      include_usage
          ? choices.empty() && response.contains("usage")
          : std::any_of(choices.begin(), choices.end(), [](const auto& choice) {
              return !choice["finish_reason"].is_null();
            });
  if (!stream || terminal) {
    response["system_fingerprint"] = fingerprint;
  }
}

nlohmann::json openai_models_json(const proto::ModelListResponse& response) {
  nlohmann::json json = {{"object", "list"}, {"data", nlohmann::json::array()}};
  thread_local ShortUUID uuid;
  for (const auto& model : response.data()) {
    nlohmann::json permission = {{"id", "modelperm-" + uuid.random()},
                                 {"object", "model_permission"},
                                 {"created", model.created()},
                                 {"allow_create_engine", false},
                                 {"allow_sampling", true},
                                 {"allow_logprobs", true},
                                 {"allow_search_indices", false},
                                 {"allow_view", true},
                                 {"allow_fine_tuning", false},
                                 {"organization", "*"},
                                 {"group", nullptr},
                                 {"is_blocking", false}};
    json["data"].push_back(
        {{"id", model.id()},
         {"object", model.object()},
         {"created", model.created()},
         {"owned_by", model.owned_by()},
         {"root",
          model.has_root() ? nlohmann::json(model.root())
                           : nlohmann::json(nullptr)},
         {"parent", nullptr},
         {"max_model_len",
          model.has_max_model_len() ? nlohmann::json(model.max_model_len())
                                    : nlohmann::json(nullptr)},
         {"permission", nlohmann::json::array({std::move(permission)})}});
  }
  return json;
}

}  // namespace xllm::api_service

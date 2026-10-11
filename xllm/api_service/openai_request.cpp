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

#include "api_service/openai_request.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <string_view>
#include <tuple>

namespace xllm::api_service {
namespace {

Status invalid(const std::string& message,
               std::string* error_param = nullptr,
               const char* param = "") {
  if (error_param != nullptr) {
    *error_param = param;
  }
  return Status(StatusCode::INVALID_ARGUMENT, message);
}

Status validate_number(const nlohmann::json& json,
                       const char* name,
                       double minimum,
                       double maximum,
                       bool integer = false,
                       std::string* error_param = nullptr) {
  const auto it = json.find(name);
  if (it == json.end() || it->is_null()) {
    return Status();
  }
  if (!it->is_number() || (integer && !it->is_number_integer())) {
    return invalid(std::string(name) + " must be " +
                       (integer ? "an integer." : "a number."),
                   error_param,
                   name);
  }
  const double value = it->get<double>();
  if (!std::isfinite(value) || value < minimum || value > maximum) {
    return invalid(std::string(name) + " is out of range.", error_param, name);
  }
  return Status();
}

Status normalize_prompts(nlohmann::json& json,
                         const char* field,
                         const char* batch_field,
                         bool& schema_error) {
  if (json.contains(batch_field)) {
    return invalid(std::string(batch_field) + " is internal; use " + field +
                   " instead.");
  }
  auto it = json.find(field);
  if (it == json.end() || it->is_null()) {
    // token_ids remains an xLLM extension for completion requests.
    if (std::string_view(field) == "prompt" && json.contains("token_ids")) {
      return Status();
    }
    return invalid(std::string(field) + " is required.");
  }
  if (it->is_string()) {
    return it->get_ref<const std::string&>().empty() &&
                   !(std::string_view(field) == "prompt" &&
                     json.contains("token_ids"))
               ? invalid(std::string(field) + " must not be empty.")
               : Status();
  }
  if (!it->is_array() || it->empty()) {
    // An empty array passes vLLM's union schema but fails prompt processing.
    schema_error = !it->is_array();
    return invalid(std::string(field) +
                   " must be a non-empty string or array.");
  }
  if (json.contains("token_ids")) {
    return invalid(std::string(field) +
                   " conflicts with internal input fields.");
  }
  const bool tokens = it->front().is_number_integer();
  if (!tokens && std::string_view(field) == "prompt" && it->size() > 1024) {
    return invalid("prompt list exceeds the maximum allowed count of 1024.");
  }
  nlohmann::json inputs = tokens ? nlohmann::json::array({*it}) : *it;
  nlohmann::json prompts = nlohmann::json::array();
  const bool text_batch = inputs.front().is_string();
  for (const auto& input : inputs) {
    if (text_batch) {
      if (!input.is_string() || input.get_ref<const std::string&>().empty()) {
        return invalid(std::string(field) + " must contain non-empty strings.");
      }
      prompts.push_back({{"text", input}});
      continue;
    }
    if (!input.is_array() || input.empty()) {
      return invalid(std::string(field) +
                     " must contain non-empty token arrays.");
    }
    for (const auto& token : input) {
      if (!token.is_number_integer() || token.get<double>() < 0 ||
          token.get<double>() > std::numeric_limits<int32_t>::max()) {
        return invalid("token ids must be non-negative int32 integers.");
      }
    }
    prompts.push_back({{"token_ids", input}});
  }
  if (tokens && std::string_view(field) == "prompt") {
    json["token_ids"] = prompts[0]["token_ids"];
    json["prompt"] = "";
    return Status();
  }
  json[batch_field] = std::move(prompts);
  json.erase(field);
  return Status();
}

Status normalize_generation(nlohmann::json& json,
                            bool chat,
                            std::string* error_param,
                            bool& schema_error) {
  schema_error = false;
  // These controls need engine support; accepting and ignoring them changes
  // the requested sampling or output constraints.
  for (const char* field : {"prompt_logprobs",
                            "structured_outputs",
                            "prompt_embeds",
                            "logprob_token_ids"}) {
    if (json.contains(field) && !json[field].is_null()) {
      return invalid(std::string(field) +
                     " is not supported by the xLLM backend.");
    }
  }
  if (json.value("use_beam_search", nlohmann::json()) == true) {
    return invalid(
        "use_beam_search is not supported; use xLLM beam_width instead.");
  }
  if (chat && (json.value("echo", nlohmann::json()) == true ||
               json.value("continue_final_message", nlohmann::json()) == true ||
               json.value("add_generation_prompt", nlohmann::json()) == false ||
               json.value("parallel_tool_calls", nlohmann::json()) == false)) {
    return invalid(
        "The requested chat rendering or parallel_tool_calls control is not "
        "supported.");
  }
  if (json.contains("response_format") && !json["response_format"].is_null()) {
    const auto& format = json["response_format"];
    if (!format.is_object() || !format.contains("type") ||
        (format["type"] != "text" &&
         !(chat && format["type"] == "json_object"))) {
      return invalid(
          "Unsupported response_format; use text or chat json_object.");
    }
  }
  schema_error = true;
  for (const char* field : {"stream",
                            "echo",
                            "ignore_eos",
                            "skip_special_tokens",
                            "include_stop_str_in_output",
                            "add_special_tokens"}) {
    if (json.contains(field) && !json[field].is_null() &&
        !json[field].is_boolean()) {
      return invalid(std::string(field) + " must be a boolean.");
    }
  }
  if (chat && json.contains("max_completion_tokens") &&
      !json["max_completion_tokens"].is_null()) {
    json["max_tokens"] = json["max_completion_tokens"];
  }
  json.erase("max_completion_tokens");
  if (!chat && (!json.contains("max_tokens") || json["max_tokens"].is_null())) {
    json["max_tokens"] = 16;
  }
  if (!json.contains("temperature") || json["temperature"].is_null()) {
    json["temperature"] = 1.0;
  }
  schema_error = false;
  for (const auto& [field, minimum, maximum, integer] :
       {std::tuple{"max_tokens", 1.0, 4294967295.0, true},
        std::tuple{"n", 1.0, 4294967295.0, true},
        std::tuple{"best_of", 1.0, 4294967295.0, true},
        std::tuple{"temperature",
                   0.0,
                   static_cast<double>(std::numeric_limits<float>::max()),
                   false},
        std::tuple{"min_p", 0.0, 1.0, false},
        std::tuple{"min_tokens", 0.0, 4294967295.0, true},
        std::tuple{"top_p", std::numeric_limits<double>::min(), 1.0, false},
        std::tuple{"top_k", -1.0, 2147483647.0, true},
        std::tuple{"presence_penalty", -2.0, 2.0, false},
        std::tuple{"frequency_penalty", -2.0, 2.0, false},
        std::tuple{"repetition_penalty",
                   std::numeric_limits<double>::min(),
                   static_cast<double>(std::numeric_limits<float>::max()),
                   false}}) {
    const bool has_param = std::string_view(field) == "max_tokens" ||
                           std::string_view(field) == "temperature" ||
                           std::string_view(field) == "top_p";
    Status status = validate_number(json,
                                    field,
                                    minimum,
                                    maximum,
                                    integer,
                                    has_param ? error_param : nullptr);
    if (!status.ok()) {
      return status;
    }
  }
  if (json["temperature"] == 0 && json.contains("n") && !json["n"].is_null() &&
      json["n"].get<uint32_t>() > 1) {
    return invalid("n must be 1 when using greedy sampling, got " +
                   std::to_string(json["n"].get<uint32_t>()) + ".");
  }
  // vLLM raises small positive temperatures before deciding whether to sample.
  if (json["temperature"].get<double>() > 0 &&
      json["temperature"].get<double>() < 0.01) {
    json["temperature"] = 0.01;
  }
  if (json.contains("top_k") && json["top_k"] == 0) {
    json["top_k"] = -1;
  }
  if (json.contains("seed") && !json["seed"].is_null()) {
    const auto& seed = json["seed"];
    if (!seed.is_number_integer() ||
        (seed.is_number_unsigned() &&
         seed.get<uint64_t>() >
             static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))) {
      return invalid("seed must be a signed 64-bit integer.");
    }
  }
  if (json.contains("min_tokens") && !json["min_tokens"].is_null() &&
      json.contains("max_tokens") && !json["max_tokens"].is_null() &&
      json["min_tokens"].get<uint64_t>() > json["max_tokens"].get<uint64_t>()) {
    return invalid("min_tokens must not exceed max_tokens.");
  }
  for (const char* field : {"allowed_token_ids", "stop_token_ids"}) {
    const auto it = json.find(field);
    if (it == json.end() || it->is_null()) {
      continue;
    }
    if (!it->is_array() ||
        (std::string_view(field) == "allowed_token_ids" && it->empty())) {
      return invalid(std::string(field) + " must be an array of token ids.");
    }
    for (const auto& token : *it) {
      if (!token.is_number_integer() || token.get<double>() < 0 ||
          token.get<double>() > std::numeric_limits<int32_t>::max()) {
        return invalid(std::string(field) +
                       " must contain non-negative int32 token ids.");
      }
    }
  }
  const auto words = json.find("bad_words");
  if (words != json.end() && !words->is_null() &&
      (!words->is_array() ||
       !std::all_of(words->begin(), words->end(), [](const auto& word) {
         return word.is_string() &&
                !word.template get_ref<const std::string&>().empty();
       }))) {
    return invalid("bad_words must be an array of non-empty strings.");
  }
  auto bias = json.find("logit_bias");
  if (bias != json.end() && !bias->is_null()) {
    if (!bias->is_object()) {
      return invalid("logit_bias must be an object.");
    }
    for (auto entry = bias->begin(); entry != bias->end(); ++entry) {
      int32_t token = 0;
      const std::string& key = entry.key();
      const auto parsed =
          std::from_chars(key.data(), key.data() + key.size(), token);
      if (parsed.ec != std::errc() || parsed.ptr != key.data() + key.size() ||
          token < 0 || !entry.value().is_number() ||
          !std::isfinite(entry.value().get<double>())) {
        return invalid(
            "logit_bias requires non-negative token ids and finite numeric "
            "biases.");
      }
      entry.value() = std::clamp(entry.value().get<double>(), -100.0, 100.0);
    }
  }
  auto stop = json.find("stop");
  if (stop != json.end() && !stop->is_null()) {
    if (stop->is_string()) {
      *stop = nlohmann::json::array({*stop});
    }
    if (!stop->is_array() ||
        !std::all_of(stop->begin(), stop->end(), [](const auto& value) {
          return value.is_string() &&
                 !value.template get_ref<const std::string&>().empty();
        })) {
      return invalid("stop must be a string or an array of non-empty strings.");
    }
  }
  const bool stream = json.contains("stream") && json["stream"] == true;
  schema_error = true;
  auto stream_options = json.find("stream_options");
  if (stream_options != json.end() && !stream_options->is_null()) {
    if (!stream_options->is_object()) {
      return invalid("stream_options must be an object.");
    }
    if (!stream && !stream_options->empty()) {
      return invalid("Stream options can only be defined when stream=true.",
                     error_param,
                     "stream_options");
    }
    for (const char* field : {"include_usage", "continuous_usage_stats"}) {
      if (stream_options->contains(field) &&
          !(*stream_options)[field].is_null() &&
          !(*stream_options)[field].is_boolean()) {
        return invalid(std::string("stream_options.") + field +
                       " must be a boolean.");
      }
    }
  }
  if (chat && json.contains("logprobs") && !json["logprobs"].is_null() &&
      !json["logprobs"].is_boolean()) {
    return invalid("logprobs must be a boolean.");
  }
  Status status = validate_number(json,
                                  chat ? "top_logprobs" : "logprobs",
                                  -1,
                                  2000,
                                  /*integer=*/true,
                                  error_param);
  if (!status.ok()) {
    return status;
  }
  if (chat && json.contains("top_logprobs") &&
      json["top_logprobs"].is_number() &&
      json["top_logprobs"].get<int32_t>() != 0 &&
      json.value("logprobs", nlohmann::json()) != true) {
    return invalid("when using top_logprobs, logprobs must be true.",
                   error_param,
                   "top_logprobs");
  }
  return Status();
}

Status normalize_chat(nlohmann::json& json,
                      std::string* error_param,
                      bool& schema_error) {
  if (!json.contains("messages") || !json["messages"].is_array() ||
      json["messages"].empty()) {
    schema_error = !json.contains("messages") || !json["messages"].is_array();
    return invalid("messages must be a non-empty array.");
  }
  for (auto& message : json["messages"]) {
    if (!message.is_object() || !message.contains("role") ||
        !message["role"].is_string()) {
      return invalid("Each message must contain a string role.");
    }
    if (message.contains("reasoning") && !message["reasoning"].is_null()) {
      message["reasoning_content"] = message["reasoning"];
      message.erase("reasoning");
    }
  }
  if (json.contains("tool_choice") && json["tool_choice"].is_null()) {
    json.erase("tool_choice");
  }
  if (json.contains("tools") && !json["tools"].is_null() &&
      (!json["tools"].is_array() || json["tools"].empty())) {
    return invalid("tools must be a non-empty array.");
  }
  if (json.contains("tools") && json["tools"].is_array()) {
    for (auto& tool : json["tools"]) {
      if (!tool.is_object() || !tool.contains("function") ||
          !tool["function"].is_object() || !tool["function"].contains("name") ||
          !tool["function"]["name"].is_string() ||
          tool["function"]["name"].get_ref<const std::string&>().empty()) {
        return invalid(
            "Each tool must define a function with a non-empty name.");
      }
      if (!tool.contains("type")) {
        tool["type"] = "function";
      }
      if (tool["type"] != "function") {
        return invalid("Only function tools are supported.");
      }
    }
  }
  const auto choice = json.find("tool_choice");
  if (choice == json.end() || *choice == "none") {
    return Status();
  }
  if (!json.contains("tools") || !json["tools"].is_array() ||
      json["tools"].empty()) {
    return invalid("When using tool_choice, tools must be set.",
                   error_param,
                   "tool_choice");
  }
  if (choice->is_string()) {
    return (*choice == "auto" || *choice == "required")
               ? Status()
               : invalid(
                     "tool_choice must be none, auto, required, or a named "
                     "function.",
                     error_param,
                     "tool_choice");
  }
  if (!choice->is_object() ||
      choice->value("type", nlohmann::json()) != "function") {
    return invalid(
        "tool_choice must name a function.", error_param, "tool_choice");
  }
  if (!choice->contains("function") || !(*choice)["function"].is_object()) {
    return invalid("tool_choice must name a function.",
                   error_param,
                   "tool_choice.function");
  }
  const auto name = (*choice)["function"].value("name", nlohmann::json());
  if (!name.is_string() || name.get_ref<const std::string&>().empty()) {
    return invalid("tool_choice must name a function.",
                   error_param,
                   "tool_choice.function.name");
  }
  const bool found = std::any_of(
      json["tools"].begin(), json["tools"].end(), [&name](const auto& tool) {
        return tool.is_object() && tool.contains("function") &&
               tool["function"].is_object() &&
               tool["function"].value("name", nlohmann::json()) == name;
      });
  return found ? Status()
               : invalid("tool_choice names a function not present in tools.",
                         error_param,
                         "tool_choice");
}

}  // namespace

std::pair<Status, std::string> normalize_openai_request(
    std::string body,
    OpenAIEndpoint endpoint,
    const std::string& default_model,
    std::string* error_param,
    bool* schema_error) {
  bool local_schema_error = true;
  bool& is_schema_error =
      schema_error != nullptr ? *schema_error : local_schema_error;
  is_schema_error = true;
  if (error_param != nullptr) {
    error_param->clear();
  }
  try {
    auto json = nlohmann::json::parse(body);
    if (!json.is_object()) {
      return {invalid("Request body must be a JSON object."), ""};
    }
    if (!json.contains("model") || json["model"].is_null()) {
      json["model"] = default_model;
    }
    if (!json["model"].is_string() ||
        json["model"].get_ref<const std::string&>().empty()) {
      return {
          invalid("model must be a non-empty string.", error_param, "model"),
          ""};
    }
    Status status;
    if (endpoint == OpenAIEndpoint::CHAT ||
        endpoint == OpenAIEndpoint::COMPLETION) {
      status = normalize_generation(
          json, endpoint == OpenAIEndpoint::CHAT, error_param, is_schema_error);
      if (status.ok()) {
        is_schema_error = true;
        status =
            endpoint == OpenAIEndpoint::CHAT
                ? normalize_chat(json, error_param, is_schema_error)
                : normalize_prompts(json, "prompt", "prompts", is_schema_error);
      }
    } else {
      if (endpoint == OpenAIEndpoint::EMBEDDING) {
        status = normalize_prompts(json, "input", "inputs", is_schema_error);
      }
      if (status.ok()) {
        status = validate_number(
            json, "dimensions", 1, 2147483647, /*integer=*/true);
      }
      const auto format = json.find("encoding_format");
      if (status.ok() && format != json.end() && !format->is_null() &&
          *format != "float" && *format != "base64" &&
          !(endpoint == OpenAIEndpoint::MM_EMBEDDING && *format == "binary")) {
        status = invalid("encoding_format must be float or base64.");
      }
    }
    return status.ok() ? std::make_pair(Status(), json.dump())
                       : std::make_pair(status, std::string());
  } catch (const nlohmann::json::exception& error) {
    is_schema_error = true;
    return {invalid("Invalid request JSON: " + std::string(error.what())), ""};
  }
}

}  // namespace xllm::api_service

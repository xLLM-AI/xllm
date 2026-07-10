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

#include "api_service/chat_json_parser.h"

#include <glog/logging.h>

#include <nlohmann/json.hpp>

namespace xllm {
namespace {

Status normalize_tool_choice(nlohmann::json* json, bool* modified) {
  if (!json->contains("tool_choice")) {
    return Status();
  }

  nlohmann::json& tool_choice = (*json)["tool_choice"];
  if (tool_choice.is_string()) {
    return Status();
  }
  if (!tool_choice.is_object()) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "tool_choice must be a string or an object.");
  }

  if (!tool_choice.contains("type") || !tool_choice["type"].is_string()) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "Object tool_choice must contain a string 'type' field.");
  }

  const std::string type = tool_choice["type"].get<std::string>();
  if (type != "function") {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "Unsupported object tool_choice type: " + type);
  }

  if (!tool_choice.contains("function") ||
      !tool_choice["function"].is_object()) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "Function tool_choice must contain a function object.");
  }

  const nlohmann::json& function = tool_choice["function"];
  if (!function.contains("name") || !function["name"].is_string() ||
      function["name"].get_ref<const std::string&>().empty()) {
    return Status(
        StatusCode::INVALID_ARGUMENT,
        "Function tool_choice must contain a non-empty function.name string.");
  }

  nlohmann::json normalized_tool_choice = {
      {"type", "function"},
      {"function", {{"name", function["name"].get<std::string>()}}}};
  tool_choice = normalized_tool_choice.dump();
  *modified = true;
  return Status();
}

}  // namespace

const ChatJsonParser& ChatJsonParser::get(ServingMode mode) {
  if (mode == ServingMode::VLM) {
    static const VlmChatJsonParser k_vlm_parser;
    return k_vlm_parser;
  }
  static const LlmChatJsonParser k_llm_parser;
  return k_llm_parser;
}

const ChatJsonParser& ChatJsonParser::anthropic() {
  static const AnthropicChatJsonParser k_anthropic_parser;
  return k_anthropic_parser;
}

std::pair<Status, std::string> VlmChatJsonParser::preprocess(
    std::string json_str) const {
  try {
    auto json = nlohmann::json::parse(json_str);
    bool modified = false;
    Status status = normalize_tool_choice(&json, &modified);
    if (!status.ok()) {
      return {status, ""};
    }

    if (!json.contains("messages") || !json["messages"].is_array()) {
      return modified ? std::make_pair(Status(), json.dump())
                      : std::make_pair(Status(), std::move(json_str));
    }

    for (auto& msg : json["messages"]) {
      if (!msg.is_object()) {
        return {Status(StatusCode::INVALID_ARGUMENT,
                       "Message in 'messages' array must be an object."),
                ""};
      }
      if (msg.contains("content") && msg["content"].is_string()) {
        nlohmann::json content = nlohmann::json::array();
        content.push_back(
            {{"type", "text"},
             {"text", msg["content"].get_ref<const std::string&>()}});
        msg["content"] = std::move(content);
        modified = true;
      }
    }

    return modified ? std::make_pair(Status(), json.dump())
                    : std::make_pair(Status(), std::move(json_str));
  } catch (const nlohmann::json::exception& e) {
    return {Status(StatusCode::INVALID_ARGUMENT,
                   "Invalid JSON format: " + std::string(e.what())),
            ""};
  } catch (const std::exception& e) {
    LOG(ERROR) << "Exception during VLM JSON preprocessing: " << e.what();
    return {Status(StatusCode::UNKNOWN,
                   "Internal server error during JSON processing."),
            ""};
  }
}

std::pair<Status, std::string> LlmChatJsonParser::preprocess(
    std::string json_str) const {
  try {
    auto json = nlohmann::json::parse(json_str);
    bool modified = false;
    Status status = normalize_tool_choice(&json, &modified);
    if (!status.ok()) {
      return {status, ""};
    }

    if (!json.contains("messages") || !json["messages"].is_array()) {
      return modified ? std::make_pair(Status(), json.dump())
                      : std::make_pair(Status(), std::move(json_str));
    }

    for (auto& msg : json["messages"]) {
      if (!msg.is_object()) {
        return {Status(StatusCode::INVALID_ARGUMENT,
                       "Message in 'messages' array must be an object."),
                ""};
      }
      if (msg.contains("content") && msg["content"].is_array()) {
        for (const auto& item : msg["content"]) {
          if (!item.is_object()) {
            return {Status(StatusCode::INVALID_ARGUMENT,
                           "Content array item must be an object."),
                    ""};
          }
          if (!item.contains("type") || item["type"] != "text") {
            return {Status(StatusCode::INVALID_ARGUMENT,
                           "Non-text content (e.g., image_url) requires "
                           "multimodal backend (-backend vlm)"),
                    ""};
          }
          if (!item.contains("text") || !item["text"].is_string()) {
            return {Status(StatusCode::INVALID_ARGUMENT,
                           "Missing or invalid 'text' field in content item."),
                    ""};
          }
        }

        size_t total_size = 0;
        size_t num_items = msg["content"].size();
        for (const auto& item : msg["content"]) {
          total_size += item["text"].get_ref<const std::string&>().size();
        }
        if (num_items > 1) {
          total_size += num_items - 1;
        }

        std::string combined_text;
        combined_text.reserve(total_size);
        bool first = true;
        for (const auto& item : msg["content"]) {
          if (!first) {
            combined_text += '\n';
          }
          combined_text += item["text"].get_ref<const std::string&>();
          first = false;
        }
        msg["content"] = combined_text;
        modified = true;
      }
    }
    return modified ? std::make_pair(Status(), json.dump())
                    : std::make_pair(Status(), std::move(json_str));
  } catch (const nlohmann::json::exception& e) {
    return {Status(StatusCode::INVALID_ARGUMENT,
                   "Invalid JSON format: " + std::string(e.what())),
            ""};
  } catch (const std::exception& e) {
    LOG(ERROR) << "Exception during JSON preprocessing: " << e.what();
    return {Status(StatusCode::UNKNOWN,
                   "Internal server error during JSON processing."),
            ""};
  }
}

std::pair<Status, std::string> AnthropicChatJsonParser::preprocess(
    std::string json_str) const {
  try {
    auto j = nlohmann::json::parse(json_str);

    if (j.contains("messages") && j["messages"].is_array()) {
      for (auto& msg : j["messages"]) {
        if (!msg.contains("content")) {
          continue;
        }
        auto& content = msg["content"];
        if (content.is_string()) {
          msg["content_string"] = content.get<std::string>();
          msg.erase("content");
        } else if (content.is_array()) {
          for (auto& block : content) {
            if (!block.is_object() ||
                block.value("type", "") != "tool_result") {
              continue;
            }
            if (block.contains("tool_use_id") && !block.contains("id")) {
              block["id"] = block["tool_use_id"];
            }
            block.erase("tool_use_id");
            if (!block.contains("content")) {
              continue;
            }
            auto& tool_content = block["content"];
            if (tool_content.is_string()) {
              block["content_string"] = tool_content.get<std::string>();
              block.erase("content");
            } else if (tool_content.is_array()) {
              block["content_list"] = {{"items", tool_content}};
              block.erase("content");
            }
          }
          nlohmann::json content_blocks;
          content_blocks["blocks"] = content;
          msg["content_blocks"] = content_blocks;
          msg.erase("content");
        }
      }
    }

    if (j.contains("system")) {
      auto& system = j["system"];
      if (system.is_string()) {
        j["system_string"] = system.get<std::string>();
        j.erase("system");
      } else if (system.is_array()) {
        nlohmann::json system_blocks;
        system_blocks["blocks"] = system;
        j["system_blocks"] = system_blocks;
        j.erase("system");
      }
    }

    return {Status(), j.dump()};
  } catch (const nlohmann::json::exception& e) {
    return {Status(StatusCode::INVALID_ARGUMENT,
                   "Invalid JSON format: " + std::string(e.what())),
            ""};
  } catch (const std::exception& e) {
    LOG(ERROR) << "Exception during Anthropic JSON preprocessing: " << e.what();
    return {Status(StatusCode::UNKNOWN,
                   "Internal server error during JSON processing."),
            ""};
  }
}

}  // namespace xllm

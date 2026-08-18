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
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/common/message.h"
#include "core/common/types.h"
#include "framework/tokenizer/tokenizer_args.h"

namespace xllm {

enum class ChatTemplateGenerationMode : uint8_t {
  CHAT = 0,
  REASONING = 1,
  UNKNOWN = 2,
};

struct ChatTemplateRenderResult final {
  std::string prompt;
  ChatTemplateGenerationMode generation_mode =
      ChatTemplateGenerationMode::UNKNOWN;
};

class ChatTemplate {
 public:
  virtual ~ChatTemplate() = default;

  virtual std::optional<std::string> apply(
      const ChatMessages& messages) const = 0;

  virtual std::optional<std::string> apply(
      const ChatMessages& messages,
      const std::vector<xllm::JsonTool>& json_tools,
      const nlohmann::ordered_json& chat_template_kwargs) const = 0;

  std::optional<ChatTemplateRenderResult> apply_with_generation_mode(
      const ChatMessages& messages,
      const std::vector<xllm::JsonTool>& json_tools,
      const nlohmann::ordered_json& chat_template_kwargs) const {
    std::optional<std::string> prompt =
        apply(messages, json_tools, chat_template_kwargs);
    if (!prompt.has_value()) {
      return std::nullopt;
    }
    const ChatTemplateGenerationMode generation_mode =
        generation_mode_from_prompt(prompt.value());
    return ChatTemplateRenderResult{std::move(prompt.value()), generation_mode};
  }

  static ChatTemplateGenerationMode generation_mode_from_prompt(
      std::string_view prompt) {
    constexpr std::string_view kJsonWhitespace = " \t\n\r";
    while (!prompt.empty() &&
           kJsonWhitespace.find(prompt.back()) != std::string_view::npos) {
      prompt.remove_suffix(1);
    }
    constexpr std::string_view kThinkingMarker = "<think>";
    constexpr std::string_view kChatMarker = "</think>";
    if (prompt.ends_with(kThinkingMarker)) {
      return ChatTemplateGenerationMode::REASONING;
    }
    if (prompt.ends_with(kChatMarker)) {
      return ChatTemplateGenerationMode::CHAT;
    }
    return ChatTemplateGenerationMode::UNKNOWN;
  }

  static std::unique_ptr<ChatTemplate> create(
      const TokenizerArgs& tokenizer_args,
      const std::string& model_type);
};

}  // namespace xllm

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

#pragma once

#include <string>
#include <utility>

#include "api_service/serving_mode.h"
#include "core/common/types.h"

namespace xllm {

// Normalizes OpenAI-style chat JSON before protobuf parsing. LLM backends
// collapse text-only content arrays into a single string; VLM backends pass
// JSON through for downstream multimodal handling.
class ChatJsonParser {
 public:
  virtual ~ChatJsonParser() = default;

  [[nodiscard]] virtual std::pair<Status, std::string> preprocess(
      std::string json_str) const = 0;

  // Returns the singleton parser for the given serving mode.
  // LLM/REC → LlmChatJsonParser, VLM → VlmChatJsonParser.
  static const ChatJsonParser& get(ServingMode mode);

  // Returns the Anthropic protocol parser (separate from serving mode).
  static const ChatJsonParser& anthropic();
};

// Text-only backend: combines array content items of type "text" into one
// string; rejects non-text parts (e.g. image_url).
class LlmChatJsonParser final : public ChatJsonParser {
 public:
  std::pair<Status, std::string> preprocess(
      std::string json_str) const override;
};

// Multimodal backend: no preprocessing; array content stays as-is.
class VlmChatJsonParser final : public ChatJsonParser {
 public:
  std::pair<Status, std::string> preprocess(
      std::string json_str) const override;
};

// Anthropic Messages API: remaps "content" (string|array) to
// "content_string"/"content_blocks", tool_result fields to protobuf names,
// and "system" (string|array) to "system_string"/"system_blocks" for
// protobuf compatibility.
class AnthropicChatJsonParser final : public ChatJsonParser {
 public:
  std::pair<Status, std::string> preprocess(
      std::string json_str) const override;
};

}  // namespace xllm

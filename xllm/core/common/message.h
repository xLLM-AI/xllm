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

#include <torch/torch.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "embedding.pb.h"

namespace xllm {
using Embedding = xllm::proto::Embedding;

struct ImageURL {
  std::string url;
  std::unordered_map<std::string, std::string> headers;
};

struct VideoURL {
  std::string url;
  std::unordered_map<std::string, std::string> headers;
};

struct AudioURL {
  std::string url;
  std::unordered_map<std::string, std::string> headers;
};

struct MMContent {
  MMContent(const std::string& type, const std::string& text)
      : type(type), text(std::move(text)) {}
  MMContent(const std::string& type, const ImageURL& image_url)
      : type(type), image_url(std::move(image_url)) {}
  MMContent(const std::string& type, const VideoURL& video_url)
      : type(type), video_url(std::move(video_url)) {}
  MMContent(const std::string& type, const AudioURL& audio_url)
      : type(type), audio_url(std::move(audio_url)) {}
  MMContent(const std::string& type, const Embedding& embedding)
      : type(type), embedding(embedding) {}

  std::string type;

  std::string text;
  ImageURL image_url;

  VideoURL video_url;
  AudioURL audio_url;

  Embedding embedding;
};
using MMContentVec = std::vector<MMContent>;

struct Message {
  using Content = std::variant<std::string, MMContentVec>;

  struct ToolCall {
    std::string id;
    std::string type;
    struct Function {
      std::string name;
      std::string arguments;
    } function;
  };
  using ToolCallVec = std::vector<ToolCall>;

  Message(const std::string& role, const std::string& content)
      : role(role), content(content) {}

  Message(const std::string& role, const MMContentVec& content)
      : role(role), content(std::move(content)) {}

  int calc_count(const std::string& type) {
    if (std::holds_alternative<std::string>(content)) {
      if (type == "text") {
        return 1;
      } else {
        return 0;
      }
    }

    const auto& mmc = std::get<MMContentVec>(content);
    int count = 0;
    for (const auto& item : mmc) {
      if (item.type == type) {
        ++count;
      }
    }

    return count;
  }

  bool has_mm_content() const {
    if (std::holds_alternative<std::string>(content)) {
      return false;
    }
    for (const auto& item : std::get<MMContentVec>(content)) {
      if (item.type != "text") {
        return true;
      }
    }
    return false;
  }

  std::string role;
  Content content;

  // Additional fields for tool calls and reasoning
  std::optional<std::string> tool_call_id;
  std::optional<std::string> reasoning_content;
  std::optional<ToolCallVec> tool_calls;
};

using ChatMessages = std::vector<Message>;

}  // namespace xllm

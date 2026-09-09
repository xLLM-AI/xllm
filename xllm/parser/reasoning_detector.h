/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <optional>
#include <string>

namespace xllm {

struct ReasoningResult {
  std::optional<std::string> normal_text = std::nullopt;
  std::optional<std::string> reasoning_text = std::nullopt;

  ReasoningResult() = default;

  ReasoningResult(std::optional<std::string> normal,
                  std::optional<std::string> reasoning)
      : normal_text(normal), reasoning_text(reasoning) {}
};

class ReasoningDetector {
 public:
  ReasoningDetector(const std::string& think_start_token,
                    const std::string& think_end_token,
                    bool force_reasoning = false,
                    bool stream_reasoning = true);

  ~ReasoningDetector() = default;

  // Detects and parses reasoning sections in the provided text. Returns both
  // reasoning content and normal text separately.
  ReasoningResult detect_and_parse(std::string& text);

  // Streaming incremental parsing for reasoning content.
  // Handles partial reasoning tags and content.
  //
  // If stream_reasoning is False:
  //     Accumulates reasoning content until the end tag is found
  // If stream_reasoning is True:
  //     Streams reasoning content as it arrives
  ReasoningResult parse_streaming_increment(std::string& new_text);

 protected:
  std::string think_start_token_;
  std::string think_end_token_;
  bool in_reasoning_;
  bool stream_reasoning_;
  std::string buffer_ = "";
  bool stripped_think_start_ = false;
};
}  // namespace xllm

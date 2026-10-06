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

#include <memory>
#include <optional>
#include <string>

#include "parser/detector_registry.h"

namespace xllm {

class ReasoningParser {
 public:
  explicit ReasoningParser(const std::string& model_type,
                           bool stream_reasoning = true,
                           bool force_reasoning = false,
                           std::optional<bool> initial_reasoning = std::nullopt,
                           bool lossless = false);

  // Non-streaming call: one-time parsing
  ReasoningResult parse_non_stream(const std::string& text);
  // Streaming call: incremental parsing
  ReasoningResult parse_stream_chunk(const std::string& chunk_text);
  ReasoningResult finish_stream();

  const std::string& start_marker() const { return detector_->start_marker(); }
  const std::string& end_marker() const { return detector_->end_marker(); }
  bool initially_in_reasoning() const {
    return detector_->initially_in_reasoning();
  }

  static std::string get_parser_auto(const std::string& parser,
                                     const std::string& model_type);

 private:
  std::unique_ptr<ReasoningDetector> detector_;
};
}  // namespace xllm

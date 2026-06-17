/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "reasoning_detector.h"

#include <glog/logging.h>

#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"

namespace xllm {
namespace {
std::string absl_trim(absl::string_view str) {
  absl::string_view trimmed = absl::StripAsciiWhitespace(str);
  return std::string(trimmed);
}
}  // namespace

ReasoningDetector::ReasoningDetector(const std::string& think_start_token,
                                     const std::string& think_end_token,
                                     bool force_reasoning,
                                     bool stream_reasoning)
    : think_start_token_(think_start_token),
      think_end_token_(think_end_token),
      in_reasoning_(force_reasoning),
      stream_reasoning_(stream_reasoning) {}

ReasoningResult ReasoningDetector::detect_and_parse(std::string& text) {
  bool in_reasoning =
      in_reasoning_ || absl::StrContains(text, think_start_token_);

  if (!in_reasoning) {
    return ReasoningResult(text, std::nullopt);
  }

  std::string processed_text =
      absl::StrReplaceAll(text, {{think_start_token_, ""}});
  processed_text = absl_trim(processed_text);

  if (!absl::StrContains(processed_text, think_end_token_)) {
    return ReasoningResult(std::nullopt, processed_text);
  }

  std::vector<absl::string_view> parts =
      absl::StrSplit(processed_text, absl::MaxSplits(think_end_token_, 1));

  std::string reasoning_text = std::string(parts[0]);
  std::string normal_text = parts.size() > 1 ? absl_trim(parts[1]) : "";

  return ReasoningResult(normal_text, reasoning_text);
}

ReasoningResult ReasoningDetector::parse_streaming_increment(
    std::string& new_text) {
  buffer_.append(new_text.data(), new_text.size());
  std::string current_text = buffer_;

  // If the current text is a prefix of the think token, keep buffering
  bool is_start_prefix = absl::StartsWith(think_start_token_, current_text) &&
                         (think_start_token_ != current_text);
  bool is_end_prefix = absl::StartsWith(think_end_token_, current_text) &&
                       (think_end_token_ != current_text);

  if (is_start_prefix || is_end_prefix) {
    return ReasoningResult();
  }

  // Strip `<think>` token if present
  if (!stripped_think_start_ &&
      absl::StrContains(current_text, think_start_token_)) {
    absl::StrReplaceAll(
        {{absl::string_view(think_start_token_), absl::string_view()}},
        &current_text);
    stripped_think_start_ = true;
    in_reasoning_ = true;
  }

  // Handle end of reasoning block
  if (in_reasoning_ && absl::StrContains(current_text, think_end_token_)) {
    std::vector<absl::string_view> parts =
        absl::StrSplit(current_text, absl::MaxSplits(think_end_token_, 1));

    std::string reasoning_text = std::string(parts[0]);
    std::string normal_text = parts.size() > 1 ? absl_trim(parts[1]) : "";

    buffer_.clear();
    in_reasoning_ = false;

    return ReasoningResult(normal_text, reasoning_text);
  }

  // Continue with reasoning content
  if (in_reasoning_) {
    if (stream_reasoning_) {
      buffer_.clear();
      return ReasoningResult(std::nullopt, std::string(current_text));
    } else {
      return ReasoningResult();
    }
  }

  // If we're not in a reasoning block return as normal text
  if (!in_reasoning_) {
    buffer_.clear();
    return ReasoningResult(std::string(current_text), std::nullopt);
  }

  return ReasoningResult();
}
}  // namespace xllm
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

#include "parser/reasoning_detector.h"

#include <glog/logging.h>

#include <algorithm>

#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"

namespace xllm {
namespace {

std::string trim_reasoning_text(absl::string_view text) {
  return std::string(absl::StripAsciiWhitespace(text));
}

}  // namespace

ReasoningDetector::ReasoningDetector(const std::string& think_start_token,
                                     const std::string& think_end_token,
                                     bool force_reasoning,
                                     bool stream_reasoning,
                                     bool lossless)
    : think_start_token_(think_start_token),
      think_end_token_(think_end_token),
      in_reasoning_(force_reasoning),
      initial_reasoning_(force_reasoning),
      stream_reasoning_(stream_reasoning),
      lossless_(lossless) {
  CHECK(!think_start_token_.empty());
  CHECK(!think_end_token_.empty());
}

void ReasoningDetector::set_initial_reasoning(bool in_reasoning) {
  CHECK(!parsing_started_) << "Cannot change reasoning state after parsing";
  in_reasoning_ = in_reasoning;
  initial_reasoning_ = in_reasoning;
}

void ReasoningDetector::set_lossless(bool lossless) {
  CHECK(!parsing_started_) << "Cannot change reasoning profile after parsing";
  lossless_ = lossless;
}

ReasoningResult ReasoningDetector::detect_and_parse(std::string& text) {
  if (!lossless_) {
    const bool in_reasoning =
        in_reasoning_ || absl::StrContains(text, think_start_token_);
    if (!in_reasoning) {
      return ReasoningResult(text, std::nullopt);
    }
    std::string processed_text =
        absl::StrReplaceAll(text, {{think_start_token_, ""}});
    processed_text = trim_reasoning_text(processed_text);
    if (!absl::StrContains(processed_text, think_end_token_)) {
      return ReasoningResult(std::nullopt, processed_text);
    }
    const std::vector<absl::string_view> parts =
        absl::StrSplit(processed_text, absl::MaxSplits(think_end_token_, 1));
    const std::string reasoning_text(parts[0]);
    const std::string normal_text =
        parts.size() > 1 ? trim_reasoning_text(parts[1]) : "";
    return ReasoningResult(normal_text, reasoning_text);
  }

  ReasoningDetector detector(think_start_token_,
                             think_end_token_,
                             in_reasoning_,
                             stream_reasoning_,
                             /*lossless=*/true);
  ReasoningResult result = detector.parse_streaming_increment(text);
  ReasoningResult tail = detector.finish_stream();
  auto append = [](std::optional<std::string>& target,
                   const std::optional<std::string>& value) {
    if (value.has_value()) {
      if (!target.has_value()) {
        target.emplace();
      }
      target->append(value.value());
    }
  };
  append(result.normal_text, tail.normal_text);
  append(result.reasoning_text, tail.reasoning_text);
  return result;
}

ReasoningResult ReasoningDetector::parse_streaming_increment(
    std::string& new_text) {
  parsing_started_ = true;
  if (lossless_) {
    return parse_lossless_increment(new_text);
  }

  buffer_.append(new_text);
  std::string current_text = buffer_;
  const bool is_start_prefix =
      absl::StartsWith(think_start_token_, current_text) &&
      think_start_token_ != current_text;
  const bool is_end_prefix = absl::StartsWith(think_end_token_, current_text) &&
                             think_end_token_ != current_text;
  if (is_start_prefix || is_end_prefix) {
    return ReasoningResult();
  }
  if (!stripped_think_start_ &&
      absl::StrContains(current_text, think_start_token_)) {
    absl::StrReplaceAll(
        {{absl::string_view(think_start_token_), absl::string_view()}},
        &current_text);
    stripped_think_start_ = true;
    in_reasoning_ = true;
  }
  if (in_reasoning_ && absl::StrContains(current_text, think_end_token_)) {
    const std::vector<absl::string_view> parts =
        absl::StrSplit(current_text, absl::MaxSplits(think_end_token_, 1));
    const std::string reasoning_text(parts[0]);
    const std::string normal_text =
        parts.size() > 1 ? trim_reasoning_text(parts[1]) : "";
    buffer_.clear();
    in_reasoning_ = false;
    return ReasoningResult(normal_text, reasoning_text);
  }
  if (in_reasoning_) {
    if (stream_reasoning_) {
      buffer_.clear();
      return ReasoningResult(std::nullopt, current_text);
    }
    return ReasoningResult();
  }
  buffer_.clear();
  return ReasoningResult(current_text, std::nullopt);
}

ReasoningResult ReasoningDetector::parse_lossless_increment(
    std::string& new_text) {
  buffer_.append(new_text);
  ReasoningResult result;
  auto append = [&](const std::string& text) {
    if (text.empty()) {
      return;
    }
    if (in_reasoning_ && !stream_reasoning_) {
      reasoning_buffer_.append(text);
      return;
    }
    std::optional<std::string>& target =
        in_reasoning_ ? result.reasoning_text : result.normal_text;
    if (!target.has_value()) {
      target.emplace();
    }
    target->append(text);
  };

  while (!buffer_.empty()) {
    const size_t start = stripped_think_start_
                             ? std::string::npos
                             : buffer_.find(think_start_token_);
    const size_t end =
        in_reasoning_ ? buffer_.find(think_end_token_) : std::string::npos;
    const size_t marker = std::min(start, end);
    if (marker != std::string::npos) {
      append(buffer_.substr(0, marker));
      if (marker == end) {
        buffer_.erase(0, marker + think_end_token_.size());
        if (!stream_reasoning_ && !reasoning_buffer_.empty()) {
          result.reasoning_text = std::move(reasoning_buffer_);
          reasoning_buffer_.clear();
        }
        in_reasoning_ = false;
        stripped_think_start_ = true;
      } else {
        buffer_.erase(0, marker + think_start_token_.size());
        stripped_think_start_ = true;
        in_reasoning_ = true;
      }
      continue;
    }

    size_t held = 0;
    auto hold_marker_suffix = [&](const std::string& token) {
      const size_t limit = std::min(buffer_.size(), token.size() - 1);
      for (size_t length = limit; length > held; --length) {
        if (buffer_.compare(
                buffer_.size() - length, length, token, 0, length) == 0) {
          held = length;
          break;
        }
      }
    };
    if (!stripped_think_start_) {
      hold_marker_suffix(think_start_token_);
    }
    if (in_reasoning_) {
      hold_marker_suffix(think_end_token_);
    }
    const size_t available = buffer_.size() - held;
    append(buffer_.substr(0, available));
    buffer_.erase(0, available);
    break;
  }
  return result;
}

ReasoningResult ReasoningDetector::finish_stream() {
  if (!in_reasoning_) {
    std::string text = std::move(buffer_);
    buffer_.clear();
    return text.empty() ? ReasoningResult()
                        : ReasoningResult(std::move(text), std::nullopt);
  }
  reasoning_buffer_.append(buffer_);
  buffer_.clear();
  std::string text = std::move(reasoning_buffer_);
  reasoning_buffer_.clear();
  return text.empty() ? ReasoningResult()
                      : ReasoningResult(std::nullopt, std::move(text));
}
}  // namespace xllm

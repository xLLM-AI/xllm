/* Copyright 2025-2026 The xLLM Authors.

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

#include "function_call/qwen25_detector.h"

#include <algorithm>
#include <iostream>
#include <string_view>

namespace xllm {
namespace function_call {

Qwen25Detector::Qwen25Detector() : BaseFormatDetector() {
  bot_token_ = "<tool_call>\n";
  eot_token_ = "\n</tool_call>";
  tool_call_separator_ = "\n";

  std::string pattern = bot_token_ + "([\\s\\S]*?)" + eot_token_;
  tool_call_regex_ = std::regex(
      pattern,
      std::regex_constants::ECMAScript | std::regex_constants::optimize);
}

bool Qwen25Detector::has_tool_call(const std::string& text) {
  return text.find(bot_token_) != std::string::npos;
}

std::string_view Qwen25Detector::trim_whitespace(std::string_view str) const {
  const char* whitespace = " \t\n\r";

  size_t start = str.find_first_not_of(whitespace);
  if (start == std::string_view::npos) {
    return std::string_view{};
  }

  size_t end = str.find_last_not_of(whitespace);

  return str.substr(start, end - start + 1);
}

std::vector<std::pair<size_t, size_t>> Qwen25Detector::find_tool_call_ranges(
    const std::string& text) const {
  std::vector<std::pair<size_t, size_t>> ranges;
  ranges.reserve(4);

  size_t search_pos = 0;
  const size_t bot_token_len = bot_token_.length();
  const size_t eot_token_len = eot_token_.length();

  while (search_pos < text.length()) {
    size_t start_pos = text.find(bot_token_, search_pos);
    if (start_pos == std::string::npos) {
      break;
    }

    size_t content_start = start_pos + bot_token_len;
    size_t end_pos = text.find(eot_token_, content_start);
    if (end_pos == std::string::npos) {
      break;
    }

    ranges.emplace_back(content_start, end_pos);
    search_pos = end_pos + eot_token_len;
  }

  return ranges;
}

StreamingParseResult Qwen25Detector::detect_and_parse(
    const std::string& text,
    const std::vector<JsonTool>& tools) {
  size_t bot_token_pos = text.find(bot_token_);

  std::string normal_text;
  if (bot_token_pos != std::string::npos) {
    std::string_view normal_text_view(text.data(), bot_token_pos);
    std::string_view trimmed = trim_whitespace(normal_text_view);
    normal_text = std::string(trimmed);
  } else {
    std::string_view trimmed = trim_whitespace(text);
    normal_text = std::string(trimmed);
    return StreamingParseResult(normal_text);
  }

  auto tool_call_ranges = find_tool_call_ranges(text);

  std::vector<ToolCallItem> calls;
  calls.reserve(tool_call_ranges.size());

  for (const auto& range : tool_call_ranges) {
    std::string_view content_view(text.data() + range.first,
                                  range.second - range.first);
    std::string_view trimmed_content = trim_whitespace(content_view);

    if (trimmed_content.empty()) {
      continue;
    }

    try {
      std::string json_content(trimmed_content);
      auto json_obj = nlohmann::json::parse(json_content);
      auto parsed_calls = parse_base_json(json_obj, tools);

      calls.insert(calls.end(),
                   std::make_move_iterator(parsed_calls.begin()),
                   std::make_move_iterator(parsed_calls.end()));
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to parse JSON part: "
                 << std::string(trimmed_content)
                 << ", JSON parse error: " << e.what();
      continue;
    }
  }

  return StreamingParseResult(std::move(normal_text), std::move(calls));
}

StreamingParseResult Qwen25Detector::parse_streaming_increment(
    const std::string& new_text,
    const std::vector<JsonTool>& tools) {
  // Streaming incremental parsing for Qwen 2.5/3 tool calls.
  // Uses base class implementation with buffering to handle partial end tokens.
  StreamingParseResult result =
      BaseFormatDetector::parse_streaming_increment(new_text, tools);

  filter_end_token(result, /*final=*/false);
  return result;
}

StreamingFinishResult Qwen25Detector::finish_stream(
    const std::vector<JsonTool>& tools) {
  StreamingFinishResult result = finish_json_buffer(tools);
  filter_end_token(result.output, /*final=*/true);
  return result;
}

void Qwen25Detector::filter_end_token(StreamingParseResult& result,
                                      bool final) {
  normal_text_buffer_ += result.normal_text;
  const std::string end_token = eot_token_.substr(1);
  size_t position = 0;
  while ((position = normal_text_buffer_.find(end_token)) !=
         std::string::npos) {
    normal_text_buffer_.erase(position, end_token.size());
  }
  const size_t held = final ? 0
                            : static_cast<size_t>(ends_with_partial_token(
                                  normal_text_buffer_, end_token));
  result.normal_text =
      normal_text_buffer_.substr(0, normal_text_buffer_.size() - held);
  normal_text_buffer_.erase(0, normal_text_buffer_.size() - held);
}

}  // namespace function_call
}  // namespace xllm

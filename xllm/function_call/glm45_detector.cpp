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

#include "function_call/glm45_detector.h"

#include <algorithm>
#include <iterator>
#include <utility>

namespace xllm {
namespace function_call {

Glm45Detector::Glm45Detector() : BaseFormatDetector() {
  bot_token_ = "<tool_call>";
  eot_token_ = "</tool_call>";

  // Regex patterns for GLM-4.5 format
  func_call_regex_ = std::regex("<tool_call>[\\s\\S]*?</tool_call>",
                                std::regex_constants::ECMAScript);
  func_detail_regex_ =
      std::regex("<tool_call>([^\\n]*)\\n([\\s\\S]*?)</tool_call>",
                 std::regex_constants::ECMAScript);
  func_arg_regex_ = std::regex(
      "<arg_key>([\\s\\S]*?)</arg_key>\\s*<arg_value>([\\s\\S]*?)</arg_value>",
      std::regex_constants::ECMAScript);
}

std::string Glm45Detector::trim_whitespace(std::string_view str) const {
  const char* whitespace = " \t\n\r";

  size_t start = str.find_first_not_of(whitespace);
  if (start == std::string_view::npos) {
    return std::string{};
  }

  size_t end = str.find_last_not_of(whitespace);

  return std::string(str.substr(start, end - start + 1));
}

bool Glm45Detector::has_tool_call(const std::string& text) {
  return text.find(bot_token_) != std::string::npos;
}

std::unordered_map<std::string, nlohmann::json> Glm45Detector::parse_arguments(
    const std::string& args) const {
  std::unordered_map<std::string, nlohmann::json> arguments;
  const std::sregex_iterator end;
  for (std::sregex_iterator iter(args.begin(), args.end(), func_arg_regex_);
       iter != end;
       ++iter) {
    const std::smatch match = *iter;
    const std::string key = trim_whitespace(match[1].str());
    const std::string value = trim_whitespace(match[2].str());
    try {
      arguments[key] = nlohmann::json::parse(value);
    } catch (const nlohmann::json::parse_error&) {
      arguments[key] = value;
    }
  }
  return arguments;
}

StreamingParseResult Glm45Detector::detect_and_parse(
    const std::string& text,
    const std::vector<JsonTool>& tools) {
  size_t idx = text.find(bot_token_);
  std::string normal_text =
      (idx != std::string::npos) ? text.substr(0, idx) : text;

  // Trim normal text
  if (!normal_text.empty()) {
    normal_text = trim_whitespace(normal_text);
  }

  if (idx == std::string::npos) {
    return StreamingParseResult(normal_text, {});
  }

  std::vector<ToolCallItem> calls;

  try {
    std::sregex_iterator iter(text.begin(), text.end(), func_call_regex_);
    std::sregex_iterator end;

    for (; iter != end; ++iter) {
      std::smatch match = *iter;
      std::string match_result = match.str();

      // Parse function name and arguments
      std::smatch func_detail;
      if (std::regex_search(match_result, func_detail, func_detail_regex_)) {
        std::string func_name = func_detail[1].str();
        std::string func_args = func_detail[2].str();

        auto arguments = parse_arguments(func_args);

        // Create JSON object for parse_base_json
        nlohmann::json match_json;
        match_json["name"] = func_name;
        match_json["parameters"] = arguments;

        auto parsed_calls = parse_base_json(match_json, tools);
        calls.insert(calls.end(),
                     std::make_move_iterator(parsed_calls.begin()),
                     std::make_move_iterator(parsed_calls.end()));
      }
    }

    return StreamingParseResult(normal_text, calls);

  } catch (const std::exception& e) {
    LOG(ERROR) << "Error in GLM-4.5 detect_and_parse: " << e.what();
    return StreamingParseResult(text, {});
  }
}

StreamingParseResult Glm45Detector::parse_streaming_increment(
    const std::string& new_text,
    const std::vector<JsonTool>& tools) {
  CHECK(!stream_finished_) << "Cannot append to a finished tool stream";
  buffer_ += new_text;
  return consume_buffer(tools, /*final=*/false);
}

StreamingParseResult Glm45Detector::consume_buffer(
    const std::vector<JsonTool>& tools,
    bool final) {
  StreamingParseResult result;
  while (!buffer_.empty()) {
    const size_t start = buffer_.find(bot_token_);
    if (start == std::string::npos) {
      size_t held = 0;
      if (!final) {
        const size_t limit = std::min(buffer_.size(), bot_token_.size() - 1);
        for (size_t length = limit; length > 0; --length) {
          if (buffer_.compare(
                  buffer_.size() - length, length, bot_token_, 0, length) ==
              0) {
            held = length;
            break;
          }
        }
      }
      const size_t available = buffer_.size() - held;
      result.normal_text.append(buffer_, 0, available);
      buffer_.erase(0, available);
      break;
    }
    if (start > 0) {
      result.normal_text.append(buffer_, 0, start);
      buffer_.erase(0, start);
    }

    const size_t end = buffer_.find(eot_token_, bot_token_.length());
    StreamingParseResult parsed;
    if (end != std::string::npos) {
      const size_t consumed = end + eot_token_.length();
      parsed = detect_and_parse(buffer_.substr(0, consumed), tools);
      buffer_.erase(0, consumed);
    } else {
      if (!final) {
        break;
      }
      const size_t name_end = buffer_.find('\n', bot_token_.length());
      if (name_end == std::string::npos) {
        break;
      }
      const std::string name = trim_whitespace(
          buffer_.substr(bot_token_.length(), name_end - bot_token_.length()));
      const std::string args = buffer_.substr(name_end + 1);
      const std::sregex_iterator args_end;
      size_t consumed_args = 0;
      bool complete_pairs = false;
      for (std::sregex_iterator iter(args.begin(), args.end(), func_arg_regex_);
           iter != args_end;
           ++iter) {
        const std::smatch match = *iter;
        const size_t position = static_cast<size_t>(match.position());
        if (!trim_whitespace(
                 args.substr(consumed_args, position - consumed_args))
                 .empty()) {
          break;
        }
        complete_pairs = true;
        consumed_args = position + static_cast<size_t>(match.length());
      }
      const std::string tail = trim_whitespace(args.substr(consumed_args));
      const bool outer_tag_prefix =
          !tail.empty() && tail.size() < eot_token_.size() &&
          eot_token_.compare(0, tail.size(), tail) == 0;
      if (name.empty() || !complete_pairs ||
          (!tail.empty() && !outer_tag_prefix)) {
        break;
      }
      nlohmann::json call = {{"name", name},
                             {"parameters", parse_arguments(args)}};
      parsed.calls = parse_base_json(call, tools);
      buffer_.clear();
    }

    for (auto& call : parsed.calls) {
      if (current_tool_id_ == -1) {
        current_tool_id_ = 0;
      }
      const size_t tool_index = static_cast<size_t>(current_tool_id_);
      prev_tool_call_arr_.resize(tool_index + 1);
      streamed_args_for_tool_.resize(tool_index + 1);
      prev_tool_call_arr_[tool_index] = {{"name", call.name.value_or("")},
                                         {"arguments", call.parameters}};
      streamed_args_for_tool_[tool_index] = call.parameters;
      call.tool_index = current_tool_id_++;
      result.calls.emplace_back(std::move(call));
    }
  }
  return result;
}

StreamingFinishResult Glm45Detector::finish_stream(
    const std::vector<JsonTool>& tools) {
  if (stream_finished_) {
    return {StreamingParseResult(), has_pending_tool_};
  }
  StreamingParseResult output = consume_buffer(tools, /*final=*/true);
  has_pending_tool_ = buffer_.find(bot_token_) != std::string::npos;
  buffer_.clear();
  stream_finished_ = true;
  return {std::move(output), has_pending_tool_};
}

}  // namespace function_call
}  // namespace xllm

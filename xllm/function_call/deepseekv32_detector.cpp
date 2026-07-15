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

#include "deepseekv32_detector.h"

#include <algorithm>
#include <cctype>
#include <sstream>

#include "base_format_detector.h"
#include "utils.h"

namespace xllm {
namespace function_call {

DeepSeekV32Detector::DeepSeekV32Detector()
    : DeepSeekV32Detector("<｜DSML｜function_calls>",
                          "</｜DSML｜function_calls>") {}

DeepSeekV32Detector::DeepSeekV32Detector(
    const std::string& tool_calls_start_token,
    const std::string& tool_calls_end_token)
    : BaseFormatDetector() {
  bot_token_ = tool_calls_start_token;
  eot_token_ = tool_calls_end_token;
  invoke_end_token_ = "</｜DSML｜invoke>";

  // Regex patterns for DeepSeek-V3.2 DSML format
  function_calls_regex_ = std::regex(bot_token_ + "([\\s\\S]*?)" + eot_token_,
                                     std::regex_constants::ECMAScript);
  // Note: For streaming, we need to match even without closing tag
  // The third group will be empty if no closing tag is found
  // Use non-greedy match for content, but ensure we match the closing tag if
  // present The pattern matches: <｜DSML｜invoke name="..."
  // >content</｜DSML｜invoke> Updated: Make closing tag required for
  // detect_and_parse, but use lookahead for better matching
  invoke_regex_ = std::regex(
      "<｜DSML｜invoke\\s+name=\"([^\"]+)\"\\s*>([\\s\\S]*?)</"
      "｜DSML｜invoke>",
      std::regex_constants::ECMAScript);
  parameter_regex_ = std::regex(
      "<｜DSML｜parameter\\s+name=\"([^\"]+)\"\\s+string=\"([^\"]+)\"\\s*>(["
      "\\s\\S]*?)</｜DSML｜parameter>",
      std::regex_constants::ECMAScript);
  partial_parameter_regex_ = std::regex(
      "<｜DSML｜parameter\\s+name=\"([^\"]+)\"\\s+string=\"([^\"]+)\"\\s*>(["
      "\\s\\S]*)$",
      std::regex_constants::ECMAScript);

  prefix_parameter_end_call_ = {"</", "｜DSML｜", "parameter"};

  // Streaming: one regex that matches with or without closing tag (like Python
  // r'<｜DSML｜invoke\s+name="([^"]+)"\s*>(.*?)(</｜DSML｜invoke>|$)')
  streaming_invoke_regex_ = std::regex(
      "<｜DSML｜invoke\\s+name=\"([^\"]+)\"\\s*>([\\s\\S]*?)(</"
      "｜DSML｜invoke>|$)",
      std::regex_constants::ECMAScript);
  utf8_buffer_.clear();
}

DeepSeekV4Detector::DeepSeekV4Detector()
    : DeepSeekV32Detector("<｜DSML｜tool_calls>", "</｜DSML｜tool_calls>") {}

std::pair<std::string, std::string> DeepSeekV32Detector::split_incomplete_utf8(
    const std::string& str) const {
  if (str.empty()) {
    return {"", ""};
  }

  size_t len = str.length();
  size_t start_pos = len;
  for (size_t i = 1; i <= len && i <= 4; ++i) {
    if ((static_cast<unsigned char>(str[len - i]) & 0xC0) != 0x80) {
      start_pos = len - i;
      break;
    }
  }

  if (start_pos == len) {
    return {str, ""};
  }

  const unsigned char start_byte = str[start_pos];
  size_t needed = 0;
  if ((start_byte & 0x80) == 0) {
    needed = 1;
  } else if ((start_byte & 0xE0) == 0xC0) {
    needed = 2;
  } else if ((start_byte & 0xF0) == 0xE0) {
    needed = 3;
  } else if ((start_byte & 0xF8) == 0xF0) {
    needed = 4;
  } else {
    return {str, ""};
  }

  const size_t available = len - start_pos;
  if (available == needed) {
    return {str, ""};
  }

  return {str.substr(0, start_pos), str.substr(start_pos)};
}

// Dump parameters to JSON string with sorted keys for stable prefix comparison
// across streaming chunks (unordered_map iteration order is undefined).
static std::string params_to_json_sorted(
    const std::unordered_map<std::string, nlohmann::json>& params) {
  std::vector<std::string> keys;
  keys.reserve(params.size());
  for (const auto& p : params) {
    keys.push_back(p.first);
  }
  std::sort(keys.begin(), keys.end());
  nlohmann::json j = nlohmann::json::object();
  for (const auto& k : keys) {
    j[k] = params.at(k);
  }
  return j.dump(-1, ' ', false, nlohmann::json::error_handler_t::ignore);
}

static std::string strip_trailing_markdown_fence(const std::string& text) {
  static const std::regex kTrailingFenceRegex(R"((?:\r?\n)?```[^\n\r`]*\s*$)",
                                              std::regex_constants::ECMAScript);
  return std::regex_replace(text, kTrailingFenceRegex, "");
}

static std::string strip_leading_markdown_fence(const std::string& text) {
  static const std::regex kLeadingFenceRegex(R"(^\s*```[^\n\r`]*\r?\n?)",
                                             std::regex_constants::ECMAScript);
  return std::regex_replace(text, kLeadingFenceRegex, "");
}

static std::string strip_trailing_json_hint(const std::string& text) {
  static const std::regex kTrailingJsonHintRegex(
      R"((?:\r?\n)\s*json\s*$)",
      std::regex_constants::ECMAScript | std::regex_constants::icase);
  return std::regex_replace(text, kTrailingJsonHintRegex, "");
}

static bool is_ascii_whitespace(char ch) {
  return std::isspace(static_cast<unsigned char>(ch)) != 0;
}

static bool match_placeholder_token(std::string_view text,
                                    size_t pos,
                                    size_t end,
                                    size_t* token_len) {
  static const std::vector<std::string> kPlaceholderTokens = {
      "、", "，", ",", "。", ".", "·", "…"};

  for (const auto& token : kPlaceholderTokens) {
    if (pos + token.size() <= end && text.substr(pos, token.size()) == token) {
      *token_len = token.size();
      return true;
    }
  }
  return false;
}

static bool is_placeholder_only_segment(std::string_view text) {
  size_t start = 0;
  size_t end = text.size();
  while (start < end && is_ascii_whitespace(text[start])) {
    ++start;
  }
  while (end > start && is_ascii_whitespace(text[end - 1])) {
    --end;
  }
  if (start == end) {
    return false;
  }

  size_t pos = start;
  size_t placeholder_count = 0;
  while (pos < end) {
    if (is_ascii_whitespace(text[pos])) {
      ++pos;
      continue;
    }

    size_t token_len = 0;
    if (!match_placeholder_token(text, pos, end, &token_len)) {
      return false;
    }
    pos += token_len;
    ++placeholder_count;
  }
  return placeholder_count >= 2;
}

static std::string strip_trailing_placeholder_punctuation(
    const std::string& text) {
  size_t trimmed_end = text.size();
  while (trimmed_end > 0 && is_ascii_whitespace(text[trimmed_end - 1])) {
    --trimmed_end;
  }
  if (trimmed_end == 0) {
    return "";
  }

  size_t line_start = trimmed_end;
  while (line_start > 0 && text[line_start - 1] != '\n' &&
         text[line_start - 1] != '\r') {
    --line_start;
  }

  if (!is_placeholder_only_segment(std::string_view(text).substr(
          line_start, trimmed_end - line_start))) {
    return text;
  }

  size_t remove_start = line_start;
  if (remove_start > 0 && text[remove_start - 1] == '\n') {
    --remove_start;
    if (remove_start > 0 && text[remove_start - 1] == '\r') {
      --remove_start;
    }
  } else if (remove_start > 0 && text[remove_start - 1] == '\r') {
    --remove_start;
  }

  return text.substr(0, remove_start);
}

static std::string strip_leading_placeholder_punctuation(
    const std::string& text) {
  size_t line_end = 0;
  while (line_end < text.size() && text[line_end] != '\n' &&
         text[line_end] != '\r') {
    ++line_end;
  }

  if (!is_placeholder_only_segment(
          std::string_view(text).substr(0, line_end))) {
    return text;
  }

  size_t remove_end = line_end;
  if (remove_end < text.size() && text[remove_end] == '\r') {
    ++remove_end;
  }
  if (remove_end < text.size() && text[remove_end] == '\n') {
    ++remove_end;
  }
  while (remove_end < text.size() && is_ascii_whitespace(text[remove_end]) &&
         text[remove_end] != '\n' && text[remove_end] != '\r') {
    ++remove_end;
  }

  return text.substr(remove_end);
}

static std::string trim_line_trailing_spaces(const std::string& text) {
  static const std::regex kSpacesBeforeNewlineRegex(
      R"([ \t]+(\r?\n))", std::regex_constants::ECMAScript);
  static const std::regex kSpacesAtEndRegex(R"([ \t]+$)",
                                            std::regex_constants::ECMAScript);

  std::string trimmed =
      std::regex_replace(text, kSpacesBeforeNewlineRegex, "$1");
  return std::regex_replace(trimmed, kSpacesAtEndRegex, "");
}

static std::string sanitize_text_before_tool_block(const std::string& text) {
  std::string sanitized = text;
  while (true) {
    std::string updated = strip_trailing_markdown_fence(sanitized);
    updated = strip_trailing_json_hint(updated);
    updated = strip_trailing_markdown_fence(updated);
    updated = strip_trailing_placeholder_punctuation(updated);

    if (updated == sanitized) {
      break;
    }
    sanitized = updated;
  }
  return trim_line_trailing_spaces(sanitized);
}

static bool ends_with_markdown_fence_prefix(const std::string& text) {
  static const std::regex kTrailingFencePrefixRegex(
      R"((?:\r?\n)?```[^\n\r`]*\s*$)", std::regex_constants::ECMAScript);
  return std::regex_search(text, kTrailingFencePrefixRegex);
}

static bool split_trailing_empty_markdown_fence_block(
    const std::string& text,
    std::string* leading_text,
    std::string* trailing_fence_block) {
  static const std::regex kTrailingEmptyFenceBlockRegex(
      R"(^([\s\S]*?)(```[^\n\r`]*\r?\n(?:\s*\r?\n)*```)[ \t\r\n]*$)",
      std::regex_constants::ECMAScript);

  std::smatch match;
  if (!std::regex_match(text, match, kTrailingEmptyFenceBlockRegex) ||
      match.size() < 3) {
    return false;
  }

  if (leading_text != nullptr) {
    *leading_text = match[1].str();
  }
  if (trailing_fence_block != nullptr) {
    *trailing_fence_block = match[2].str();
  }
  return true;
}

static size_t find_trailing_whitespace_start(const std::string& text) {
  size_t idx = text.size();
  while (idx > 0 &&
         std::isspace(static_cast<unsigned char>(text[idx - 1])) != 0) {
    --idx;
  }
  return idx;
}

static std::string normalize_deferred_whitespace(const std::string& text) {
  if (text.empty()) {
    return text;
  }

  bool has_newline = (text.find('\n') != std::string::npos ||
                      text.find('\r') != std::string::npos);
  if (has_newline) {
    return "\n";
  }

  return " ";
}

std::string DeepSeekV32Detector::trim_whitespace(std::string_view str) const {
  const char* whitespace = " \t\n\r";

  size_t start = str.find_first_not_of(whitespace);
  if (start == std::string_view::npos) {
    return std::string{};
  }

  size_t end = str.find_last_not_of(whitespace);

  return std::string(str.substr(start, end - start + 1));
}

bool DeepSeekV32Detector::has_tool_call(const std::string& text) {
  // Check for DSML format first (primary format)
  if (text.find(bot_token_) != std::string::npos ||
      text.find("<｜DSML｜invoke") != std::string::npos) {
    return true;
  }

  // Fallback: Check for JSON format (in case chat_template is not applied)
  // Look for JSON tool_calls pattern: {"tool_calls": [...]} or "tool_calls":
  // [...]
  if (text.find("\"tool_calls\"") != std::string::npos ||
      text.find("'tool_calls'") != std::string::npos ||
      text.find("tool_calls") != std::string::npos) {
    // Additional check: verify it's actually a JSON structure
    size_t tool_calls_pos = text.find("\"tool_calls\"");
    if (tool_calls_pos == std::string::npos) {
      tool_calls_pos = text.find("'tool_calls'");
    }
    if (tool_calls_pos == std::string::npos) {
      tool_calls_pos = text.find("tool_calls");
    }

    if (tool_calls_pos != std::string::npos) {
      // Check if there's a JSON array after "tool_calls"
      size_t array_start = text.find('[', tool_calls_pos);
      if (array_start != std::string::npos) {
        return true;
      }
    }
  }

  return false;
}

std::unordered_map<std::string, nlohmann::json>
DeepSeekV32Detector::parse_parameters_from_xml(
    const std::string& invoke_content,
    bool allow_partial) const {
  std::unordered_map<std::string, nlohmann::json> parameters;

  // First, try to parse as direct JSON (Format 2)
  std::string invoke_content_stripped = trim_whitespace(invoke_content);

  if (invoke_content_stripped.size() >= 2 &&
      invoke_content_stripped[0] == '{' &&
      invoke_content_stripped[invoke_content_stripped.size() - 1] == '}') {
    try {
      nlohmann::json parsed = nlohmann::json::parse(invoke_content_stripped);
      if (parsed.is_object()) {
        // Convert JSON object to map
        for (auto& [key, value] : parsed.items()) {
          parameters[key] = value;
        }
        return parameters;
      }
    } catch (const nlohmann::json::parse_error&) {
      // If JSON parsing fails, fall through to XML parsing
    }
  }

  // Fall back to XML parameter tag parsing (Format 1)
  std::sregex_iterator param_iter(
      invoke_content.begin(), invoke_content.end(), parameter_regex_);
  std::sregex_iterator param_end;

  size_t last_match_end = 0;
  for (; param_iter != param_end; ++param_iter) {
    std::smatch match = *param_iter;
    if (match.size() >= 4) {
      std::string param_name = trim_whitespace(match[1].str());
      std::string param_type = trim_whitespace(match[2].str());
      // For string type parameters, preserve the original value including
      // whitespace For non-string types, we may need to trim for JSON parsing
      std::string param_value = match[3].str();
      last_match_end = match.position() + match.length();

      // Convert value based on type
      if (param_type == "true") {  // string type - preserve original value
        parameters[param_name] = param_value;
      } else {
        // For non-string types, trim whitespace before JSON parsing
        std::string trimmed_value = trim_whitespace(param_value);
        try {
          parameters[param_name] = nlohmann::json::parse(trimmed_value);
        } catch (const nlohmann::json::parse_error&) {
          parameters[param_name] = trimmed_value;
        }
      }
    }
  }

  // If allowed, try to parse a partial parameter at the end
  if (allow_partial) {
    std::string remaining_content = invoke_content.substr(last_match_end);

    // Remove incomplete parameter_end_call prefix in case they are captured
    for (auto it = prefix_parameter_end_call_.rbegin();
         it != prefix_parameter_end_call_.rend();
         ++it) {
      const std::string& token = *it;
      while (!remaining_content.empty() &&
             remaining_content.size() >= token.size() &&
             remaining_content.substr(remaining_content.size() -
                                      token.size()) == token) {
        remaining_content = remaining_content.substr(
            0, remaining_content.size() - token.size());
      }
    }

    // Match start of a parameter tag + value (potentially incomplete)
    std::smatch partial_match;
    if (std::regex_search(
            remaining_content, partial_match, partial_parameter_regex_)) {
      if (partial_match.size() >= 4) {
        std::string param_name = trim_whitespace(partial_match[1].str());
        std::string param_type = trim_whitespace(partial_match[2].str());
        std::string param_value = trim_whitespace(partial_match[3].str());

        if (param_type == "true") {
          parameters[param_name] = param_value;
        } else {
          if (param_value.empty()) {
            return parameters;
          }
          // Use partial_json_loads for non-string types
          auto [parsed_value, _] = partial_json_loads(param_value, Allow::ALL);
          parameters[param_name] = parsed_value;
        }
      }
    }
  }

  return parameters;
}

// Member function to parse JSON format tool calls
std::vector<ToolCallItem> DeepSeekV32Detector::parse_json_tool_calls(
    const std::string& text,
    const std::vector<JsonTool>& tools) {
  std::vector<ToolCallItem> calls;

  try {
    // Try to extract JSON from text
    // Look for {"tool_calls": [...]} pattern
    size_t json_start = text.find('{');
    if (json_start == std::string::npos) {
      return calls;
    }

    // Find the matching closing brace
    int brace_count = 0;
    size_t json_end = json_start;
    bool in_string = false;
    bool escape_next = false;

    for (size_t i = json_start; i < text.length(); ++i) {
      char c = text[i];
      if (escape_next) {
        escape_next = false;
        continue;
      }
      if (c == '\\') {
        escape_next = true;
        continue;
      }
      if (c == '"') {
        in_string = !in_string;
        continue;
      }
      if (in_string) {
        continue;
      }
      if (c == '{') {
        brace_count++;
      } else if (c == '}') {
        brace_count--;
        if (brace_count == 0) {
          json_end = i + 1;
          break;
        }
      }
    }

    if (brace_count != 0) {
      // Incomplete JSON, try to find tool_calls array directly
      size_t tool_calls_pos = text.find("\"tool_calls\"");
      if (tool_calls_pos == std::string::npos) {
        tool_calls_pos = text.find("'tool_calls'");
      }
      if (tool_calls_pos == std::string::npos) {
        tool_calls_pos = text.find("tool_calls");
      }

      if (tool_calls_pos != std::string::npos) {
        size_t array_start = text.find('[', tool_calls_pos);
        if (array_start != std::string::npos) {
          // Try to parse array
          int bracket_count = 0;
          size_t array_end = array_start;
          for (size_t i = array_start; i < text.length(); ++i) {
            char c = text[i];
            if (c == '[') {
              bracket_count++;
            } else if (c == ']') {
              bracket_count--;
              if (bracket_count == 0) {
                array_end = i + 1;
                break;
              }
            }
          }

          if (bracket_count == 0) {
            std::string array_str =
                text.substr(array_start, array_end - array_start);
            try {
              nlohmann::json tool_calls_array =
                  nlohmann::json::parse(array_str);
              if (tool_calls_array.is_array()) {
                for (size_t i = 0; i < tool_calls_array.size(); ++i) {
                  const auto& tool_call = tool_calls_array[i];
                  if (tool_call.contains("name") &&
                      tool_call.contains("arguments")) {
                    std::string func_name =
                        tool_call["name"].get<std::string>();
                    nlohmann::json func_args = nlohmann::json::object();
                    if (tool_call["arguments"].is_string()) {
                      try {
                        func_args = nlohmann::json::parse(
                            tool_call["arguments"].get<std::string>());
                      } catch (const nlohmann::json::parse_error&) {
                        func_args = nlohmann::json::object();
                      }
                    } else {
                      func_args = tool_call["arguments"];
                    }

                    // Use parse_base_json to validate tool name
                    nlohmann::json match_result;
                    match_result["name"] = func_name;
                    match_result["parameters"] = func_args;

                    auto parsed_calls =
                        this->parse_base_json(match_result, tools);
                    calls.insert(calls.end(),
                                 std::make_move_iterator(parsed_calls.begin()),
                                 std::make_move_iterator(parsed_calls.end()));
                  }
                }
              }
            } catch (const nlohmann::json::parse_error&) {
              // Ignore parse errors
            }
          }
        }
      }

      return calls;
    }

    std::string json_str = text.substr(json_start, json_end - json_start);
    nlohmann::json parsed = nlohmann::json::parse(json_str);

    if (parsed.contains("tool_calls") && parsed["tool_calls"].is_array()) {
      const auto& tool_calls_array = parsed["tool_calls"];
      for (size_t i = 0; i < tool_calls_array.size(); ++i) {
        const auto& tool_call = tool_calls_array[i];

        std::string func_name;
        nlohmann::json func_args = nlohmann::json::object();

        if (tool_call.contains("function")) {
          const auto& func = tool_call["function"];
          if (func.contains("name")) {
            func_name = func["name"].get<std::string>();
          }
          if (func.contains("arguments")) {
            if (func["arguments"].is_string()) {
              func_args =
                  nlohmann::json::parse(func["arguments"].get<std::string>());
            } else {
              func_args = func["arguments"];
            }
          }
        } else if (tool_call.contains("name")) {
          func_name = tool_call["name"].get<std::string>();
          if (tool_call.contains("arguments")) {
            if (tool_call["arguments"].is_string()) {
              func_args = nlohmann::json::parse(
                  tool_call["arguments"].get<std::string>());
            } else {
              func_args = tool_call["arguments"];
            }
          }
        }

        if (!func_name.empty()) {
          // Use parse_base_json to validate tool name
          nlohmann::json match_result;
          match_result["name"] = func_name;
          match_result["parameters"] = func_args;

          auto parsed_calls = this->parse_base_json(match_result, tools);
          calls.insert(calls.end(),
                       std::make_move_iterator(parsed_calls.begin()),
                       std::make_move_iterator(parsed_calls.end()));
        }
      }
    }
  } catch (const nlohmann::json::parse_error& e) {
    LOG(ERROR) << "Error parsing JSON tool calls: " << e.what();
  } catch (const std::exception& e) {
    LOG(ERROR) << "Error in parse_json_tool_calls: " << e.what();
  }

  return calls;
}

StreamingParseResult DeepSeekV32Detector::detect_and_parse(
    const std::string& text,
    const std::vector<JsonTool>& tools) {
  size_t idx = text.find(bot_token_);
  std::string normal_text;

  // Check if we have function_calls tag or just invoke tags
  bool has_function_calls_tag = (idx != std::string::npos);
  bool has_invoke_tag = (text.find("<｜DSML｜invoke") != std::string::npos);

  // Extract normal_text: before function_calls tag or before first invoke tag
  if (has_function_calls_tag) {
    normal_text = sanitize_text_before_tool_block(text.substr(0, idx));
  } else if (has_invoke_tag) {
    size_t invoke_pos = text.find("<｜DSML｜invoke");
    normal_text = sanitize_text_before_tool_block(text.substr(0, invoke_pos));
  } else {
    normal_text = text;
  }

  // Trim normal text
  if (!normal_text.empty()) {
    normal_text = trim_whitespace(normal_text);
  }

  // Check for JSON format (fallback)
  bool has_json_format = (text.find("\"tool_calls\"") != std::string::npos ||
                          text.find("'tool_calls'") != std::string::npos ||
                          text.find("tool_calls") != std::string::npos);

  if (!has_function_calls_tag && !has_invoke_tag && !has_json_format) {
    return StreamingParseResult(normal_text, {});
  }

  std::vector<ToolCallItem> calls;

  try {
    // Try DSML format first
    if (has_function_calls_tag || has_invoke_tag) {
      std::string search_content;

      // Extract content between function_calls tags if present
      if (has_function_calls_tag) {
        std::smatch function_calls_match;
        if (std::regex_search(
                text, function_calls_match, function_calls_regex_)) {
          search_content = function_calls_match[1].str();
        } else {
          // If function_calls tag exists but regex doesn't match,
          // search from the tag position
          size_t start_pos = text.find(bot_token_) + bot_token_.length();
          size_t end_pos = text.find(eot_token_, start_pos);
          if (end_pos != std::string::npos) {
            search_content = text.substr(start_pos, end_pos - start_pos);
          } else {
            search_content = text.substr(start_pos);
          }
        }
      } else {
        // No function_calls tag, search in the whole text after normal_text
        search_content = text.substr(normal_text.length());
      }

      // Find all invoke blocks
      std::sregex_iterator invoke_iter(
          search_content.begin(), search_content.end(), invoke_regex_);
      std::sregex_iterator invoke_end;

      for (; invoke_iter != invoke_end; ++invoke_iter) {
        std::smatch invoke_match = *invoke_iter;
        if (invoke_match.size() >= 3) {
          std::string func_name = trim_whitespace(invoke_match[1].str());
          std::string invoke_content = invoke_match[2].str();

          // For detect_and_parse, we only process complete invokes with closing
          // tag Since the regex now requires the closing tag, if we matched, we
          // have a complete invoke Parse parameters from XML format
          auto func_args_map = parse_parameters_from_xml(invoke_content, false);

          // Convert map to JSON object
          nlohmann::json func_args = nlohmann::json::object();
          for (const auto& [key, value] : func_args_map) {
            func_args[key] = value;
          }

          // Construct match_result for parse_base_json
          nlohmann::json match_result;
          match_result["name"] = func_name;
          match_result["parameters"] = func_args;

          auto parsed_calls = this->parse_base_json(match_result, tools);
          calls.insert(calls.end(),
                       std::make_move_iterator(parsed_calls.begin()),
                       std::make_move_iterator(parsed_calls.end()));
        }
      }
    }

    // Fallback to JSON format if DSML parsing didn't find anything
    if (calls.empty() && has_json_format) {
      calls = this->parse_json_tool_calls(text, tools);
      // If we found JSON tool calls, remove them from normal_text
      if (!calls.empty()) {
        // Try to remove JSON part from normal_text
        size_t json_start = text.find('{');
        if (json_start != std::string::npos &&
            json_start < normal_text.length()) {
          normal_text =
              sanitize_text_before_tool_block(text.substr(0, json_start));
          normal_text = trim_whitespace(normal_text);
        }
      }
    }

    return StreamingParseResult(normal_text, calls);

  } catch (const std::exception& e) {
    LOG(ERROR) << "Error in DeepSeek-V3.2 detect_and_parse: " << e.what();
    return StreamingParseResult(text, {});
  }
}

StreamingParseResult DeepSeekV32Detector::parse_streaming_increment(
    const std::string& new_text,
    const std::vector<JsonTool>& tools) {
  if (!new_text.empty()) {
    std::string merged_text = utf8_buffer_ + new_text;
    auto [complete_new_text, incomplete_tail] =
        split_incomplete_utf8(merged_text);
    utf8_buffer_ = incomplete_tail;
    buffer_ += complete_new_text;
  } else {
    utf8_buffer_.clear();
  }
  std::string current_text = buffer_;

  // Check if buffer contains any DSML markers or ends with potential tag prefix
  std::vector<std::string> dsml_markers = {"｜DSML｜", "<｜", "</｜"};
  bool potentially_dsml = false;
  for (const auto& marker : dsml_markers) {
    if (current_text.find(marker) != std::string::npos) {
      potentially_dsml = true;
      break;
    }
  }

  // Also check if text ends with start of a tag
  std::vector<std::string> dsml_prefixes = {"<", "<｜", "</", "</｜"};
  bool ends_with_prefix = false;
  std::string current_text_rstrip = trim_whitespace(current_text);
  for (const auto& prefix : dsml_prefixes) {
    if (current_text_rstrip.size() >= prefix.size() &&
        current_text_rstrip.substr(current_text_rstrip.size() -
                                   prefix.size()) == prefix) {
      ends_with_prefix = true;
      break;
    }
  }

  bool ends_with_open_brace =
      (!current_text_rstrip.empty() && current_text_rstrip.back() == '{');

  bool ends_with_markdown_fence_hint =
      ends_with_markdown_fence_prefix(current_text_rstrip);

  std::string text_before_empty_fence_block;
  std::string trailing_empty_fence_block;
  bool has_trailing_empty_fence_block =
      split_trailing_empty_markdown_fence_block(current_text,
                                                &text_before_empty_fence_block,
                                                &trailing_empty_fence_block);

  bool has_json_tool_calls =
      (current_text.find("\"tool_calls\"") != std::string::npos ||
       current_text.find("'tool_calls'") != std::string::npos);

  // Detect incomplete JSON block robustly (handles split key chunks like
  // "tool_ca" + "lls"), so we don't flush partial JSON as normal text.
  bool potentially_json_block = false;
  size_t first_brace = current_text.find('{');
  if (first_brace != std::string::npos) {
    int brace_count = 0;
    bool in_string = false;
    bool escape_next = false;
    for (size_t i = first_brace; i < current_text.length(); ++i) {
      char c = current_text[i];
      if (escape_next) {
        escape_next = false;
        continue;
      }
      if (c == '\\') {
        escape_next = true;
        continue;
      }
      if (c == '"') {
        in_string = !in_string;
        continue;
      }
      if (in_string) {
        continue;
      }
      if (c == '{') {
        brace_count++;
      } else if (c == '}') {
        brace_count--;
      }
    }
    potentially_json_block = (brace_count > 0);
  }

  if (new_text.empty()) {
    if (current_text.empty()) {
      return StreamingParseResult("", {});
    }

    if (has_tool_call(current_text) || potentially_dsml ||
        has_json_tool_calls || ends_with_prefix || potentially_json_block ||
        ends_with_open_brace) {
      buffer_.clear();
      return StreamingParseResult("", {});
    }

    std::string output_text = has_trailing_empty_fence_block
                                  ? text_before_empty_fence_block
                                  : current_text;
    output_text = sanitize_text_before_tool_block(output_text);
    output_text = strip_leading_markdown_fence(output_text);
    output_text = strip_leading_placeholder_punctuation(output_text);
    buffer_.clear();
    return StreamingParseResult(output_text, {});
  }

  if (!has_tool_call(current_text) && !potentially_dsml && !ends_with_prefix &&
      !has_json_tool_calls && !ends_with_open_brace &&
      !potentially_json_block) {
    if (has_trailing_empty_fence_block) {
      buffer_ = trailing_empty_fence_block;
      std::string leading_text =
          sanitize_text_before_tool_block(text_before_empty_fence_block);
      leading_text = trim_whitespace(leading_text);
      if (!leading_text.empty()) {
        return StreamingParseResult(leading_text, {});
      }
      return StreamingParseResult("", {});
    }

    if (ends_with_markdown_fence_hint) {
      return StreamingParseResult("", {});
    }

    // Remove any stray closing tags
    std::string output_text = current_text;
    size_t pos = 0;
    while ((pos = output_text.find(eot_token_, pos)) != std::string::npos) {
      output_text.erase(pos, eot_token_.length());
    }
    pos = 0;
    while ((pos = output_text.find(invoke_end_token_, pos)) !=
           std::string::npos) {
      output_text.erase(pos, invoke_end_token_.length());
    }

    output_text = sanitize_text_before_tool_block(output_text);
    output_text = strip_leading_markdown_fence(output_text);
    output_text = strip_leading_placeholder_punctuation(output_text);

    // Keep minimal trailing whitespace in buffer to preserve token boundaries
    // across chunks without accumulating excessive blank lines/spaces.
    size_t trailing_ws_start = find_trailing_whitespace_start(output_text);
    if (trailing_ws_start < output_text.size()) {
      std::string emit_text = output_text.substr(0, trailing_ws_start);
      buffer_ =
          normalize_deferred_whitespace(output_text.substr(trailing_ws_start));
      if (!emit_text.empty()) {
        return StreamingParseResult(emit_text, {});
      }
      return StreamingParseResult("", {});
    }

    buffer_.clear();

    return StreamingParseResult(output_text, {});
  }

  // Initialize tool indices if needed
  if (tool_indices_.empty()) {
    tool_indices_ = get_tool_indices(tools);
  }

  std::vector<ToolCallItem> all_calls;

  try {
    // Handle JSON tool_calls format in streaming mode.
    // Only trigger on quoted key to avoid treating ordinary prose as JSON.
    // If incomplete, keep buffering and wait for more chunks.
    if (has_json_tool_calls && !potentially_dsml) {
      // If the JSON object has started but is not closed yet, do not parse
      // the tool_calls array early. Parsing too early can leak raw JSON into
      // normal_text and then emit a trailing '}' in a later chunk.
      if (potentially_json_block) {
        // Stream out leading prose before the JSON object as soon as possible,
        // then keep only JSON in the buffer to continue incremental parsing.
        size_t json_start = current_text.find('{');
        if (json_start != std::string::npos && json_start > 0) {
          std::string leading_text = sanitize_text_before_tool_block(
              current_text.substr(0, json_start));
          leading_text = trim_whitespace(leading_text);
          buffer_ = current_text.substr(json_start);
          if (!leading_text.empty()) {
            return StreamingParseResult(leading_text, {});
          }
        }
        return StreamingParseResult("", {});
      }

      auto json_calls = this->parse_json_tool_calls(current_text, tools);
      if (!json_calls.empty()) {
        for (size_t i = 0; i < json_calls.size(); ++i) {
          json_calls[i].tool_index = static_cast<int32_t>(i);
        }

        // Remove parsed JSON block from normal text
        std::string normal_text = current_text;
        size_t json_start = current_text.find('{');
        if (json_start != std::string::npos) {
          int brace_count = 0;
          bool in_string = false;
          bool escape_next = false;
          size_t json_end = std::string::npos;

          for (size_t i = json_start; i < current_text.length(); ++i) {
            char c = current_text[i];
            if (escape_next) {
              escape_next = false;
              continue;
            }
            if (c == '\\') {
              escape_next = true;
              continue;
            }
            if (c == '"') {
              in_string = !in_string;
              continue;
            }
            if (in_string) {
              continue;
            }
            if (c == '{') {
              brace_count++;
            } else if (c == '}') {
              brace_count--;
              if (brace_count == 0) {
                json_end = i;
                break;
              }
            }
          }

          if (json_end != std::string::npos) {
            std::string prefix = sanitize_text_before_tool_block(
                current_text.substr(0, json_start));
            std::string suffix =
                strip_leading_markdown_fence(current_text.substr(json_end + 1));
            normal_text = prefix + suffix;
          }
        }

        buffer_.clear();
        return StreamingParseResult(normal_text, json_calls);
      }

      // Incomplete JSON tool_calls block, wait for more chunks.
      return StreamingParseResult("", {});
    }

    // Continue with DSML format parsing
    // Loop to handle multiple consecutive invoke blocks
    while (true) {
      std::smatch invoke_match;
      // Use single regex that matches with or without closing tag (aligned with
      // Python ds32_de.py: group 3 is "</｜DSML｜invoke>" or "" for
      // end-of-string)
      bool has_match = std::regex_search(
          current_text, invoke_match, streaming_invoke_regex_);

      if (!has_match || invoke_match.size() < 4) {
        break;
      }

      std::string func_name = trim_whitespace(invoke_match[1].str());
      std::string invoke_content = invoke_match[2].str();
      // Group 3 is either "</｜DSML｜invoke>" (complete) or "" (incomplete)
      bool is_tool_end = !invoke_match[3].str().empty();

      // Initialize state if this is the first tool call
      if (current_tool_id_ == -1) {
        current_tool_id_ = 0;
        prev_tool_call_arr_.clear();
        streamed_args_for_tool_.clear();
        streamed_args_for_tool_.push_back("");
        current_tool_name_sent_ = false;
      }

      // Ensure arrays are large enough for current tool
      while (prev_tool_call_arr_.size() <=
             static_cast<size_t>(current_tool_id_)) {
        prev_tool_call_arr_.push_back({});
      }
      while (streamed_args_for_tool_.size() <=
             static_cast<size_t>(current_tool_id_)) {
        streamed_args_for_tool_.push_back("");
      }

      // 1. Send tool name if not sent yet
      if (!current_tool_name_sent_) {
        all_calls.push_back(ToolCallItem(current_tool_id_, func_name, ""));
        current_tool_name_sent_ = true;
      }

      // 2. Parse current parameters (partial or complete)
      auto current_params =
          parse_parameters_from_xml(invoke_content, !is_tool_end);
      // Use sorted-key dump so find_common_prefix is stable across chunks
      std::string current_args_json = params_to_json_sorted(current_params);

      // 3. Calculate and send incremental arguments
      size_t sent_len = streamed_args_for_tool_[current_tool_id_].length();
      std::string argument_diff;

      if (is_tool_end) {
        // If complete, send everything remaining
        if (sent_len < current_args_json.length()) {
          argument_diff = current_args_json.substr(sent_len);
        }
      } else {
        // If partial, send stable prefix diff
        if (prev_tool_call_arr_[current_tool_id_].find("arguments") !=
            prev_tool_call_arr_[current_tool_id_].end()) {
          std::string prev_args_json =
              prev_tool_call_arr_[current_tool_id_]["arguments"];

          if (current_args_json != prev_args_json) {
            std::string prefix =
                find_common_prefix(prev_args_json, current_args_json);
            if (prefix.length() > sent_len) {
              argument_diff = prefix.substr(sent_len);
            }
          }
        }
      }

      if (!argument_diff.empty()) {
        auto [complete_argument_diff, incomplete_tail] =
            split_incomplete_utf8(argument_diff);
        (void)incomplete_tail;
        argument_diff = std::move(complete_argument_diff);
      }

      if (!argument_diff.empty()) {
        all_calls.push_back(
            ToolCallItem(current_tool_id_, std::nullopt, argument_diff));
        streamed_args_for_tool_[current_tool_id_] += argument_diff;
      }

      // Update the stored arguments (store as JSON string)
      prev_tool_call_arr_[current_tool_id_]["name"] = func_name;
      prev_tool_call_arr_[current_tool_id_]["arguments"] = current_args_json;

      // Check if tool call is complete (has closing tag)
      if (is_tool_end) {
        // Remove completed invoke and everything before it.
        // This mirrors ds32_de.py behavior:
        // self._buffer = current_text[invoke_match.end():]
        size_t match_length = invoke_match.length();
        size_t match_end_in_current_text =
            invoke_match.position() + match_length;
        buffer_ = current_text.substr(match_end_in_current_text);
        current_text = buffer_;

        current_tool_id_++;
        current_tool_name_sent_ = false;
        continue;
      }
      // Incomplete invoke: return streamed calls so far, wait for more chunks
      break;
    }

    // No more invoke blocks found
    return StreamingParseResult("", all_calls);

  } catch (const std::exception& e) {
    LOG(ERROR) << "Error in parse_streaming_increment: " << e.what();
    return StreamingParseResult(current_text, {});
  }
}

}  // namespace function_call
}  // namespace xllm

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

#include "glm47_detector.h"

#include <algorithm>
#include <iostream>
#include <sstream>

namespace xllm {
namespace function_call {

Glm47Detector::Glm47Detector() : BaseFormatDetector() {
  bot_token_ = "<tool_call>";
  eot_token_ = "</tool_call>";

  last_arguments_ = "";
  streamed_raw_length_ = 0;
  reset_streaming_state();
}

void Glm47Detector::reset_streaming_state() {
  stream_state_ = StreamState::INIT;
  current_key_ = "";
  current_value_ = "";
  xml_tag_buffer_ = "";
  is_first_param_ = true;
  value_started_ = false;
  cached_value_type_ = "";
  utf8_buffer_ = "";
}

std::pair<std::string, std::string> Glm47Detector::split_incomplete_utf8(
    const std::string& str) const {
  if (str.empty()) {
    return {"", ""};
  }

  size_t len = str.length();

  // Find the start of the last potential character by scanning backwards.
  size_t start_pos = len;
  for (size_t i = 1; i <= len && i <= 4; ++i) {
    if ((static_cast<unsigned char>(str[len - i]) & 0xC0) != 0x80) {
      start_pos = len - i;
      break;
    }
  }

  if (start_pos == len) {
    // String ends with more than 3 continuation bytes, or is empty.
    // This is invalid; do not buffer to prevent DoS via unbounded growth.
    return {str, ""};
  }

  // Check the character starting at start_pos
  const unsigned char start_byte = str[start_pos];
  size_t needed;
  if ((start_byte & 0x80) == 0) {
    needed = 1;
  } else if ((start_byte & 0xE0) == 0xC0) {
    needed = 2;
  } else if ((start_byte & 0xF0) == 0xE0) {
    needed = 3;
  } else if ((start_byte & 0xF8) == 0xF0) {
    needed = 4;
  } else {
    // Invalid start byte. The fragment from start_pos is corrupt.
    // Do not buffer to prevent DoS via unbounded growth.
    return {str, ""};
  }

  const size_t available = len - start_pos;
  if (available == needed) {
    // The last character has the correct number of bytes. Assume it's complete.
    return {str, ""};
  } else {
    // The last character is incomplete or has extra bytes. Treat as incomplete.
    return {str.substr(0, start_pos), str.substr(start_pos)};
  }
}

std::string Glm47Detector::trim_whitespace(std::string_view str) const {
  const char* whitespace = " \t\n\r";

  size_t start = str.find_first_not_of(whitespace);
  if (start == std::string_view::npos) {
    return std::string{};
  }

  size_t end = str.find_last_not_of(whitespace);

  return std::string(str.substr(start, end - start + 1));
}

bool Glm47Detector::has_tool_call(const std::string& text) {
  return text.find(bot_token_) != std::string::npos;
}

std::vector<std::pair<size_t, size_t>> Glm47Detector::find_tool_call_ranges(
    const std::string& text) const {
  std::vector<std::pair<size_t, size_t>> ranges;
  // Pre-allocate for typical case: most requests have 1-4 tool calls
  ranges.reserve(4);

  size_t search_pos = 0;
  const size_t bot_len = bot_token_.length();
  const size_t eot_len = eot_token_.length();

  while (search_pos < text.length()) {
    size_t start_pos = text.find(bot_token_, search_pos);
    if (start_pos == std::string::npos) break;

    size_t content_start = start_pos + bot_len;
    size_t end_pos = text.find(eot_token_, content_start);
    if (end_pos == std::string::npos) break;

    ranges.emplace_back(content_start, end_pos);
    search_pos = end_pos + eot_len;
  }
  return ranges;
}

std::string Glm47Detector::extract_normal_text(
    const std::string& text,
    const std::vector<std::pair<size_t, size_t>>& ranges) const {
  if (ranges.empty()) {
    return trim_whitespace(text);
  }

  std::string normal_text;
  size_t last_end = 0;
  const size_t bot_len = bot_token_.length();
  const size_t eot_len = eot_token_.length();

  for (const auto& range : ranges) {
    size_t block_start = range.first - bot_len;
    if (block_start > last_end) {
      normal_text.append(text, last_end, block_start - last_end);
    }
    last_end = range.second + eot_len;
  }

  if (last_end < text.length()) {
    size_t trailing_bot = text.find(bot_token_, last_end);
    if (trailing_bot == std::string::npos) {
      normal_text.append(text, last_end, std::string::npos);
    } else if (trailing_bot > last_end) {
      normal_text.append(text, last_end, trailing_bot - last_end);
    }
  }

  return trim_whitespace(normal_text);
}

std::pair<std::string, std::string> Glm47Detector::parse_tool_call_content(
    const std::string& content) const {
  const std::string arg_key_tag = "<arg_key>";
  size_t arg_pos = content.find(arg_key_tag);

  if (arg_pos == std::string::npos) {
    // No arguments, entire content is function name
    return {trim_whitespace(content), ""};
  }

  std::string func_name = trim_whitespace(content.substr(0, arg_pos));
  std::string args_raw = content.substr(arg_pos);
  return {func_name, args_raw};
}

std::vector<std::pair<std::string, std::string>>
Glm47Detector::extract_argument_pairs(const std::string& args_raw) const {
  std::vector<std::pair<std::string, std::string>> pairs;

  const std::string key_open = "<arg_key>";
  const std::string key_close = "</arg_key>";
  const std::string val_open = "<arg_value>";
  const std::string val_close = "</arg_value>";

  size_t pos = 0;
  while (pos < args_raw.length()) {
    size_t key_start = args_raw.find(key_open, pos);
    if (key_start == std::string::npos) break;
    key_start += key_open.length();

    size_t key_end = args_raw.find(key_close, key_start);
    if (key_end == std::string::npos) break;

    size_t val_start = args_raw.find(val_open, key_end);
    if (val_start == std::string::npos) break;

    // Check for an intervening key tag, which indicates a malformed pair where
    // a key is missing its value.
    size_t next_key_start =
        args_raw.find(key_open, key_end + key_close.length());
    if (next_key_start != std::string::npos && next_key_start < val_start) {
      // Skip to the next key, as this one is missing a value.
      pos = next_key_start;
      continue;
    }

    val_start += val_open.length();

    size_t val_end = args_raw.find(val_close, val_start);
    if (val_end == std::string::npos) break;

    std::string key = args_raw.substr(key_start, key_end - key_start);
    std::string value = args_raw.substr(val_start, val_end - val_start);
    pairs.emplace_back(key, value);

    pos = val_end + val_close.length();
  }
  return pairs;
}

std::string Glm47Detector::get_argument_type(
    const std::string& func_name,
    const std::string& arg_key,
    const std::vector<JsonTool>& tools) const {
  // Build name to tool map
  std::unordered_map<std::string, const JsonTool*> name2tool;
  for (const auto& tool : tools) {
    name2tool[tool.function.name] = &tool;
  }

  auto it = name2tool.find(func_name);
  if (it == name2tool.end()) {
    return "";
  }

  const JsonTool* tool = it->second;
  if (!tool->function.parameters.contains("properties")) {
    return "";
  }

  const auto& properties = tool->function.parameters["properties"];
  if (!properties.is_object() || !properties.contains(arg_key)) {
    return "";
  }

  const auto& prop = properties[arg_key];
  if (prop.contains("type") && prop["type"].is_string()) {
    return prop["type"].get<std::string>();
  }

  return "";
}

nlohmann::json Glm47Detector::convert_to_number(
    const std::string& value) const {
  try {
    std::string trimmed = trim_whitespace(value);
    if (trimmed.find('.') != std::string::npos ||
        trimmed.find('e') != std::string::npos ||
        trimmed.find('E') != std::string::npos) {
      return nlohmann::json(std::stod(trimmed));
    } else {
      return nlohmann::json(std::stoll(trimmed));
    }
  } catch (const std::exception&) {
    return nlohmann::json(value);
  }
}

std::pair<nlohmann::json, bool> Glm47Detector::parse_arguments(
    const std::string& json_value,
    const std::string& arg_type) const {
  // Strategy 1: Direct JSON parsing
  try {
    nlohmann::json parsed_value = nlohmann::json::parse(json_value);

    // Type coercion for number type
    if (arg_type == "number" && parsed_value.is_string()) {
      parsed_value = convert_to_number(parsed_value.get<std::string>());
    }

    return {parsed_value, true};
  } catch (const nlohmann::json::parse_error&) {
    // Continue to next strategy
  }

  // Strategy 2: Unescape and parse
  try {
    std::string wrapped = "{\"tmp\": \"" + json_value + "\"}";
    nlohmann::json temp = nlohmann::json::parse(wrapped);
    nlohmann::json parsed_value =
        nlohmann::json::parse(temp["tmp"].get<std::string>());

    if (arg_type == "number" && parsed_value.is_string()) {
      parsed_value = convert_to_number(parsed_value.get<std::string>());
    }

    return {parsed_value, true};
  } catch (const nlohmann::json::parse_error&) {
    // Continue to next strategy
  } catch (const std::exception&) {
    // Continue to next strategy
  }

  // Strategy 3: Treat as string
  try {
    return {nlohmann::json(json_value), true};
  } catch (const std::exception&) {
    return {nlohmann::json(json_value), false};
  }
}

std::unordered_map<std::string, nlohmann::json>
Glm47Detector::parse_argument_pairs(
    const std::vector<std::pair<std::string, std::string>>& pairs,
    const std::string& func_name,
    const std::vector<JsonTool>& tools) const {
  std::unordered_map<std::string, nlohmann::json> arguments;

  for (const auto& [arg_key, arg_value] : pairs) {
    std::string key = trim_whitespace(arg_key);
    std::string value = trim_whitespace(arg_value);

    std::string arg_type = get_argument_type(func_name, key, tools);
    auto [parsed_value, is_good_json] = parse_arguments(value, arg_type);

    if (arg_type == "string") {
      // Only convert to string if explicitly defined as string type
      if (parsed_value.is_string()) {
        arguments[key] = parsed_value;
      } else if (parsed_value.is_object() || parsed_value.is_array()) {
        // If parsed as dict/list but schema says string, convert to JSON string
        arguments[key] = parsed_value.dump();
      } else {
        arguments[key] = parsed_value.dump();
      }
    } else if (arg_type.empty()) {
      // If type is not defined, keep the parsed value as-is
      arguments[key] = is_good_json ? parsed_value : nlohmann::json(value);
    } else {
      // For other types (number, object, array, etc.), use parsed value
      arguments[key] = is_good_json ? parsed_value : nlohmann::json(value);
    }
  }

  return arguments;
}

StreamingParseResult Glm47Detector::detect_and_parse(
    const std::string& text,
    const std::vector<JsonTool>& tools) {
  std::vector<ToolCallItem> calls;

  try {
    // Use string-based parsing instead of regex to avoid stack overflow
    auto ranges = find_tool_call_ranges(text);

    if (ranges.empty()) {
      size_t idx = text.find(bot_token_);
      std::string normal_text = trim_whitespace(text);
      if (idx != std::string::npos) {
        normal_text = trim_whitespace(text.substr(0, idx));
      }
      return StreamingParseResult(normal_text, {});
    }

    std::string normal_text = extract_normal_text(text, ranges);

    for (const auto& range : ranges) {
      std::string content =
          text.substr(range.first, range.second - range.first);
      auto [func_name, args_raw] = parse_tool_call_content(content);
      auto pairs = extract_argument_pairs(args_raw);

      auto arguments = parse_argument_pairs(pairs, func_name, tools);

      // Create JSON object for parse_base_json
      nlohmann::json match_json;
      match_json["name"] = func_name;
      match_json["parameters"] = arguments;

      auto parsed_calls = parse_base_json(match_json, tools);
      calls.insert(calls.end(),
                   std::make_move_iterator(parsed_calls.begin()),
                   std::make_move_iterator(parsed_calls.end()));
    }

    return StreamingParseResult(normal_text, calls);

  } catch (const std::exception& e) {
    LOG(ERROR) << "Error in GLM-4.7 detect_and_parse: " << e.what();
    return StreamingParseResult(text, {});
  }
}

std::string Glm47Detector::get_value_type(
    const std::string& func_name,
    const std::string& key,
    const std::vector<JsonTool>& tools) const {
  std::string arg_type = get_argument_type(func_name, key, tools);
  if (!arg_type.empty()) {
    return arg_type;
  }

  // Auto-detect type from value (best effort)
  std::string first_chars = trim_whitespace(current_value_);
  if (first_chars.length() >= 10) {
    first_chars = first_chars.substr(0, 10);
  }

  if (!first_chars.empty()) {
    char first_char = first_chars[0];
    if (std::isdigit(first_char) || first_char == '-' || first_char == '.') {
      return "number";
    } else if (first_char == '{' || first_char == '[') {
      return "object";
    }
  }

  return "string";
}

std::string Glm47Detector::format_value_complete(
    const std::string& value,
    const std::string& value_type) const {
  if (value_type == "string") {
    // Ensure proper JSON string formatting with quotes
    return nlohmann::json(value).dump();
  } else if (value_type == "number") {
    try {
      nlohmann::json num = convert_to_number(trim_whitespace(value));
      return num.dump();
    } catch (const std::exception&) {
      // Fallback to string if not a valid number
      LOG(WARNING) << "Failed to parse '" << value
                   << "' as number, treating as string";
      return nlohmann::json(value).dump();
    }
  } else {
    // For object/array types, return as-is (should already be valid JSON)
    return value;
  }
}

std::string Glm47Detector::process_xml_to_json_streaming(
    const std::string& raw_increment,
    const std::string& func_name,
    const std::vector<JsonTool>& tools) {
  std::string json_output;

  for (char ch : raw_increment) {
    xml_tag_buffer_ += ch;

    if (stream_state_ == StreamState::INIT ||
        stream_state_ == StreamState::BETWEEN) {
      if (xml_tag_buffer_.size() >= 9 &&
          xml_tag_buffer_.substr(xml_tag_buffer_.size() - 9) == "<arg_key>") {
        stream_state_ = StreamState::IN_KEY;
        current_key_ = "";
        xml_tag_buffer_ = "";
        json_output += is_first_param_ ? "{" : ", ";
        is_first_param_ = false;
      }
    } else if (stream_state_ == StreamState::IN_KEY) {
      if (xml_tag_buffer_.size() >= 10 &&
          xml_tag_buffer_.substr(xml_tag_buffer_.size() - 10) == "</arg_key>") {
        current_key_ = xml_tag_buffer_.substr(0, xml_tag_buffer_.size() - 10);
        current_key_ = trim_whitespace(current_key_);
        xml_tag_buffer_ = "";
        stream_state_ = StreamState::WAITING_VALUE;
        json_output += nlohmann::json(current_key_).dump() + ": ";
      }
    } else if (stream_state_ == StreamState::WAITING_VALUE) {
      if (xml_tag_buffer_.size() >= 11 &&
          xml_tag_buffer_.substr(xml_tag_buffer_.size() - 11) ==
              "<arg_value>") {
        stream_state_ = StreamState::IN_VALUE;
        current_value_ = "";
        xml_tag_buffer_ = "";
        value_started_ = false;
        // Determine and cache the value type at the start
        cached_value_type_ = get_value_type(func_name, current_key_, tools);
      }
    } else if (stream_state_ == StreamState::IN_VALUE) {
      if (xml_tag_buffer_.size() >= 12 &&
          xml_tag_buffer_.substr(xml_tag_buffer_.size() - 12) ==
              "</arg_value>") {
        std::string final_value =
            xml_tag_buffer_.substr(0, xml_tag_buffer_.size() - 12);
        current_value_ += final_value;

        // Use cached value type for consistency
        std::string value_type =
            cached_value_type_.empty() ? "string" : cached_value_type_;

        if (value_started_) {
          // Output any remaining content (including buffered UTF-8 bytes)
          std::string full_final = utf8_buffer_ + final_value;
          utf8_buffer_ = "";
          if (!full_final.empty()) {
            if (value_type == "string") {
              try {
                std::string escaped = nlohmann::json(full_final).dump();
                json_output += escaped.substr(1, escaped.size() - 2);
              } catch (const std::exception& e) {
                // If JSON parsing fails, log and output with JSON escaping
                LOG(WARNING) << "Failed to escape final content: " << e.what();
                for (unsigned char c : full_final) {
                  if (c == '"')
                    json_output += "\\\"";
                  else if (c == '\\')
                    json_output += "\\\\";
                  else if (c < 0x20) {
                    // Escape control characters as \uXXXX
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    json_output += buf;
                  } else
                    json_output += static_cast<char>(c);
                }
              }
            } else {
              json_output += full_final;
            }
          }
          // Always output closing quote for string type when value was started
          if (value_type == "string") {
            json_output += "\"";
          }
        } else {
          // Value was never started (empty or complete in one chunk)
          json_output += format_value_complete(current_value_, value_type);
        }

        xml_tag_buffer_ = "";
        stream_state_ = StreamState::BETWEEN;
        current_value_ = "";
        value_started_ = false;
        cached_value_type_ = "";
      } else {
        // Check if buffer could be start of closing tag
        std::string closing_tag = "</arg_value>";
        bool is_potential_closing =
            xml_tag_buffer_.size() <= closing_tag.size() &&
            closing_tag.substr(0, xml_tag_buffer_.size()) == xml_tag_buffer_;

        if (!is_potential_closing) {
          std::string content = xml_tag_buffer_;
          // Use cached value type for consistency
          std::string value_type =
              cached_value_type_.empty() ? "string" : cached_value_type_;

          if (value_type == "string") {
            if (!value_started_) {
              json_output += "\"";
              value_started_ = true;
            }
            if (!content.empty()) {
              // Prepend any buffered UTF-8 bytes from previous chunk
              std::string full_content = utf8_buffer_ + content;

              // Split into complete UTF-8 and incomplete tail
              auto [complete_utf8, incomplete_tail] =
                  split_incomplete_utf8(full_content);

              if (!complete_utf8.empty()) {
                try {
                  std::string escaped = nlohmann::json(complete_utf8).dump();
                  json_output += escaped.substr(1, escaped.size() - 2);
                  current_value_ += complete_utf8;
                } catch (const std::exception& e) {
                  // If JSON parsing still fails, log and output with JSON
                  // escaping
                  LOG(WARNING) << "Failed to escape content: " << e.what();
                  for (unsigned char c : complete_utf8) {
                    if (c == '"')
                      json_output += "\\\"";
                    else if (c == '\\')
                      json_output += "\\\\";
                    else if (c < 0x20) {
                      // Escape control characters as \uXXXX
                      char buf[8];
                      snprintf(buf, sizeof(buf), "\\u%04x", c);
                      json_output += buf;
                    } else
                      json_output += static_cast<char>(c);
                  }
                  current_value_ += complete_utf8;
                }
              }

              // Buffer the incomplete UTF-8 tail for next chunk
              utf8_buffer_ = incomplete_tail;
              xml_tag_buffer_ = "";
            }
          } else if (value_type == "number") {
            if (!content.empty()) {
              if (!value_started_) {
                value_started_ = true;
              }
              json_output += content;
              current_value_ += content;
              xml_tag_buffer_ = "";
            }
          } else {
            // For object/array types, output as-is
            if (!content.empty()) {
              if (!value_started_) {
                value_started_ = true;
              }
              json_output += content;
              current_value_ += content;
              xml_tag_buffer_ = "";
            }
          }
        }
      }
    }
  }

  return json_output;
}

StreamingParseResult Glm47Detector::parse_streaming_increment(
    const std::string& new_text,
    const std::vector<JsonTool>& tools) {
  buffer_ += new_text;
  std::string current_text = buffer_;

  // Check if we have a tool call
  bool has_tool_call_marker =
      current_text.find(bot_token_) != std::string::npos;

  if (!has_tool_call_marker) {
    // Check if buffer could be the start of a tool call
    bool is_potential_start = false;
    for (size_t i = 1; i <= std::min(current_text.size(), bot_token_.size());
         ++i) {
      if (current_text.size() >= i &&
          bot_token_.substr(0, i) ==
              current_text.substr(current_text.size() - i)) {
        is_potential_start = true;
        break;
      }
    }

    if (!is_potential_start) {
      // Not a potential tool call, return as normal text
      std::string output_text = current_text;
      buffer_.clear();
      // Remove any stray closing tags
      size_t pos = 0;
      while ((pos = output_text.find(eot_token_, pos)) != std::string::npos) {
        output_text.erase(pos, eot_token_.length());
      }
      return StreamingParseResult(output_text, {});
    } else {
      // Could be start of tool call, keep buffering
      return StreamingParseResult("", {});
    }
  }

  // Initialize tool indices if needed
  if (tool_indices_.empty()) {
    tool_indices_ = get_tool_indices(tools);
  }

  std::vector<ToolCallItem> calls;

  try {
    // Use string-based parsing instead of regex to avoid stack overflow
    size_t bot_pos = current_text.find(bot_token_);
    if (bot_pos == std::string::npos) {
      return StreamingParseResult("", {});
    }

    size_t content_start = bot_pos + bot_token_.length();
    size_t eot_pos = current_text.find(eot_token_, content_start);
    bool is_tool_end_flag = (eot_pos != std::string::npos);

    // Extract content (partial or complete)
    std::string content =
        is_tool_end_flag
            ? current_text.substr(content_start, eot_pos - content_start)
            : current_text.substr(content_start);

    // Parse function name and args
    auto [func_name, func_args_raw] = parse_tool_call_content(content);

    // Initialize state if this is the first tool call
    if (current_tool_id_ == -1) {
      current_tool_id_ = 0;
      prev_tool_call_arr_.clear();
      streamed_args_for_tool_.clear();
      streamed_args_for_tool_.push_back("");
      streamed_raw_length_ = 0;
      current_tool_name_sent_ = false;
      reset_streaming_state();
    }

    // Ensure we have enough entries in our tracking arrays
    while (prev_tool_call_arr_.size() <=
           static_cast<size_t>(current_tool_id_)) {
      prev_tool_call_arr_.push_back({});
    }
    while (streamed_args_for_tool_.size() <=
           static_cast<size_t>(current_tool_id_)) {
      streamed_args_for_tool_.push_back("");
    }

    // Send tool name first if not sent yet
    if (!current_tool_name_sent_) {
      // Only send function name when we're sure it's complete:
      // - Either we have <arg_key> (arguments started)
      // - Or we have </tool_call> (tool call ended with no args)
      if (func_name.empty() || (func_args_raw.empty() && !is_tool_end_flag)) {
        // Function name not yet complete, wait for more data
        return StreamingParseResult("", {});
      }
      calls.push_back(ToolCallItem(current_tool_id_, func_name, ""));
      current_tool_name_sent_ = true;
      streamed_raw_length_ = 0;
      reset_streaming_state();
      // Store the tool call info
      prev_tool_call_arr_[current_tool_id_]["name"] = func_name;
      prev_tool_call_arr_[current_tool_id_]["arguments"] = "";
    } else {
      // Process XML to JSON streaming
      size_t current_raw_length = func_args_raw.size();

      if (current_raw_length > streamed_raw_length_) {
        // Get the new raw XML content
        std::string raw_increment = func_args_raw.substr(streamed_raw_length_);

        // Convert XML increment to JSON increment using state machine
        std::string json_increment =
            process_xml_to_json_streaming(raw_increment, func_name, tools);

        if (!json_increment.empty()) {
          calls.push_back(
              ToolCallItem(current_tool_id_, std::nullopt, json_increment));
          last_arguments_ += json_increment;
          streamed_args_for_tool_[current_tool_id_] += json_increment;
        }

        // Update the streamed length
        streamed_raw_length_ = current_raw_length;
      }

      if (is_tool_end_flag) {
        if (is_first_param_) {
          std::string empty_object = "{}";
          calls.push_back(
              ToolCallItem(current_tool_id_, std::nullopt, empty_object));
          last_arguments_ += empty_object;
        } else if (last_arguments_.empty() || last_arguments_.back() != '}') {
          std::string closing_brace = "}";
          calls.push_back(
              ToolCallItem(current_tool_id_, std::nullopt, closing_brace));
          last_arguments_ += closing_brace;
          streamed_args_for_tool_[current_tool_id_] += closing_brace;
        }

        // Use string-based argument extraction
        auto pairs = extract_argument_pairs(func_args_raw);
        if (!pairs.empty()) {
          auto arguments = parse_argument_pairs(pairs, func_name, tools);
          nlohmann::json args_json = arguments;
          prev_tool_call_arr_[current_tool_id_]["arguments"] = args_json.dump();
        }

        // Remove the completed tool call from buffer
        buffer_ = current_text.substr(eot_pos + eot_token_.length());

        StreamingParseResult result("", calls);
        current_tool_id_++;
        last_arguments_ = "";
        current_tool_name_sent_ = false;
        streamed_raw_length_ = 0;
        reset_streaming_state();
        return result;
      }
    }

    return StreamingParseResult("", calls);

  } catch (const std::exception& e) {
    LOG(ERROR) << "Error in parse_streaming_increment: " << e.what();
    return StreamingParseResult(current_text, {});
  }
}

}  // namespace function_call
}  // namespace xllm

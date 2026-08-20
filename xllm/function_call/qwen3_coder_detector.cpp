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

#include "qwen3_coder_detector.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <optional>
#include <string_view>
#include <vector>

namespace xllm {
namespace function_call {

namespace {

bool try_parse_int64(const std::string& text, int64_t* out) {
  if (out == nullptr) {
    return false;
  }
  try {
    size_t idx = 0;
    long long value = std::stoll(text, &idx);
    if (idx != text.length()) {
      return false;
    }
    *out = static_cast<int64_t>(value);
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

bool try_parse_double(const std::string& text, double* out) {
  if (out == nullptr) {
    return false;
  }
  try {
    size_t idx = 0;
    double value = std::stod(text, &idx);
    if (idx != text.length()) {
      return false;
    }
    *out = value;
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

}  // namespace

Qwen3CoderDetector::Qwen3CoderDetector()
    : BaseFormatDetector(),
      parsed_pos_(0),
      current_tool_param_count_(0),
      json_started_(false),
      is_inside_tool_call_(false) {
  tool_call_start_token_ = "<tool_call>";
  tool_call_end_token_ = "</tool_call>";
  tool_call_prefix_ = "<function=";
  function_end_token_ = "</function>";
  parameter_prefix_ = "<parameter=";
  parameter_end_token_ = "</parameter>";
}

bool Qwen3CoderDetector::starts_with(std::string_view text,
                                     std::string_view prefix) {
  return text.length() >= prefix.length() &&
         text.substr(0, prefix.length()) == prefix;
}

std::string Qwen3CoderDetector::to_lower_copy(const std::string& input) {
  std::string lowered = input;
  std::transform(
      lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
        return std::tolower(c);
      });
  return lowered;
}

std::string Qwen3CoderDetector::trim_ascii_whitespace(std::string_view input) {
  const char* whitespace = " \t\n\r\f\v";
  size_t start = input.find_first_not_of(whitespace);
  if (start == std::string_view::npos) {
    return "";
  }
  size_t end = input.find_last_not_of(whitespace);
  return std::string(input.substr(start, end - start + 1));
}

bool Qwen3CoderDetector::has_tool_call(const std::string& text) {
  return text.find(tool_call_start_token_) != std::string::npos;
}

nlohmann::json Qwen3CoderDetector::get_arguments_config(
    const std::string& func_name,
    const std::vector<JsonTool>& tools) const {
  for (const auto& tool : tools) {
    if (tool.type == "function" && tool.function.name == func_name) {
      const auto& params = tool.function.parameters;
      if (params.is_object() && params.contains("properties") &&
          params["properties"].is_object()) {
        return params["properties"];
      }
      if (params.is_object()) {
        return params;
      }
      return nlohmann::json::object();
    }
  }

  LOG(WARNING) << "Tool '" << func_name
               << "' is not defined in the tools list.";
  return nlohmann::json::object();
}

nlohmann::json Qwen3CoderDetector::convert_param_value(
    const std::string& param_value,
    const std::string& param_name,
    const nlohmann::json& param_config,
    const std::string& func_name) const {
  std::string trimmed = trim_ascii_whitespace(param_value);
  std::string lower_value = to_lower_copy(trimmed);

  if (lower_value == "null") {
    return nullptr;
  }

  if (!param_config.empty() && !param_config.contains(param_name)) {
    LOG(WARNING) << "Parsed parameter '" << param_name
                 << "' is not defined in tool '" << func_name
                 << "', directly returning string value.";
    return param_value;
  }

  std::string param_type = "string";
  if (param_config.contains(param_name) &&
      param_config[param_name].is_object() &&
      param_config[param_name].contains("type") &&
      param_config[param_name]["type"].is_string()) {
    param_type =
        to_lower_copy(param_config[param_name]["type"].get<std::string>());
  }

  if (param_type == "string" || param_type == "str" || param_type == "text" ||
      param_type == "varchar" || param_type == "char" || param_type == "enum") {
    return param_value;
  }

  if (starts_with(param_type, "int") || starts_with(param_type, "uint") ||
      starts_with(param_type, "long") || starts_with(param_type, "short") ||
      starts_with(param_type, "unsigned")) {
    int64_t int_value = 0;
    if (try_parse_int64(trimmed, &int_value)) {
      return int_value;
    }
    LOG(WARNING) << "Parsed value '" << param_value << "' of parameter '"
                 << param_name << "' is not an integer in tool '" << func_name
                 << "', degrading to string.";
    return param_value;
  }

  if (starts_with(param_type, "num") || starts_with(param_type, "float")) {
    double float_value = 0.0;
    if (try_parse_double(trimmed, &float_value)) {
      bool maybe_convert = trimmed.find('.') == std::string::npos &&
                           trimmed.find('e') == std::string::npos &&
                           trimmed.find('E') == std::string::npos;
      if (maybe_convert && std::isfinite(float_value)) {
        double rounded = std::round(float_value);
        if (std::abs(float_value - rounded) <=
                std::numeric_limits<double>::epsilon() &&
            rounded >=
                static_cast<double>(std::numeric_limits<int64_t>::min()) &&
            rounded <=
                static_cast<double>(std::numeric_limits<int64_t>::max())) {
          return static_cast<int64_t>(rounded);
        }
      }
      return float_value;
    }
    LOG(WARNING) << "Parsed value '" << param_value << "' of parameter '"
                 << param_name << "' is not a float in tool '" << func_name
                 << "', degrading to string.";
    return param_value;
  }

  if (param_type == "boolean" || param_type == "bool" ||
      param_type == "binary") {
    if (lower_value != "true" && lower_value != "false") {
      LOG(WARNING) << "Parsed value '" << param_value << "' of parameter '"
                   << param_name
                   << "' is not a boolean (`true` or `false`) in tool '"
                   << func_name << "', degrading to false.";
    }
    return lower_value == "true";
  }

  if (param_type == "object" || param_type == "array" || param_type == "arr" ||
      starts_with(param_type, "dict") || starts_with(param_type, "list")) {
    try {
      return nlohmann::json::parse(param_value);
    } catch (const std::exception&) {
      LOG(WARNING) << "Parsed value '" << param_value << "' of parameter '"
                   << param_name << "' cannot be parsed by json.loads in tool '"
                   << func_name << "', degrading to string.";
      return param_value;
    }
  }

  // Best-effort fallback similar to ast.literal_eval behavior.
  if (lower_value == "true") {
    return true;
  }
  if (lower_value == "false") {
    return false;
  }

  int64_t int_value = 0;
  if (try_parse_int64(trimmed, &int_value)) {
    return int_value;
  }
  double float_value = 0.0;
  if (try_parse_double(trimmed, &float_value)) {
    return float_value;
  }

  try {
    return nlohmann::json::parse(param_value);
  } catch (const std::exception&) {
    // Ignore and fallback to string.
  }

  if (param_value.length() >= 2 && param_value.front() == '\'' &&
      param_value.back() == '\'') {
    return param_value.substr(1, param_value.length() - 2);
  }

  LOG(WARNING) << "Parsed value '" << param_value << "' of parameter '"
               << param_name
               << "' cannot be converted with fallback rules in tool '"
               << func_name << "', degrading to string.";
  return param_value;
}

void Qwen3CoderDetector::parse_parameters(const std::string& params_text,
                                          const std::string& func_name,
                                          const std::vector<JsonTool>& tools,
                                          nlohmann::json* parsed_params) const {
  if (parsed_params == nullptr) {
    return;
  }

  const nlohmann::json param_config = get_arguments_config(func_name, tools);
  size_t pos = 0;

  while (pos < params_text.length()) {
    size_t param_start = params_text.find(parameter_prefix_, pos);
    if (param_start == std::string::npos) {
      break;
    }

    size_t name_start = param_start + parameter_prefix_.length();
    size_t name_end = params_text.find('>', name_start);
    if (name_end == std::string::npos) {
      break;
    }

    size_t value_start = name_end + 1;

    size_t cand_end_param = params_text.find(parameter_end_token_, value_start);
    size_t cand_next_param = params_text.find(parameter_prefix_, value_start);
    size_t cand_end_func = params_text.find(function_end_token_, value_start);

    size_t end_pos = std::string::npos;
    size_t end_token_len = 0;

    if (cand_end_param != std::string::npos) {
      end_pos = cand_end_param;
      end_token_len = parameter_end_token_.length();
    }
    if (cand_next_param != std::string::npos && cand_next_param < end_pos) {
      end_pos = cand_next_param;
      end_token_len = 0;
    }
    if (cand_end_func != std::string::npos && cand_end_func < end_pos) {
      end_pos = cand_end_func;
      end_token_len = 0;
    }

    if (end_pos == std::string::npos) {
      break;
    }

    std::string param_name =
        params_text.substr(name_start, name_end - name_start);
    std::string raw_value =
        params_text.substr(value_start, end_pos - value_start);

    if (!raw_value.empty() && raw_value.front() == '\n') {
      raw_value.erase(raw_value.begin());
    }
    if (!raw_value.empty() && raw_value.back() == '\n') {
      raw_value.pop_back();
    }

    (*parsed_params)[param_name] =
        convert_param_value(raw_value, param_name, param_config, func_name);

    pos = end_pos + end_token_len;
  }
}

void Qwen3CoderDetector::parse_tool_call_content(
    const std::string& tool_content,
    const std::vector<JsonTool>& tools,
    int32_t* tool_idx,
    std::vector<ToolCallItem>* calls) const {
  if (tool_idx == nullptr || calls == nullptr) {
    return;
  }

  size_t pos = 0;
  while (pos < tool_content.length()) {
    size_t function_start = tool_content.find(tool_call_prefix_, pos);
    if (function_start == std::string::npos) {
      break;
    }

    size_t name_start = function_start + tool_call_prefix_.length();
    size_t name_end = tool_content.find('>', name_start);
    if (name_end == std::string::npos) {
      break;
    }

    std::string func_name =
        tool_content.substr(name_start, name_end - name_start);

    size_t params_start = name_end + 1;
    size_t function_end = tool_content.find(function_end_token_, params_start);

    std::string params_text;
    if (function_end == std::string::npos) {
      params_text = tool_content.substr(params_start);
      pos = tool_content.length();
    } else {
      params_text =
          tool_content.substr(params_start, function_end - params_start);
      pos = function_end + function_end_token_.length();
    }

    nlohmann::json parsed_params = nlohmann::json::object();
    parse_parameters(params_text, func_name, tools, &parsed_params);

    calls->emplace_back(*tool_idx, func_name, parsed_params.dump());
    (*tool_idx)++;

    if (function_end == std::string::npos) {
      break;
    }
  }
}

StreamingParseResult Qwen3CoderDetector::detect_and_parse(
    const std::string& text,
    const std::vector<JsonTool>& tools) {
  const bool has_tool_call_token =
      !tool_call_start_token_.empty() &&
      text.find(tool_call_start_token_) != std::string::npos;
  const bool has_function_token =
      text.find(tool_call_prefix_) != std::string::npos;

  if (!has_tool_call_token && !has_function_token) {
    return StreamingParseResult(text);
  }

  std::vector<std::string> raw_tool_calls;
  size_t search_pos = 0;

  while (search_pos < text.length()) {
    size_t block_start = text.find(tool_call_start_token_, search_pos);
    if (block_start == std::string::npos) {
      break;
    }
    size_t content_start = block_start + tool_call_start_token_.length();
    size_t block_end = text.find(tool_call_end_token_, content_start);
    if (block_end == std::string::npos) {
      break;
    }

    raw_tool_calls.emplace_back(
        text.substr(content_start, block_end - content_start));
    search_pos = block_end + tool_call_end_token_.length();
  }

  if (raw_tool_calls.empty() && has_function_token) {
    raw_tool_calls.emplace_back(text);
  }

  std::vector<ToolCallItem> calls;
  int32_t tool_idx = 0;
  for (const auto& tool_content : raw_tool_calls) {
    parse_tool_call_content(tool_content, tools, &tool_idx, &calls);
  }

  size_t start_idx = text.find(tool_call_start_token_);
  if (start_idx == std::string::npos) {
    start_idx = text.find(tool_call_prefix_);
  }
  std::string normal_text = (start_idx != std::string::npos && start_idx > 0)
                                ? text.substr(0, start_idx)
                                : "";

  return StreamingParseResult(std::move(normal_text), std::move(calls));
}

StreamingParseResult Qwen3CoderDetector::parse_streaming_increment(
    const std::string& new_text,
    const std::vector<JsonTool>& tools) {
  buffer_ += new_text;
  if (buffer_.empty()) {
    return StreamingParseResult();
  }

  std::vector<ToolCallItem> calls;
  std::string normal_text;

  while (true) {
    if (parsed_pos_ >= buffer_.length()) {
      break;
    }

    std::string_view current_slice(buffer_.data() + parsed_pos_,
                                   buffer_.length() - parsed_pos_);
    if (current_slice.empty()) {
      break;
    }

    if (starts_with(current_slice, tool_call_start_token_)) {
      parsed_pos_ += tool_call_start_token_.length();
      is_inside_tool_call_ = true;
      continue;
    }

    if (starts_with(current_slice, tool_call_prefix_)) {
      size_t end_angle = current_slice.find('>');
      if (end_angle == std::string_view::npos) {
        break;
      }

      std::string func_name(current_slice.substr(
          tool_call_prefix_.length(), end_angle - tool_call_prefix_.length()));
      current_tool_id_ += 1;
      current_tool_name_sent_ = true;
      current_tool_param_count_ = 0;
      json_started_ = false;
      current_func_name_ = func_name;

      calls.emplace_back(current_tool_id_, func_name, "");
      parsed_pos_ += end_angle + 1;
      continue;
    }

    if (starts_with(current_slice, parameter_prefix_)) {
      size_t name_end = current_slice.find('>');
      if (name_end == std::string_view::npos) {
        break;
      }

      size_t value_start = name_end + 1;
      std::string_view rest = current_slice.substr(value_start);

      size_t cand_end_param = rest.find(parameter_end_token_);
      size_t cand_next_param = rest.find(parameter_prefix_);
      size_t cand_end_func = rest.find(function_end_token_);

      size_t end_pos = std::string::npos;
      size_t end_token_len = 0;

      if (cand_end_param != std::string::npos) {
        end_pos = cand_end_param;
        end_token_len = parameter_end_token_.length();
      }
      if (cand_next_param != std::string::npos && cand_next_param < end_pos) {
        end_pos = cand_next_param;
        end_token_len = 0;
      }
      if (cand_end_func != std::string::npos && cand_end_func < end_pos) {
        end_pos = cand_end_func;
        end_token_len = 0;
      }

      if (end_pos == std::string::npos) {
        break;
      }

      std::string param_name(current_slice.substr(
          parameter_prefix_.length(), name_end - parameter_prefix_.length()));
      std::string raw_value(rest.substr(0, end_pos));

      if (!raw_value.empty() && raw_value.front() == '\n') {
        raw_value.erase(raw_value.begin());
      }
      if (!raw_value.empty() && raw_value.back() == '\n') {
        raw_value.pop_back();
      }

      if (!json_started_) {
        calls.emplace_back(current_tool_id_, std::nullopt, "{");
        json_started_ = true;
      }

      const std::string func_name = current_func_name_.value_or("");
      nlohmann::json param_config = get_arguments_config(func_name, tools);
      nlohmann::json converted =
          convert_param_value(raw_value, param_name, param_config, func_name);

      std::string json_key_val =
          nlohmann::json(param_name).dump() + ": " + converted.dump();
      std::string fragment =
          (current_tool_param_count_ > 0 ? ", " : "") + json_key_val;

      calls.emplace_back(current_tool_id_, std::nullopt, fragment);
      current_tool_param_count_ += 1;

      parsed_pos_ += name_end + 1 + end_pos + end_token_len;
      continue;
    }

    if (starts_with(current_slice, function_end_token_)) {
      if (!json_started_) {
        calls.emplace_back(current_tool_id_, std::nullopt, "{");
        json_started_ = true;
      }
      calls.emplace_back(current_tool_id_, std::nullopt, "}");
      parsed_pos_ += function_end_token_.length();
      current_func_name_.reset();
      continue;
    }

    if (starts_with(current_slice, tool_call_end_token_)) {
      parsed_pos_ += tool_call_end_token_.length();
      is_inside_tool_call_ = false;
      continue;
    }

    size_t next_open_angle = current_slice.find('<');
    if (next_open_angle == std::string::npos) {
      if (!is_inside_tool_call_) {
        normal_text.append(current_slice);
      }
      parsed_pos_ += current_slice.length();
      continue;
    }

    if (next_open_angle == 0) {
      std::vector<std::string_view> possible_tags = {tool_call_start_token_,
                                                     tool_call_end_token_,
                                                     tool_call_prefix_,
                                                     function_end_token_,
                                                     parameter_prefix_,
                                                     parameter_end_token_};

      bool is_potential_tag = false;
      for (const auto& tag : possible_tags) {
        if (starts_with(tag, current_slice)) {
          is_potential_tag = true;
          break;
        }
      }

      if (is_potential_tag) {
        break;
      }

      if (!is_inside_tool_call_) {
        normal_text.push_back('<');
      }
      parsed_pos_ += 1;
      continue;
    }

    if (!is_inside_tool_call_) {
      normal_text.append(current_slice.substr(0, next_open_angle));
    }
    parsed_pos_ += next_open_angle;
  }

  if (parsed_pos_ > 0) {
    buffer_ = buffer_.substr(parsed_pos_);
    parsed_pos_ = 0;
  }

  return StreamingParseResult(std::move(normal_text), std::move(calls));
}

}  // namespace function_call
}  // namespace xllm

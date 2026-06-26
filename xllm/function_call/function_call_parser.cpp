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

#include "function_call_parser.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

#include "absl/strings/str_join.h"
#include "core/util/uuid.h"
#include "deepseekv32_detector.h"
#include "deepseekv3_detector.h"
#include "glm45_detector.h"
#include "glm47_detector.h"
#include "kimik2_detector.h"
#include "qwen25_detector.h"
#include "qwen3_coder_detector.h"

namespace xllm {
namespace function_call {

namespace {

const std::unordered_map<std::string, std::vector<std::string>> auto_paser_map =
    {
        {"qwen25", {"qwen2", "qwen3"}},
        {"qwen3_coder", {"qwen3_coder", "qwen35"}},
        {"kimi_k2", {"kimi_k2", "kimi_k25"}},
        {"deepseekv3", {"deepseek_v3"}},
        {"deepseekv32", {"deepseek_v32"}},
        {"deepseekv4", {"deepseek_v4", "deepseek_v4_mtp"}},
        {"glm5", {"glm_moe_dsa", "glm_moe_dsa_mtp"}},
        // GLM-4.5 and GLM-4.7 are not supported for tool call parser
        // auto-selection
        // {"glm45", {"glm4_moe"}},
        // {"glm47", {"glm4_moe"}},

};

std::string get_auto_paser_map_supported() {
  std::vector<std::string> keys;
  for (const auto& [key, value] : auto_paser_map) {
    for (const auto& v : value) {
      keys.push_back(v);
    }
  }
  return absl::StrJoin(keys, ", ");
}

const std::unordered_map<std::string,
                         std::function<std::unique_ptr<BaseFormatDetector>()>>
    detector_factories = {
        {"qwen25", [] { return std::make_unique<Qwen25Detector>(); }},
        {"qwen3_coder", [] { return std::make_unique<Qwen3CoderDetector>(); }},
        {"kimi_k2", [] { return std::make_unique<KimiK2Detector>(); }},
        {"deepseekv3", [] { return std::make_unique<DeepSeekV3Detector>(); }},
        {"deepseekv32", [] { return std::make_unique<DeepSeekV32Detector>(); }},
        {"deepseekv4", [] { return std::make_unique<DeepSeekV4Detector>(); }},
        {"glm45", [] { return std::make_unique<Glm45Detector>(); }},
        {"glm47", [] { return std::make_unique<Glm47Detector>(); }},
        // glm5 use glm47 detector
        {"glm5", [] { return std::make_unique<Glm47Detector>(); }},
};

std::string get_supported_detector_factories() {
  std::vector<std::string> keys;
  for (const auto& [key, value] : detector_factories) {
    keys.push_back(key);
  }
  return absl::StrJoin(keys, ", ");
}

}  // namespace

std::string FunctionCallParser::get_parser_auto(const std::string& parser,
                                                const std::string& model_type) {
  if (parser.empty()) {
    return "";
  }
  if (parser == "auto") {
    // find the tool call parser that supports the model type
    for (const auto& [key, value] : auto_paser_map) {
      if (std::find(value.begin(), value.end(), model_type) != value.end()) {
        LOG(INFO) << "Using tool call parser: " << key
                  << " for model type: " << model_type;
        return key;
      }
    }
    LOG(FATAL) << "Unsupported model type for auto tool call parser: "
               << model_type << ". Supported model types are: "
               << get_auto_paser_map_supported();
    return "";
  } else {
    // check if the tool call parser is supported
    if (parser == "qwen2" || parser == "qwen3") {
      return "qwen25";
    }
    if (parser == "qwen35") {
      return "qwen3_coder";
    }
    if (detector_factories.find(parser) != detector_factories.end()) {
      return parser;
    }
    LOG(FATAL) << "Unsupported tool call parser: " << parser
               << ". Supported parsers are: "
               << get_supported_detector_factories();
    return "";
  }
}

FunctionCallParser::FunctionCallParser(const std::vector<JsonTool>& tools,
                                       const std::string& tool_call_parser)
    : tools_(tools) {
  detector_ = create_detector(tool_call_parser);
  CHECK(detector_ != nullptr)
      << "Unsupported tool_call_parser: " << tool_call_parser;
}

bool FunctionCallParser::has_tool_call(const std::string& text) const {
  return detector_->has_tool_call(text);
}

std::tuple<std::string, std::vector<ToolCallItem>>
FunctionCallParser::parse_non_stream(const std::string& full_text) {
  StreamingParseResult parsed_result =
      detector_->detect_and_parse(full_text, tools_);

  if (!parsed_result.calls.empty()) {
    return std::make_tuple(parsed_result.normal_text, parsed_result.calls);
  } else {
    return std::make_tuple(full_text, std::vector<ToolCallItem>());
  }
}

StreamingParseResult FunctionCallParser::parse_streaming_increment(
    const std::string& new_text) {
  return detector_->parse_streaming_increment(new_text, tools_);
}

std::unique_ptr<BaseFormatDetector> FunctionCallParser::create_detector(
    const std::string& tool_call_parser) {
  if (tool_call_parser.empty()) {
    return nullptr;
  }

  auto it = detector_factories.find(tool_call_parser);
  if (it != detector_factories.end()) {
    return it->second();
  }
  LOG(ERROR) << "Unsupported tool call parser: " << tool_call_parser;

  return nullptr;
}

namespace utils {

std::vector<ToolCallItem> parse_function_calls(
    const std::string& text,
    const std::vector<JsonTool>& tools,
    const std::string& parser_type) {
  try {
    FunctionCallParser parser(tools, parser_type);
    auto [normal_text, calls] = parser.parse_non_stream(text);
    return calls;
  } catch (const std::exception& e) {
    LOG(ERROR) << "Error parsing function calls: " << e.what();
    return {};
  }
}

bool has_function_calls(const std::string& text,
                        const std::string& parser_type) {
  try {
    FunctionCallParser parser({}, parser_type);
    return parser.has_tool_call(text);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Error checking function calls: " << e.what();
    return false;
  }
}

StreamingParseResult parse_streaming_increment(
    const std::string& new_text,
    const std::vector<JsonTool>& tools,
    const std::string& parser_type) {
  try {
    FunctionCallParser parser(tools, parser_type);
    return parser.parse_streaming_increment(new_text);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Error in streaming parsing: " << e.what();
    return StreamingParseResult();
  }
}

thread_local ShortUUID short_uuid;

std::string generate_tool_call_id() { return "call_" + short_uuid.random(); }

}  // namespace utils

}  // namespace function_call
}  // namespace xllm

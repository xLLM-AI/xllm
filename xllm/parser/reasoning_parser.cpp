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

#include "xllm/parser/reasoning_parser.h"

#include <glog/logging.h>

namespace xllm {

ReasoningParser::ReasoningParser(const std::string& model_type,
                                 bool stream_reasoning,
                                 bool force_reasoning) {
  detector_ = DetectorRegistry::get_instance().get_detector(
      model_type, stream_reasoning, force_reasoning);
}

ReasoningResult ReasoningParser::parse_non_stream(const std::string& text) {
  return detector_->detect_and_parse(const_cast<std::string&>(text));
}

ReasoningResult ReasoningParser::parse_stream_chunk(
    const std::string& chunk_text) {
  return detector_->parse_streaming_increment(
      const_cast<std::string&>(chunk_text));
}

std::string ReasoningParser::get_parser_auto(const std::string& parser,
                                             const std::string& model_type) {
  if (parser.empty()) {
    return "";
  }
  auto& registry = DetectorRegistry::get_instance();
  if (parser == "auto") {
    // find the reasoning parser that supports the model type
    std::string parser_name =
        registry.get_parser_name_by_model_type(model_type);
    LOG(INFO) << "Using reasoning parser: " << parser_name
              << " for model type: " << model_type;
    return parser_name;
  } else {
    // check if the reasoning parser is supported
    if (registry.has_detector(parser)) {
      return parser;
    }
    LOG(FATAL) << "Unsupported reasoning parser: " << parser
               << ". Supported parsers are: "
               << registry.get_supported_parsers();
    return "";
  }
}

}  // namespace xllm

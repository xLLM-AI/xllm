/* Copyright 2026 The xLLM Authors.

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

#include <functional>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "api_service/stream_output_parser.h"
#include "core/framework/request/request_output.h"

namespace xllm::api_service {

class ResponsesOutput final {
 public:
  using EventWriter = std::function<bool(const nlohmann::json&)>;

  ResponsesOutput(nlohmann::json initial_response,
                  bool stream,
                  std::vector<JsonTool> tools,
                  std::string tool_parser,
                  std::string reasoning_parser,
                  bool force_reasoning,
                  EventWriter event_writer);

  bool append(const RequestOutput& output);
  bool fail(StatusCode code, const std::string& message);
  const nlohmann::json& snapshot() const { return response_; }
  bool started() const { return started_; }
  bool finished() const { return finished_; }
  const Status& status() const { return status_; }

 private:
  bool start(const RequestOutput& output);
  bool emit(nlohmann::json event);
  bool parse_text(const std::string& text);
  bool parse_normal_text(const std::string& text);
  bool append_tool_output(const function_call::StreamingParseResult& parsed);
  bool append_text(const std::string& type, const std::string& text);
  bool append_tool(const function_call::ToolCallItem& call);
  bool flush_parsers(bool incomplete);
  bool flush_tools(bool incomplete);
  bool complete(const std::string& status,
                const nlohmann::json& incomplete_details);
  bool finalize_items(const std::string& status);
  bool set_usage(const Usage& usage);
  bool set_error(const std::string& message);
  std::optional<size_t> add_text_item(const std::string& type);
  std::pair<Status, std::string> take_utf8(const std::string& text, bool final);

  nlohmann::json response_;
  const bool stream_;
  std::vector<JsonTool> tools_;
  std::string tool_parser_format_;
  std::string reasoning_parser_format_;
  bool force_reasoning_;
  EventWriter event_writer_;
  std::shared_ptr<StreamOutputParser> parser_;
  std::unordered_map<int32_t, size_t> tool_indexes_;
  std::optional<size_t> text_item_index_;
  std::optional<size_t> reasoning_item_index_;
  std::string utf8_tail_;
  std::string tool_text_;
  std::string finish_reason_;
  Status status_;
  int64_t sequence_number_ = 0;
  bool started_ = false;
  bool finished_ = false;
  bool failing_ = false;
};

}  // namespace xllm::api_service

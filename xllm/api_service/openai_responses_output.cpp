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

#include "api_service/openai_responses_output.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>

#include <algorithm>
#include <cstdint>
#include <utility>

#include "core/util/uuid.h"

namespace xllm::api_service {
namespace {

std::string item_id(const std::string& prefix) {
  return prefix + ShortUUID().random();
}

nlohmann::json item_location(const nlohmann::json& item, size_t index) {
  return {{"item_id", item.at("id")},
          {"output_index", index},
          {"content_index", 0}};
}

}  // namespace

ResponsesOutput::ResponsesOutput(nlohmann::json initial_response,
                                 bool stream,
                                 std::vector<JsonTool> tools,
                                 std::string tool_parser,
                                 std::string reasoning_parser,
                                 bool force_reasoning,
                                 EventWriter event_writer)
    : response_(std::move(initial_response)),
      stream_(stream),
      tools_(std::move(tools)),
      tool_parser_format_(std::move(tool_parser)),
      reasoning_parser_format_(std::move(reasoning_parser)),
      force_reasoning_(force_reasoning),
      event_writer_(std::move(event_writer)) {
  CHECK(response_.is_object());
  CHECK(!stream_ || event_writer_);
  response_["output"] = nlohmann::json::array();
  response_["usage"] = nullptr;
  response_["status"] = "in_progress";
  response_["error"] = nullptr;
  response_["incomplete_details"] = nullptr;
}

bool ResponsesOutput::emit(nlohmann::json event) {
  if (!stream_) {
    return true;
  }
  event["sequence_number"] = sequence_number_++;
  if (!event_writer_(event)) {
    status_ =
        Status(StatusCode::CANCELLED, "The Responses client disconnected.");
    response_["status"] = "cancelled";
    finished_ = true;
    return false;
  }
  return true;
}

bool ResponsesOutput::start(const RequestOutput& output) {
  if (started_) {
    if (output.force_reasoning.has_value() &&
        output.force_reasoning.value() != force_reasoning_) {
      return set_error("Generation changed its initial reasoning state.");
    }
    return true;
  }
  if (output.force_reasoning.has_value()) {
    force_reasoning_ = output.force_reasoning.value();
  }
  parser_ = std::make_shared<StreamOutputParser>(tools_,
                                                 tool_parser_format_,
                                                 reasoning_parser_format_,
                                                 force_reasoning_,
                                                 output.force_reasoning,
                                                 /*lossless_reasoning=*/true);
  started_ = true;
  return emit({{"type", "response.created"}, {"response", response_}}) &&
         emit({{"type", "response.in_progress"}, {"response", response_}});
}

std::pair<Status, std::string> ResponsesOutput::take_utf8(
    const std::string& text,
    bool final) {
  std::string combined = std::move(utf8_tail_);
  utf8_tail_.clear();
  combined += text;
  size_t position = 0;
  while (position < combined.size()) {
    const uint8_t first = static_cast<uint8_t>(combined[position]);
    size_t length = 0;
    if (first < 0x80) {
      length = 1;
    } else if (first >= 0xC2 && first <= 0xDF) {
      length = 2;
    } else if (first >= 0xE0 && first <= 0xEF) {
      length = 3;
    } else if (first >= 0xF0 && first <= 0xF4) {
      length = 4;
    } else {
      return {Status(StatusCode::UNKNOWN, "Generation produced invalid UTF-8."),
              ""};
    }
    const size_t available = std::min(length, combined.size() - position);
    for (size_t byte = 1; byte < available; ++byte) {
      const uint8_t value = static_cast<uint8_t>(combined[position + byte]);
      if ((value & 0xC0) != 0x80 ||
          (byte == 1 && ((first == 0xE0 && value < 0xA0) ||
                         (first == 0xED && value >= 0xA0) ||
                         (first == 0xF0 && value < 0x90) ||
                         (first == 0xF4 && value > 0x8F)))) {
        return {
            Status(StatusCode::UNKNOWN, "Generation produced invalid UTF-8."),
            ""};
      }
    }
    if (available < length) {
      if (final) {
        return {Status(StatusCode::UNKNOWN,
                       "Generation ended inside a UTF-8 code point."),
                ""};
      }
      utf8_tail_ = combined.substr(position);
      combined.resize(position);
      break;
    }
    position += length;
  }
  return {Status(), std::move(combined)};
}

std::optional<size_t> ResponsesOutput::add_text_item(const std::string& type) {
  const bool reasoning = type == "reasoning";
  const size_t index = response_["output"].size();
  nlohmann::json item = {{"id", item_id(reasoning ? "rs_" : "msg_")},
                         {"type", type},
                         {"status", "in_progress"},
                         {"content", nlohmann::json::array()}};
  if (reasoning) {
    item["summary"] = nlohmann::json::array();
  } else {
    item["role"] = "assistant";
  }
  response_["output"].emplace_back(item);
  if (!emit({{"type", "response.output_item.added"},
             {"output_index", index},
             {"item", item}})) {
    return std::nullopt;
  }
  nlohmann::json part = {{"type", reasoning ? "reasoning_text" : "output_text"},
                         {"text", ""}};
  if (!reasoning) {
    part["annotations"] = nlohmann::json::array();
    part["logprobs"] = nlohmann::json::array();
    auto event = item_location(item, index);
    event["type"] = "response.content_part.added";
    event["part"] = part;
    if (!emit(std::move(event))) {
      return std::nullopt;
    }
  }
  response_["output"][index]["content"].emplace_back(std::move(part));
  return index;
}

bool ResponsesOutput::append_text(const std::string& type,
                                  const std::string& text) {
  if (text.empty()) {
    return true;
  }
  const bool reasoning = type == "reasoning";
  auto& item_index = reasoning ? reasoning_item_index_ : text_item_index_;
  if (!item_index.has_value()) {
    item_index = add_text_item(type);
    if (!item_index.has_value()) {
      return false;
    }
  }
  auto& item = response_["output"][item_index.value()];
  item["content"][0]["text"].get_ref<std::string&>() += text;
  auto event = item_location(item, item_index.value());
  event["type"] = reasoning ? "response.reasoning_text.delta"
                            : "response.output_text.delta";
  event["delta"] = text;
  if (!reasoning) {
    event["logprobs"] = nlohmann::json::array();
  }
  return emit(std::move(event));
}

bool ResponsesOutput::append_tool(const function_call::ToolCallItem& call) {
  if (call.tool_index < 0) {
    return set_error("Function parser returned a negative tool index.");
  }
  auto position = tool_indexes_.find(call.tool_index);
  if (position == tool_indexes_.end()) {
    if (!call.name.has_value() || call.name->empty() ||
        static_cast<size_t>(call.tool_index) != tool_indexes_.size()) {
      return set_error(
          "Function arguments arrived without a contiguous named call.");
    }
    const auto declared = std::find_if(
        tools_.begin(), tools_.end(), [&call](const JsonTool& tool) {
          return tool.function.name == call.name.value();
        });
    if (declared == tools_.end()) {
      return set_error("Model generated an undefined function: " +
                       call.name.value());
    }
    const size_t index = response_["output"].size();
    nlohmann::json item = {{"id", item_id("fc_")},
                           {"type", "function_call"},
                           {"status", "in_progress"},
                           {"call_id", item_id("call_")},
                           {"name", call.name.value()},
                           {"arguments", ""}};
    response_["output"].emplace_back(item);
    position = tool_indexes_.emplace(call.tool_index, index).first;
    if (!emit({{"type", "response.output_item.added"},
               {"output_index", index},
               {"item", item}})) {
      return false;
    }
  }
  auto& item = response_["output"][position->second];
  if (call.name.has_value() && item["name"] != call.name.value()) {
    return set_error("Function parser changed an emitted function name.");
  }
  if (call.parameters.empty()) {
    return true;
  }
  item["arguments"].get_ref<std::string&>() += call.parameters;
  return emit({{"type", "response.function_call_arguments.delta"},
               {"item_id", item["id"]},
               {"output_index", position->second},
               {"delta", call.parameters}});
}

bool ResponsesOutput::parse_normal_text(const std::string& text) {
  auto* tool_parser = parser_->get_tool_call_parser(/*index=*/0);
  if (tool_parser == nullptr) {
    return append_text("message", text);
  }
  if (!stream_) {
    tool_text_ += text;
    return true;
  }
  return text.empty() ||
         append_tool_output(tool_parser->parse_streaming_increment(text));
}

bool ResponsesOutput::append_tool_output(
    const function_call::StreamingParseResult& parsed) {
  if (!append_text("message", parsed.normal_text)) {
    return false;
  }
  for (const auto& call : parsed.calls) {
    if (!append_tool(call)) {
      return false;
    }
  }
  return true;
}

bool ResponsesOutput::parse_text(const std::string& text) {
  auto [utf8_status, complete_text] = take_utf8(text, /*final=*/false);
  if (!utf8_status.ok()) {
    return set_error(utf8_status.message());
  }
  auto* reasoning_parser = parser_->get_reasoning_parser(/*index=*/0);
  if (reasoning_parser == nullptr) {
    return parse_normal_text(complete_text);
  }
  const auto parsed = reasoning_parser->parse_stream_chunk(complete_text);
  if (parsed.reasoning_text.has_value() &&
      !append_text("reasoning", parsed.reasoning_text.value())) {
    return false;
  }
  return !parsed.normal_text.has_value() ||
         parse_normal_text(parsed.normal_text.value());
}

bool ResponsesOutput::flush_parsers(bool incomplete) {
  auto [utf8_status, complete_text] = take_utf8("", /*final=*/true);
  if (!utf8_status.ok()) {
    return set_error(utf8_status.message());
  }
  CHECK(complete_text.empty());
  auto* reasoning_parser = parser_->get_reasoning_parser(/*index=*/0);
  if (reasoning_parser != nullptr) {
    const auto remaining = reasoning_parser->finish_stream();
    if (remaining.reasoning_text.has_value() &&
        !append_text("reasoning", remaining.reasoning_text.value())) {
      return false;
    }
    if (remaining.normal_text.has_value() &&
        !parse_normal_text(remaining.normal_text.value())) {
      return false;
    }
  }
  return flush_tools(incomplete);
}

bool ResponsesOutput::flush_tools(bool incomplete) {
  auto* tool_parser = parser_->get_tool_call_parser(/*index=*/0);
  if (tool_parser == nullptr) {
    return true;
  }
  if (!stream_) {
    std::string text = std::exchange(tool_text_, "");
    if (!incomplete) {
      auto [normal_text, calls] = tool_parser->parse_non_stream(text);
      for (size_t index = 0; index < calls.size(); ++index) {
        calls[index].tool_index = static_cast<int32_t>(index);
      }
      return append_tool_output({std::move(normal_text), std::move(calls)});
    }
    if (!text.empty() &&
        !append_tool_output(tool_parser->parse_streaming_increment(text))) {
      return false;
    }
  }
  const auto remaining = tool_parser->finish_stream();
  if (!append_tool_output(remaining.output)) {
    return false;
  }
  if (!incomplete && remaining.has_pending_tool) {
    return set_error("Generation ended inside a function call.");
  }
  return true;
}

bool ResponsesOutput::set_usage(const Usage& usage) {
  if (usage.num_prompt_tokens < 0 || usage.num_generated_tokens < 0 ||
      usage.num_cached_tokens < 0 ||
      usage.num_total_tokens !=
          usage.num_prompt_tokens + usage.num_generated_tokens ||
      usage.num_cached_tokens > usage.num_prompt_tokens) {
    return set_error("Generation returned inconsistent token usage.");
  }
  response_["usage"] = {
      {"input_tokens", usage.num_prompt_tokens},
      {"output_tokens", usage.num_generated_tokens},
      {"total_tokens", usage.num_total_tokens},
      {"input_tokens_details",
       {{"cached_tokens", usage.num_cached_tokens}, {"cache_write_tokens", 0}}},
      {"output_tokens_details", {{"reasoning_tokens", 0}}}};
  return true;
}

bool ResponsesOutput::append(const RequestOutput& output) {
  if (finished_) {
    return false;
  }
  if (output.cancelled) {
    status_ = Status(StatusCode::CANCELLED, "Generation was cancelled.");
    response_["status"] = "cancelled";
    finished_ = true;
    return false;
  }
  if (output.status.has_value() && !output.status->ok()) {
    return fail(output.status->code(), output.status->message());
  }
  if (!start(output)) {
    return false;
  }
  if (output.usage.has_value() && !set_usage(output.usage.value())) {
    return false;
  }
  for (const auto& sequence : output.outputs) {
    if (sequence.index != 0) {
      return set_error("Responses generation returned more than one sequence.");
    }
    if (!parse_text(sequence.text)) {
      return false;
    }
    if (sequence.finish_reason.has_value()) {
      if (!finish_reason_.empty() &&
          finish_reason_ != sequence.finish_reason.value()) {
        return set_error("Generation changed its finish reason.");
      }
      finish_reason_ = sequence.finish_reason.value();
    }
  }
  if (!output.finished) {
    return true;
  }
  if (!output.usage.has_value()) {
    return set_error("Completed generation did not supply token usage.");
  }
  const bool incomplete = finish_reason_ == "length";
  if (!incomplete && finish_reason_ != "stop" &&
      finish_reason_ != "function_call") {
    return set_error("Generation returned an unsupported finish reason: " +
                     finish_reason_);
  }
  if (!flush_parsers(incomplete)) {
    return false;
  }
  return complete(incomplete ? "incomplete" : "completed",
                  incomplete ? nlohmann::json{{"reason", "max_output_tokens"}}
                             : nlohmann::json(nullptr));
}

bool ResponsesOutput::finalize_items(const std::string& status) {
  if (status == "completed") {
    for (const auto& item : response_["output"]) {
      if (item["type"] == "function_call") {
        const auto arguments =
            nlohmann::json::parse(item["arguments"].get<std::string>(),
                                  nullptr,
                                  /*allow_exceptions=*/false);
        if (!arguments.is_object()) {
          return set_error(
              "Completed function arguments are not a JSON object.");
        }
      }
    }
  }
  const std::string item_status =
      status == "completed" ? "completed" : "incomplete";
  for (size_t index = 0; index < response_["output"].size(); ++index) {
    auto& item = response_["output"][index];
    item["status"] = item_status;
    if (item["type"] == "function_call") {
      if (!emit({{"type", "response.function_call_arguments.done"},
                 {"item_id", item["id"]},
                 {"output_index", index},
                 {"name", item["name"]},
                 {"arguments", item["arguments"]}})) {
        return false;
      }
    } else {
      const bool reasoning = item["type"] == "reasoning";
      auto event = item_location(item, index);
      event["type"] = reasoning ? "response.reasoning_text.done"
                                : "response.output_text.done";
      event["text"] = item["content"][0]["text"];
      if (!reasoning) {
        event["logprobs"] = nlohmann::json::array();
      }
      if (!emit(std::move(event))) {
        return false;
      }
      if (!reasoning) {
        event = item_location(item, index);
        event["type"] = "response.content_part.done";
        event["part"] = item["content"][0];
        if (!emit(std::move(event))) {
          return false;
        }
      }
    }
    if (!emit({{"type", "response.output_item.done"},
               {"output_index", index},
               {"item", item}})) {
      return false;
    }
  }
  return true;
}

bool ResponsesOutput::complete(const std::string& status,
                               const nlohmann::json& incomplete_details) {
  if (finished_) {
    return false;
  }
  if (!finalize_items(status)) {
    return false;
  }
  response_["status"] = status;
  response_["completed_at"] =
      status == "completed" ? nlohmann::json(absl::ToUnixSeconds(absl::Now()))
                            : nlohmann::json(nullptr);
  response_["incomplete_details"] = incomplete_details;
  finished_ = true;
  return emit({{"type", "response." + status}, {"response", response_}});
}

bool ResponsesOutput::set_error(const std::string& message) {
  return fail(StatusCode::UNKNOWN, message);
}

bool ResponsesOutput::fail(StatusCode code, const std::string& message) {
  if (finished_ || failing_) {
    return false;
  }
  failing_ = true;
  status_ = Status(code, message);
  response_["error"] = {{"code", "server_error"}, {"message", message}};
  if (started_) {
    flush_tools(/*incomplete=*/true);
    if (finished_) {
      failing_ = false;
      return false;
    }
  }
  response_["status"] = "failed";
  for (auto& item : response_["output"]) {
    item["status"] = "incomplete";
  }
  finished_ = true;
  failing_ = false;
  if (started_) {
    emit({{"type", "response.failed"}, {"response", response_}});
  }
  return false;
}

}  // namespace xllm::api_service

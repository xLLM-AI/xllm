/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "framework/chat_template/deepseek_v4_cpp_template.h"

#include <absl/strings/str_join.h>
#include <absl/strings/str_replace.h>
#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace xllm {
namespace {

// ============================================================
// Special Tokens
// ============================================================

constexpr const char* kBosToken = "<｜begin▁of▁sentence｜>";
constexpr const char* kEosToken = "<｜end▁of▁sentence｜>";
constexpr const char* kThinkingStartToken = "<think>";
constexpr const char* kThinkingEndToken = "</think>";
constexpr const char* kDsmlToken = "｜DSML｜";

constexpr const char* kUserSpToken = "<｜User｜>";
constexpr const char* kAssistantSpToken = "<｜Assistant｜>";
constexpr const char* kLatestReminderSpToken = "<｜latest_reminder｜>";

constexpr const char* kToolCallsBlockName = "tool_calls";

// Task special tokens
const std::unordered_map<std::string, std::string>& task_sp_tokens() {
  static const std::unordered_map<std::string, std::string> kMap = {
      {"action", "<｜action｜>"},
      {"query", "<｜query｜>"},
      {"authority", "<｜authority｜>"},
      {"domain", "<｜domain｜>"},
      {"title", "<｜title｜>"},
      {"read_url", "<｜read_url｜>"},
  };
  return kMap;
}

// ============================================================
// Templates
// ============================================================

constexpr const char* kToolsTemplate =
    "## Tools\n"
    "\n"
    "You have access to a set of tools to help "
    "answer the user's question. You can invoke "
    "tools by writing a "
    "\"<{dsml_token}tool_calls>\" block like the "
    "following:\n"
    "\n"
    "<{dsml_token}tool_calls>\n"
    "<{dsml_token}invoke name=\"$TOOL_NAME\">\n"
    "<{dsml_token}parameter "
    "name=\"$PARAMETER_NAME\" "
    "string=\"true|false\">"
    "$PARAMETER_VALUE"
    "</{dsml_token}parameter>\n"
    "...\n"
    "</{dsml_token}invoke>\n"
    "<{dsml_token}invoke name=\"$TOOL_NAME2\">\n"
    "...\n"
    "</{dsml_token}invoke>\n"
    "</{dsml_token}tool_calls>\n"
    "\n"
    "String parameters should be specified as is "
    "and set `string=\"true\"`. For all other types "
    "(numbers, booleans, arrays, objects), pass the "
    "value in JSON format and set "
    "`string=\"false\"`.\n"
    "\n"
    "If thinking_mode is enabled (triggered by "
    "{thinking_start_token}), you MUST output your "
    "complete reasoning inside "
    "{thinking_start_token}..."
    "{thinking_end_token} BEFORE any tool calls or "
    "final response.\n"
    "\n"
    "Otherwise, output directly after "
    "{thinking_end_token} with tool calls or final "
    "response.\n"
    "\n"
    "### Available Tool Schemas\n"
    "\n"
    "{tool_schemas}\n"
    "\n"
    "You MUST strictly follow the above defined "
    "tool name and parameter schemas to invoke "
    "tool calls.\n";

constexpr const char* kResponseFormatTemplate =
    "## Response Format:\n\n"
    "You MUST strictly adhere to the following "
    "schema to reply:\n{schema}";

constexpr const char* kToolCallTemplate =
    "<{dsml_token}invoke name=\"{name}\">\n"
    "{arguments}\n"
    "</{dsml_token}invoke>";

constexpr const char* kToolOutputTemplate =
    "<tool_result>{content}</tool_result>";

constexpr const char* kReasoningEffortMax =
    "Reasoning Effort: Absolute maximum with no "
    "shortcuts permitted.\n"
    "You MUST be very thorough in your thinking "
    "and comprehensively decompose the problem to "
    "resolve the root cause, rigorously "
    "stress-testing your logic against all "
    "potential paths, edge cases, and adversarial "
    "scenarios.\n"
    "Explicitly write out your entire deliberation "
    "process, documenting every intermediate step, "
    "considered alternative, and rejected "
    "hypothesis to ensure absolutely no assumption "
    "is left unchecked.\n\n";

constexpr const char* kThinkingModeThinking = "thinking";
constexpr const char* kThinkingModeChat = "chat";

constexpr const char* kRoleSystem = "system";
constexpr const char* kRoleDeveloper = "developer";
constexpr const char* kRoleUser = "user";
constexpr const char* kRoleTool = "tool";
constexpr const char* kRoleAssistant = "assistant";
constexpr const char* kRoleLatestReminder = "latest_reminder";
constexpr const char* kRoleDirectSearchResults = "direct_search_results";

// ============================================================
// Utility Functions
// ============================================================

std::string to_json(const nlohmann::ordered_json& value) {
  try {
    return value.dump(
        /*indent=*/-1,
        /*indent_char=*/' ',
        /*ensure_ascii=*/false,
        nlohmann::json::error_handler_t::replace);
  } catch (const std::exception&) {
    return value.dump(
        /*indent=*/-1,
        /*indent_char=*/' ',
        /*ensure_ascii=*/true,
        nlohmann::json::error_handler_t::replace);
  }
}

nlohmann::ordered_json parse_json_object_or_empty(const std::string& text) {
  if (text.empty()) {
    return nlohmann::ordered_json::object();
  }
  try {
    nlohmann::ordered_json parsed = nlohmann::json::parse(text);
    if (parsed.is_object()) {
      return parsed;
    }
    return nlohmann::ordered_json::object();
  } catch (const std::exception&) {
    nlohmann::ordered_json fallback = nlohmann::ordered_json::object();
    fallback["arguments"] = text;
    return fallback;
  }
}

bool get_thinking_enabled(const nlohmann::ordered_json& kwargs) {
  if (kwargs.contains("thinking") && kwargs["thinking"].is_boolean()) {
    return kwargs["thinking"].get<bool>();
  }
  if (kwargs.contains("enable_thinking") &&
      kwargs["enable_thinking"].is_boolean()) {
    return kwargs["enable_thinking"].get<bool>();
  }
  return false;
}

std::string get_thinking_mode(const nlohmann::ordered_json& kwargs) {
  if (kwargs.contains("thinking_mode") && kwargs["thinking_mode"].is_string()) {
    return kwargs["thinking_mode"].get<std::string>();
  }
  return get_thinking_enabled(kwargs) ? kThinkingModeThinking
                                      : kThinkingModeChat;
}

std::string get_reasoning_effort(const nlohmann::ordered_json& kwargs) {
  if (kwargs.contains("reasoning_effort") &&
      kwargs["reasoning_effort"].is_string()) {
    return kwargs["reasoning_effort"].get<std::string>();
  }
  return "";
}

std::vector<nlohmann::ordered_json> tool_calls_from_openai_format(
    const Message::ToolCallVec& tool_calls) {
  std::vector<nlohmann::ordered_json> out;
  out.reserve(tool_calls.size());
  for (const Message::ToolCall& tc : tool_calls) {
    nlohmann::ordered_json item;
    item["id"] = tc.id;
    item["name"] = tc.function.name;
    item["arguments"] = parse_json_object_or_empty(tc.function.arguments);
    out.emplace_back(std::move(item));
  }
  return out;
}

std::string encode_arguments_to_dsml(const nlohmann::ordered_json& tool_call) {
  std::ostringstream oss;
  nlohmann::ordered_json arguments = nlohmann::json::object();
  if (tool_call.contains("arguments")) {
    arguments = tool_call["arguments"];
    if (arguments.is_string()) {
      arguments = parse_json_object_or_empty(arguments.get<std::string>());
    }
  }
  bool first = true;
  for (auto it = arguments.begin(); it != arguments.end(); ++it) {
    if (!first) {
      oss << "\n";
    }
    first = false;
    const nlohmann::ordered_json& value = it.value();
    bool is_string = value.is_string();
    oss << "<" << kDsmlToken << "parameter name=\"" << it.key()
        << "\" string=\"" << (is_string ? "true" : "false") << "\">";
    if (is_string) {
      oss << value.get<std::string>();
    } else {
      oss << to_json(value);
    }
    oss << "</" << kDsmlToken << "parameter>";
  }
  return oss.str();
}

int32_t find_last_user_index(const nlohmann::ordered_json& messages) {
  for (int32_t idx = static_cast<int32_t>(messages.size()) - 1; idx >= 0;
       --idx) {
    std::string role = messages[idx].value("role", "");
    if (role == kRoleUser || role == kRoleDeveloper) {
      return idx;
    }
  }
  return -1;
}

std::string get_text_content(const Message::Content& content) {
  if (std::holds_alternative<std::string>(content)) {
    return std::get<std::string>(content);
  }
  return "";
}

// ============================================================
// Tool Rendering
// ============================================================

std::string render_tools(const nlohmann::ordered_json& tools) {
  std::vector<std::string> schemas;
  schemas.reserve(tools.size());
  for (const nlohmann::ordered_json& tool : tools) {
    schemas.emplace_back(to_json(tool));
  }
  return absl::StrReplaceAll(std::string(kToolsTemplate),
                             {{"{tool_schemas}", absl::StrJoin(schemas, "\n")},
                              {"{dsml_token}", kDsmlToken},
                              {"{thinking_start_token}", kThinkingStartToken},
                              {"{thinking_end_token}", kThinkingEndToken}});
}

std::string render_tool_calls(
    const std::vector<nlohmann::ordered_json>& tool_calls) {
  if (tool_calls.empty()) {
    return "";
  }
  std::vector<std::string> chunks;
  chunks.reserve(tool_calls.size());
  for (const nlohmann::ordered_json& tc : tool_calls) {
    std::string invoke =
        absl::StrReplaceAll(std::string(kToolCallTemplate),
                            {{"{dsml_token}", kDsmlToken},
                             {"{name}", tc.value("name", "")},
                             {"{arguments}", encode_arguments_to_dsml(tc)}});
    chunks.emplace_back(std::move(invoke));
  }
  std::string block = std::string("<") + kDsmlToken + kToolCallsBlockName +
                      ">\n" + absl::StrJoin(chunks, "\n") + "\n</" +
                      kDsmlToken + kToolCallsBlockName + ">";
  return "\n\n" + block;
}

// ============================================================
// Preprocessing: merge tool messages
// ============================================================

nlohmann::ordered_json merge_tool_messages(
    const nlohmann::ordered_json& messages) {
  nlohmann::ordered_json merged = nlohmann::json::array();

  for (const auto& msg : messages) {
    std::string role = msg.value("role", "");

    if (role == kRoleTool) {
      nlohmann::ordered_json tool_block;
      tool_block["type"] = "tool_result";
      tool_block["tool_use_id"] = msg.value("tool_call_id", "");
      tool_block["content"] = msg.value("content", "");

      if (!merged.empty() && merged.back().value("role", "") == kRoleUser &&
          merged.back().contains("content_blocks")) {
        merged.back()["content_blocks"].emplace_back(std::move(tool_block));
      } else {
        nlohmann::ordered_json new_msg;
        new_msg["role"] = kRoleUser;
        new_msg["content_blocks"] = nlohmann::json::array({tool_block});
        merged.emplace_back(std::move(new_msg));
      }
    } else if (role == kRoleUser) {
      nlohmann::ordered_json text_block;
      text_block["type"] = "text";
      text_block["text"] = msg.value("content", "");

      if (!merged.empty() && merged.back().value("role", "") == kRoleUser &&
          merged.back().contains("content_blocks") &&
          !merged.back().contains("task")) {
        merged.back()["content_blocks"].emplace_back(std::move(text_block));
      } else {
        nlohmann::ordered_json new_msg;
        new_msg["role"] = kRoleUser;
        new_msg["content"] = msg.value("content", "");
        new_msg["content_blocks"] = nlohmann::json::array({text_block});
        for (const auto& key : {"task", "wo_eos", "mask"}) {
          if (msg.contains(key)) {
            new_msg[key] = msg[key];
          }
        }
        merged.emplace_back(std::move(new_msg));
      }
    } else {
      merged.emplace_back(msg);
    }
  }
  return merged;
}

nlohmann::ordered_json sort_tool_results_by_call_order(
    const nlohmann::ordered_json& messages) {
  nlohmann::ordered_json result = messages;
  std::unordered_map<std::string, int32_t> last_tc_order;

  for (auto& msg : result) {
    std::string role = msg.value("role", "");
    if (role == kRoleAssistant && msg.contains("tool_calls") &&
        msg["tool_calls"].is_array()) {
      last_tc_order.clear();
      int32_t idx = 0;
      for (const auto& tc : msg["tool_calls"]) {
        std::string tc_id;
        if (tc.contains("id")) {
          tc_id = tc.value("id", "");
        } else if (tc.contains("function") && tc["function"].contains("id")) {
          tc_id = tc["function"].value("id", "");
        }
        if (!tc_id.empty()) {
          last_tc_order[tc_id] = idx;
        }
        ++idx;
      }
    } else if (role == kRoleUser && msg.contains("content_blocks")) {
      auto& blocks = msg["content_blocks"];
      int32_t tool_count = 0;
      for (const auto& b : blocks) {
        if (b.value("type", "") == "tool_result") {
          ++tool_count;
        }
      }
      if (tool_count > 1 && !last_tc_order.empty()) {
        std::vector<nlohmann::ordered_json> tool_blocks;
        for (const auto& b : blocks) {
          if (b.value("type", "") == "tool_result") {
            tool_blocks.emplace_back(b);
          }
        }
        std::sort(tool_blocks.begin(),
                  tool_blocks.end(),
                  [&](const nlohmann::ordered_json& a,
                      const nlohmann::ordered_json& b) {
                    auto ia = last_tc_order.find(a.value("tool_use_id", ""));
                    auto ib = last_tc_order.find(b.value("tool_use_id", ""));
                    int32_t oa = ia != last_tc_order.end() ? ia->second : 0;
                    int32_t ob = ib != last_tc_order.end() ? ib->second : 0;
                    return oa < ob;
                  });
        int32_t si = 0;
        nlohmann::ordered_json new_blocks = nlohmann::json::array();
        for (const auto& b : blocks) {
          if (b.value("type", "") == "tool_result") {
            new_blocks.emplace_back(tool_blocks[si++]);
          } else {
            new_blocks.emplace_back(b);
          }
        }
        msg["content_blocks"] = std::move(new_blocks);
      }
    }
  }
  return result;
}

// ============================================================
// Drop Thinking
// ============================================================

nlohmann::ordered_json drop_thinking_messages(
    const nlohmann::ordered_json& messages) {
  int32_t last_user_idx = find_last_user_index(messages);
  nlohmann::ordered_json result = nlohmann::json::array();

  for (int32_t idx = 0; idx < static_cast<int32_t>(messages.size()); ++idx) {
    std::string role = messages[idx].value("role", "");
    if (role == kRoleUser || role == kRoleSystem || role == kRoleTool ||
        role == kRoleLatestReminder || role == kRoleDirectSearchResults ||
        idx >= last_user_idx) {
      result.emplace_back(messages[idx]);
    } else if (role == kRoleAssistant) {
      nlohmann::ordered_json copied = messages[idx];
      copied.erase("reasoning");
      result.emplace_back(std::move(copied));
    }
    // developer before last_user_idx is dropped
  }
  return result;
}

// ============================================================
// Message Rendering
// ============================================================

nlohmann::ordered_json normalize_messages(
    const ChatMessages& messages,
    const std::vector<xllm::JsonTool>& tools) {
  nlohmann::ordered_json out = nlohmann::json::array();
  for (const Message& message : messages) {
    nlohmann::ordered_json item;
    item["role"] = message.role;
    item["content"] = get_text_content(message.content);
    if (message.reasoning_content.has_value()) {
      item["reasoning"] = message.reasoning_content.value();
    }
    if (message.tool_calls.has_value()) {
      item["tool_calls"] =
          tool_calls_from_openai_format(message.tool_calls.value());
    }
    if (message.tool_call_id.has_value()) {
      item["tool_call_id"] = message.tool_call_id.value();
    }
    out.emplace_back(std::move(item));
  }
  if (!tools.empty()) {
    nlohmann::ordered_json tools_json = nlohmann::json::array();
    for (const xllm::JsonTool& tool : tools) {
      nlohmann::ordered_json openai_tool;
      openai_tool["type"] = "function";
      nlohmann::ordered_json function;
      function["name"] = tool.function.name;
      function["description"] = tool.function.description;
      function["parameters"] = tool.function.parameters;
      openai_tool["function"] = std::move(function);
      tools_json.emplace_back(std::move(openai_tool));
    }
    nlohmann::ordered_json sys_msg;
    sys_msg["role"] = kRoleSystem;
    sys_msg["tools"] = std::move(tools_json);
    out.insert(out.begin(), std::move(sys_msg));
  }
  return out;
}

std::string render_message(const nlohmann::ordered_json& messages,
                           int32_t index,
                           const std::string& thinking_mode,
                           bool drop_thinking,
                           const std::string& reasoning_effort) {
  if (index < 0 || index >= static_cast<int32_t>(messages.size())) {
    LOG(FATAL) << "Message index out of range";
  }

  std::string prompt;
  const nlohmann::ordered_json& msg = messages[index];
  int32_t last_user_idx = find_last_user_index(messages);
  std::string role = msg.value("role", "");
  std::string content = msg.value("content", "");

  nlohmann::ordered_json tools = nlohmann::json::array();
  if (msg.contains("tools")) {
    tools = msg["tools"];
  }
  nlohmann::ordered_json response_format;
  if (msg.contains("response_format")) {
    response_format = msg["response_format"];
  }

  // Reasoning effort prefix at index 0
  if (index == 0 && thinking_mode == kThinkingModeThinking &&
      reasoning_effort == "max") {
    prompt += kReasoningEffortMax;
  }

  if (role == kRoleSystem) {
    prompt += content.empty() ? "" : content;
    if (tools.is_array() && !tools.empty()) {
      nlohmann::ordered_json fn_tools = nlohmann::json::array();
      for (const auto& t : tools) {
        if (t.contains("function")) {
          fn_tools.emplace_back(t["function"]);
        }
      }
      prompt += "\n\n" + render_tools(fn_tools);
    }
    if (!response_format.is_null()) {
      prompt += "\n\n" +
                absl::StrReplaceAll(std::string(kResponseFormatTemplate),
                                    {{"{schema}", to_json(response_format)}});
    }
  } else if (role == kRoleDeveloper) {
    if (content.empty()) {
      LOG(FATAL) << "Developer message content is empty";
    }
    std::string dev_content = std::string(kUserSpToken) + content;
    if (tools.is_array() && !tools.empty()) {
      nlohmann::ordered_json fn_tools = nlohmann::json::array();
      for (const auto& t : tools) {
        if (t.contains("function")) {
          fn_tools.emplace_back(t["function"]);
        }
      }
      dev_content += "\n\n" + render_tools(fn_tools);
    }
    if (!response_format.is_null()) {
      dev_content += "\n\n" + absl::StrReplaceAll(
                                  std::string(kResponseFormatTemplate),
                                  {{"{schema}", to_json(response_format)}});
    }
    prompt += dev_content;
  } else if (role == kRoleUser) {
    prompt += kUserSpToken;
    if (msg.contains("content_blocks") && msg["content_blocks"].is_array()) {
      std::vector<std::string> parts;
      for (const auto& block : msg["content_blocks"]) {
        std::string btype = block.value("type", "");
        if (btype == "text") {
          parts.emplace_back(block.value("text", ""));
        } else if (btype == "tool_result") {
          std::string tc = block.value("content", "");
          parts.emplace_back(absl::StrReplaceAll(
              std::string(kToolOutputTemplate), {{"{content}", tc}}));
        }
      }
      prompt += absl::StrJoin(parts, "\n\n");
    } else {
      prompt += content.empty() ? "" : content;
    }
  } else if (role == kRoleLatestReminder) {
    prompt += std::string(kLatestReminderSpToken) + content;
  } else if (role == kRoleTool) {
    LOG(FATAL) << "deepseek_v4 merges tool messages into user; please "
                  "preprocess with merge_tool_messages()";
  } else if (role == kRoleAssistant) {
    std::string thinking_part;
    std::string tc_content;

    std::vector<nlohmann::ordered_json> tcs;
    if (msg.contains("tool_calls") && msg["tool_calls"].is_array()) {
      tcs.reserve(msg["tool_calls"].size());
      for (const auto& tc : msg["tool_calls"]) {
        tcs.emplace_back(tc);
      }
    }
    if (!tcs.empty()) {
      tc_content = render_tool_calls(tcs);
    }

    std::string summary_content = content;
    std::string reasoning = msg.value("reasoning", "");

    bool prev_has_task = index - 1 >= 0 && messages[index - 1].contains("task");

    if (thinking_mode == kThinkingModeThinking && !prev_has_task) {
      if (!drop_thinking || index > last_user_idx) {
        thinking_part = reasoning + kThinkingEndToken;
      }
    }

    bool wo_eos = msg.contains("wo_eos") && msg["wo_eos"].is_boolean() &&
                  msg["wo_eos"].get<bool>();
    if (wo_eos) {
      prompt += thinking_part + summary_content + tc_content;
    } else {
      prompt += thinking_part + summary_content + tc_content + kEosToken;
    }
  } else {
    LOG(FATAL) << "Unknown role: " + role;
  }

  // Transition: append Assistant + thinking token
  if (index + 1 < static_cast<int32_t>(messages.size()) &&
      messages[index + 1].value("role", "") != kRoleAssistant &&
      messages[index + 1].value("role", "") != kRoleLatestReminder) {
    return prompt;
  }

  std::string task;
  if (msg.contains("task") && msg["task"].is_string()) {
    task = msg["task"].get<std::string>();
  }

  if (!task.empty()) {
    auto it = task_sp_tokens().find(task);
    if (it == task_sp_tokens().end()) {
      LOG(FATAL) << "Invalid task: " + task;
    }
    if (task != "action") {
      prompt += it->second;
    } else {
      prompt += kAssistantSpToken;
      prompt += (thinking_mode != kThinkingModeThinking) ? kThinkingEndToken
                                                         : kThinkingStartToken;
      prompt += it->second;
    }
  } else if (role == kRoleUser || role == kRoleDeveloper) {
    prompt += kAssistantSpToken;
    if (!drop_thinking && thinking_mode == kThinkingModeThinking) {
      prompt += kThinkingStartToken;
    } else if (drop_thinking && thinking_mode == kThinkingModeThinking &&
               index >= last_user_idx) {
      prompt += kThinkingStartToken;
    } else {
      prompt += kThinkingEndToken;
    }
  }

  return prompt;
}

}  // namespace

// ============================================================
// Public Interface
// ============================================================

DeepseekV4CppTemplate::DeepseekV4CppTemplate(const TokenizerArgs& args)
    : args_(args) {}

std::optional<std::string> DeepseekV4CppTemplate::apply(
    const ChatMessages& messages) const {
  const std::vector<xllm::JsonTool> empty_tools;
  const nlohmann::ordered_json kwargs = nlohmann::json::object();
  return apply(messages, empty_tools, kwargs);
}

std::optional<std::string> DeepseekV4CppTemplate::apply(
    const ChatMessages& messages,
    const std::vector<xllm::JsonTool>& json_tools,
    const nlohmann::ordered_json& chat_template_kwargs) const {
  try {
    nlohmann::ordered_json normalized =
        normalize_messages(messages, json_tools);
    std::string thinking_mode = get_thinking_mode(chat_template_kwargs);
    std::string reasoning_effort = get_reasoning_effort(chat_template_kwargs);

    // Preprocess: merge tool + sort
    normalized = merge_tool_messages(normalized);
    normalized = sort_tool_results_by_call_order(normalized);

    // drop_thinking = last msg is user
    bool drop_thinking = false;
    if (!normalized.empty()) {
      std::string last_role = normalized.back().value("role", "");
      drop_thinking = (last_role == kRoleUser);
    }

    // V4: disable drop_thinking when tools exist
    if (drop_thinking) {
      for (const auto& m : normalized) {
        if (m.contains("tools")) {
          drop_thinking = false;
          break;
        }
      }
    }

    if (thinking_mode == kThinkingModeThinking && drop_thinking) {
      normalized = drop_thinking_messages(normalized);
    }

    std::string prompt;
    prompt +=
        args_.bos_token().empty() ? std::string(kBosToken) : args_.bos_token();

    for (int32_t idx = 0; idx < static_cast<int32_t>(normalized.size());
         ++idx) {
      prompt += render_message(
          normalized, idx, thinking_mode, drop_thinking, reasoning_effort);
    }
    return prompt;
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to apply DeepSeek V4 native "
               << "template: " << e.what();
    return std::nullopt;
  }
}

}  // namespace xllm

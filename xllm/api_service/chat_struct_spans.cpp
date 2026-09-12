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

#include "api_service/chat_struct_spans.h"

#include <algorithm>

#include "api_service/json_scan.h"

namespace xllm {
namespace {

using json_scan::ScanAction;
using json_scan::ValueKind;

bool has_uppercase(std::string_view key) {
  return std::any_of(key.begin(), key.end(), [](char current) {
    return current >= 'A' && current <= 'Z';
  });
}

// Records the span of the object value at `cursor` and advances past it.
ScanAction record_object_span(std::string_view json,
                              size_t& cursor,
                              JsonSpan* span) {
  span->begin = cursor;
  if (!json_scan::skip_value(json, cursor)) {
    return ScanAction::BAIL;
  }
  span->end = cursor;
  return ScanAction::CONSUMED;
}

// Walks one tool object, recording the span of function.parameters when it is
// an object. Any other shape of "function" or "parameters" -- other than null,
// which both parsers leave unset -- is left to the reference parser.
bool scan_tool(std::string_view json,
               size_t& pos,
               size_t tool_index,
               std::vector<std::pair<size_t, JsonSpan>>* spans) {
  bool seen_function = false;
  return json_scan::scan_object(
      json, pos, [&](std::string_view key, ValueKind kind, size_t& cursor) {
        if (key != "function") {
          return ScanAction::SKIP;
        }
        if (seen_function) {
          return ScanAction::BAIL;
        }
        seen_function = true;
        if (kind == ValueKind::NULL_VALUE) {
          return ScanAction::SKIP;
        }
        if (kind != ValueKind::OBJECT) {
          return ScanAction::BAIL;
        }
        bool seen_parameters = false;
        const bool accounted = json_scan::scan_object(
            json,
            cursor,
            [&](std::string_view member, ValueKind member_kind, size_t& at) {
              if (member != "parameters") {
                return ScanAction::SKIP;
              }
              if (seen_parameters) {
                return ScanAction::BAIL;
              }
              seen_parameters = true;
              if (member_kind == ValueKind::NULL_VALUE) {
                return ScanAction::SKIP;
              }
              if (member_kind != ValueKind::OBJECT) {
                return ScanAction::BAIL;
              }
              JsonSpan span;
              const ScanAction action = record_object_span(json, at, &span);
              if (action == ScanAction::CONSUMED) {
                spans->emplace_back(tool_index, span);
              }
              return action;
            });
        return accounted ? ScanAction::CONSUMED : ScanAction::BAIL;
      });
}

bool scan_tools(std::string_view json,
                size_t& pos,
                std::vector<std::pair<size_t, JsonSpan>>* spans) {
  return json_scan::scan_array(
      json, pos, [&](size_t index, ValueKind kind, size_t& cursor) {
        // json2pb rejects a non-object tool outright; let the reference parser
        // word that rejection.
        if (kind != ValueKind::OBJECT) {
          return ScanAction::BAIL;
        }
        return scan_tool(json, cursor, index, spans) ? ScanAction::CONSUMED
                                                     : ScanAction::BAIL;
      });
}

}  // namespace

std::optional<ChatStructSpans> locate_chat_struct_members(
    std::string_view json) {
  ChatStructSpans spans;
  bool seen_kwargs = false;
  bool seen_tools = false;
  size_t pos = 0;
  json_scan::skip_whitespace(json, pos);
  const bool accounted = json_scan::scan_object(
      json, pos, [&](std::string_view key, ValueKind kind, size_t& cursor) {
        if (has_uppercase(key)) {
          return ScanAction::BAIL;
        }
        if (key == "chat_template_kwargs") {
          if (seen_kwargs) {
            return ScanAction::BAIL;
          }
          seen_kwargs = true;
          if (kind == ValueKind::NULL_VALUE) {
            return ScanAction::SKIP;
          }
          if (kind != ValueKind::OBJECT) {
            return ScanAction::BAIL;
          }
          JsonSpan span;
          const ScanAction action = record_object_span(json, cursor, &span);
          if (action == ScanAction::CONSUMED) {
            spans.chat_template_kwargs = span;
          }
          return action;
        }
        if (key == "tools") {
          if (seen_tools) {
            return ScanAction::BAIL;
          }
          seen_tools = true;
          if (kind == ValueKind::NULL_VALUE) {
            return ScanAction::SKIP;
          }
          if (kind != ValueKind::ARRAY) {
            return ScanAction::BAIL;
          }
          return scan_tools(json, cursor, &spans.tool_parameters)
                     ? ScanAction::CONSUMED
                     : ScanAction::BAIL;
        }
        return ScanAction::SKIP;
      });
  if (!accounted || !json_scan::only_whitespace_remains(json, pos)) {
    return std::nullopt;
  }
  return spans;
}

}  // namespace xllm

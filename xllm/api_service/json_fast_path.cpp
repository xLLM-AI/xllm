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

#include "api_service/json_fast_path.h"

#include <cstddef>

#include "api_service/json_scan.h"

namespace xllm {
namespace {

using json_scan::ScanAction;
using json_scan::ValueKind;

// Walks the elements of an LLM "messages" array, failing when the preprocessing
// pass would act on one: a non-object element, which it rejects, or
// array-valued content, which it collapses into a string.
bool skip_llm_messages(std::string_view json, size_t& pos) {
  return json_scan::scan_array(
      json, pos, [&json](size_t /*unused*/, ValueKind kind, size_t& cursor) {
        if (kind != ValueKind::OBJECT) {
          return ScanAction::BAIL;
        }
        const bool accounted = json_scan::scan_object(
            json,
            cursor,
            [](std::string_view key,
               ValueKind member_kind,
               size_t& /*unused*/) {
              if (key == "content" && member_kind == ValueKind::ARRAY) {
                return ScanAction::BAIL;
              }
              return ScanAction::SKIP;
            });
        return accounted ? ScanAction::CONSUMED : ScanAction::BAIL;
      });
}

// The scan has only proven the body needs no rewriting if it accounted for the
// whole body, trailing whitespace aside. Trailing content is rejected by the
// DOM parser, so hand those bodies over as well.
JsonFastPath verdict(std::string_view json, size_t pos, bool accounted) {
  if (!accounted || !json_scan::only_whitespace_remains(json, pos)) {
    return JsonFastPath::NEEDS_FULL_PARSE;
  }
  return JsonFastPath::SKIP_PREPROCESS;
}

}  // namespace

JsonFastPath completion_prompt_fast_path(std::string_view json) {
  size_t pos = 0;
  json_scan::skip_whitespace(json, pos);
  const bool accounted = json_scan::scan_object(
      json, pos, [](std::string_view key, ValueKind kind, size_t& /*unused*/) {
        if (key == "prompt" && kind == ValueKind::ARRAY) {
          return ScanAction::BAIL;
        }
        return ScanAction::SKIP;
      });
  return verdict(json, pos, accounted);
}

JsonFastPath llm_chat_fast_path(std::string_view json) {
  size_t pos = 0;
  json_scan::skip_whitespace(json, pos);
  const bool accounted = json_scan::scan_object(
      json, pos, [&json](std::string_view key, ValueKind kind, size_t& cursor) {
        if (key == "tool_choice") {
          // A string is left alone; an object is normalised, and any other kind
          // is rejected.
          return kind == ValueKind::STRING ? ScanAction::SKIP
                                           : ScanAction::BAIL;
        }
        if (key == "messages" && kind == ValueKind::ARRAY) {
          return skip_llm_messages(json, cursor) ? ScanAction::CONSUMED
                                                 : ScanAction::BAIL;
        }
        return ScanAction::SKIP;
      });
  return verdict(json, pos, accounted);
}

}  // namespace xllm

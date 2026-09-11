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

#include <array>
#include <cstddef>

namespace xllm {
namespace {

// Deepest container nesting the scan skips through before giving up. A body
// nested deeper than this goes to the DOM parser, which is the conservative
// answer rather than a wrong one.
constexpr size_t kMaxScanDepth = 64;

enum class ValueKind : int8_t {
  NULL_VALUE,
  BOOL,
  NUMBER,
  STRING,
  ARRAY,
  OBJECT,
};

enum class MemberAction : int8_t {
  // The member proves the preprocessing pass is needed.
  BAIL,
  // The walker should skip over the member's value.
  SKIP,
  // The handler already advanced the cursor past the member's value.
  CONSUMED,
};

void skip_whitespace(std::string_view json, size_t& pos) {
  while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' ||
                               json[pos] == '\n' || json[pos] == '\r')) {
    ++pos;
  }
}

// Skips the string literal whose opening quote sits at `pos`. The closing quote
// is found with string_view::find, so skipping a 64K prompt costs a memchr
// rather than a character-by-character walk.
bool skip_string(std::string_view json, size_t& pos) {
  if (pos >= json.size() || json[pos] != '"') {
    return false;
  }
  const size_t content_begin = pos + 1;
  size_t search = content_begin;
  while (true) {
    const size_t quote = json.find('"', search);
    if (quote == std::string_view::npos) {
      return false;
    }
    // An odd run of backslashes immediately before the quote escapes it.
    size_t backslashes = 0;
    while (quote - backslashes > content_begin &&
           json[quote - backslashes - 1] == '\\') {
      ++backslashes;
    }
    if (backslashes % 2 == 0) {
      pos = quote + 1;
      return true;
    }
    search = quote + 1;
  }
}

// Reads a member key. Fails on a malformed literal, and on one carrying an
// escape sequence: the scan does not decode escapes, and "\u0070rompt" must not
// be mistaken for a key other than "prompt".
bool scan_member_key(std::string_view json,
                     size_t& pos,
                     std::string_view& key) {
  if (pos >= json.size() || json[pos] != '"') {
    return false;
  }
  const size_t begin = pos + 1;
  size_t cursor = begin;
  while (cursor < json.size()) {
    if (json[cursor] == '\\') {
      return false;
    }
    if (json[cursor] == '"') {
      key = json.substr(begin, cursor - begin);
      pos = cursor + 1;
      return true;
    }
    ++cursor;
  }
  return false;
}

bool value_kind_at(std::string_view json, size_t pos, ValueKind& kind) {
  if (pos >= json.size()) {
    return false;
  }
  switch (json[pos]) {
    case '{':
      kind = ValueKind::OBJECT;
      return true;
    case '[':
      kind = ValueKind::ARRAY;
      return true;
    case '"':
      kind = ValueKind::STRING;
      return true;
    case 't':
    case 'f':
      kind = ValueKind::BOOL;
      return true;
    case 'n':
      kind = ValueKind::NULL_VALUE;
      return true;
    case '-':
      kind = ValueKind::NUMBER;
      return true;
    default:
      break;
  }
  if (json[pos] >= '0' && json[pos] <= '9') {
    kind = ValueKind::NUMBER;
    return true;
  }
  return false;
}

// Skips the container whose opening bracket sits at `pos`, tracking string
// literals and matching brackets so that a mismatched body is reported as
// unaccountable instead of being silently accepted.
bool skip_container(std::string_view json, size_t& pos) {
  std::array<char, kMaxScanDepth> closers{};
  size_t depth = 0;
  while (pos < json.size()) {
    const char current = json[pos];
    if (current == '"') {
      if (!skip_string(json, pos)) {
        return false;
      }
      continue;
    }
    if (current == '{' || current == '[') {
      if (depth == kMaxScanDepth) {
        return false;
      }
      closers[depth] = (current == '{') ? '}' : ']';
      ++depth;
    } else if (current == '}' || current == ']') {
      if (depth == 0 || closers[depth - 1] != current) {
        return false;
      }
      --depth;
      if (depth == 0) {
        ++pos;
        return true;
      }
    }
    ++pos;
  }
  return false;
}

bool skip_value(std::string_view json, size_t& pos) {
  ValueKind kind = ValueKind::NULL_VALUE;
  if (!value_kind_at(json, pos, kind)) {
    return false;
  }
  if (kind == ValueKind::STRING) {
    return skip_string(json, pos);
  }
  if (kind == ValueKind::OBJECT || kind == ValueKind::ARRAY) {
    return skip_container(json, pos);
  }
  // null / true / false / number: consume the token up to the next structural
  // character. The grammar of the token itself is left to the real parser.
  const size_t begin = pos;
  while (pos < json.size() && json[pos] != ',' && json[pos] != '}' &&
         json[pos] != ']' && json[pos] != ' ' && json[pos] != '\t' &&
         json[pos] != '\n' && json[pos] != '\r') {
    ++pos;
  }
  return pos > begin;
}

// Walks the members of the object whose '{' sits at `pos`. `on_member` receives
// the key, the kind of its value, and the cursor positioned on that value.
template <typename OnMember>
bool scan_object(std::string_view json, size_t& pos, OnMember on_member) {
  if (pos >= json.size() || json[pos] != '{') {
    return false;
  }
  ++pos;
  skip_whitespace(json, pos);
  if (pos < json.size() && json[pos] == '}') {
    ++pos;
    return true;
  }
  while (true) {
    skip_whitespace(json, pos);
    std::string_view key;
    if (!scan_member_key(json, pos, key)) {
      return false;
    }
    skip_whitespace(json, pos);
    if (pos >= json.size() || json[pos] != ':') {
      return false;
    }
    ++pos;
    skip_whitespace(json, pos);
    ValueKind kind = ValueKind::NULL_VALUE;
    if (!value_kind_at(json, pos, kind)) {
      return false;
    }
    const MemberAction action = on_member(key, kind, pos);
    if (action == MemberAction::BAIL) {
      return false;
    }
    if (action == MemberAction::SKIP && !skip_value(json, pos)) {
      return false;
    }
    skip_whitespace(json, pos);
    if (pos >= json.size()) {
      return false;
    }
    if (json[pos] == ',') {
      ++pos;
      continue;
    }
    if (json[pos] == '}') {
      ++pos;
      return true;
    }
    return false;
  }
}

// Walks the elements of an LLM "messages" array, failing when the preprocessing
// pass would act on one: a non-object element, which it rejects, or
// array-valued content, which it collapses into a string.
bool skip_llm_messages(std::string_view json, size_t& pos) {
  if (pos >= json.size() || json[pos] != '[') {
    return false;
  }
  ++pos;
  skip_whitespace(json, pos);
  if (pos < json.size() && json[pos] == ']') {
    ++pos;
    return true;
  }
  while (true) {
    skip_whitespace(json, pos);
    const bool accounted = scan_object(
        json,
        pos,
        [](std::string_view key, ValueKind kind, size_t& /*unused*/) {
          if (key == "content" && kind == ValueKind::ARRAY) {
            return MemberAction::BAIL;
          }
          return MemberAction::SKIP;
        });
    if (!accounted) {
      return false;
    }
    skip_whitespace(json, pos);
    if (pos >= json.size()) {
      return false;
    }
    if (json[pos] == ',') {
      ++pos;
      continue;
    }
    if (json[pos] == ']') {
      ++pos;
      return true;
    }
    return false;
  }
}

// The scan has only proven the body needs no rewriting if it accounted for the
// whole body, trailing whitespace aside. Trailing content is rejected by the
// DOM parser, so hand those bodies over as well.
JsonFastPath verdict(std::string_view json, size_t pos, bool accounted) {
  if (!accounted) {
    return JsonFastPath::NEEDS_FULL_PARSE;
  }
  skip_whitespace(json, pos);
  return pos == json.size() ? JsonFastPath::SKIP_PREPROCESS
                            : JsonFastPath::NEEDS_FULL_PARSE;
}

}  // namespace

JsonFastPath completion_prompt_fast_path(std::string_view json) {
  size_t pos = 0;
  skip_whitespace(json, pos);
  const bool accounted = scan_object(
      json, pos, [](std::string_view key, ValueKind kind, size_t& /*unused*/) {
        if (key == "prompt" && kind == ValueKind::ARRAY) {
          return MemberAction::BAIL;
        }
        return MemberAction::SKIP;
      });
  return verdict(json, pos, accounted);
}

JsonFastPath llm_chat_fast_path(std::string_view json) {
  size_t pos = 0;
  skip_whitespace(json, pos);
  const bool accounted = scan_object(
      json, pos, [&json](std::string_view key, ValueKind kind, size_t& cursor) {
        if (key == "tool_choice") {
          // A string is left alone; an object is normalised, and any other kind
          // is rejected.
          return kind == ValueKind::STRING ? MemberAction::SKIP
                                           : MemberAction::BAIL;
        }
        if (key == "messages" && kind == ValueKind::ARRAY) {
          return skip_llm_messages(json, cursor) ? MemberAction::CONSUMED
                                                 : MemberAction::BAIL;
        }
        return MemberAction::SKIP;
      });
  return verdict(json, pos, accounted);
}

}  // namespace xllm

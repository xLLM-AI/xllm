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

#include "api_service/json_scan.h"

#include <array>

namespace xllm::json_scan {
namespace {

// Deepest container nesting the scan skips through before giving up. A body
// nested deeper than this goes to the real parser, which is the conservative
// answer rather than a wrong one.
constexpr size_t kMaxScanDepth = 64;

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

}  // namespace

void skip_whitespace(std::string_view json, size_t& pos) {
  while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' ||
                               json[pos] == '\n' || json[pos] == '\r')) {
    ++pos;
  }
}

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

bool only_whitespace_remains(std::string_view json, size_t pos) {
  skip_whitespace(json, pos);
  return pos == json.size();
}

}  // namespace xllm::json_scan

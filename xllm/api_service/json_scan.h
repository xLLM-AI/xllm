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

#include <cstddef>
#include <cstdint>
#include <string_view>

// Structural scanning of JSON request bodies without building a DOM.
//
// These primitives walk the bytes of a body, tracking string literals and
// bracket matching, so that callers can ask structural questions -- what kind
// of value a member holds, where its bytes lie -- at memchr speed. They are not
// a validator: every function returns false as soon as it meets input it cannot
// account for, and callers are expected to fall back to a real parser then.
namespace xllm::json_scan {

enum class ValueKind : int8_t {
  NULL_VALUE,
  BOOL,
  NUMBER,
  STRING,
  ARRAY,
  OBJECT,
};

// What a member or element visitor asks the walker to do next.
enum class ScanAction : int8_t {
  // Stop: the input needs the real parser.
  BAIL,
  // Skip over the value at the cursor.
  SKIP,
  // The visitor already advanced the cursor past the value.
  CONSUMED,
};

void skip_whitespace(std::string_view json, size_t& pos);

// Skips the string literal whose opening quote sits at `pos`. The closing quote
// is found with string_view::find, so skipping a 64K prompt costs a memchr
// rather than a character-by-character walk.
bool skip_string(std::string_view json, size_t& pos);

// Reads a member key. Fails on a malformed literal, and on one carrying an
// escape sequence: the scan does not decode escapes, and "\u0070rompt" must not
// be mistaken for a key other than "prompt".
bool scan_member_key(std::string_view json, size_t& pos, std::string_view& key);

bool value_kind_at(std::string_view json, size_t pos, ValueKind& kind);

// Skips one complete value of any kind. Containers are skipped iteratively with
// bracket matching, so a mismatched body is reported rather than accepted.
bool skip_value(std::string_view json, size_t& pos);

// True when only whitespace remains after `pos`.
bool only_whitespace_remains(std::string_view json, size_t pos);

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
    const ScanAction action = on_member(key, kind, pos);
    if (action == ScanAction::BAIL) {
      return false;
    }
    if (action == ScanAction::SKIP && !skip_value(json, pos)) {
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

// Walks the elements of the array whose '[' sits at `pos`. `on_element`
// receives the element index, the kind of the element, and the cursor
// positioned on it.
template <typename OnElement>
bool scan_array(std::string_view json, size_t& pos, OnElement on_element) {
  if (pos >= json.size() || json[pos] != '[') {
    return false;
  }
  ++pos;
  skip_whitespace(json, pos);
  if (pos < json.size() && json[pos] == ']') {
    ++pos;
    return true;
  }
  size_t index = 0;
  while (true) {
    skip_whitespace(json, pos);
    ValueKind kind = ValueKind::NULL_VALUE;
    if (!value_kind_at(json, pos, kind)) {
      return false;
    }
    const ScanAction action = on_element(index, kind, pos);
    if (action == ScanAction::BAIL) {
      return false;
    }
    if (action == ScanAction::SKIP && !skip_value(json, pos)) {
      return false;
    }
    ++index;
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

}  // namespace xllm::json_scan

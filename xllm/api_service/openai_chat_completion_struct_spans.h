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
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace xllm {

// Byte range [begin, end) of a value inside a request body.
struct JsonSpan {
  size_t begin = 0;
  size_t end = 0;
};

// Where the google.protobuf.Struct members of an OpenAI chat body lie:
// chat_template_kwargs at the root, and tools[i].function.parameters.
struct ChatStructSpans {
  std::optional<JsonSpan> chat_template_kwargs;
  // Index into "tools" paired with the span of
  // tools[index].function.parameters.
  std::vector<std::pair<size_t, JsonSpan>> tool_parameters;
};

// Locates the Struct-typed members of a chat body with a structural scan.
//
// Returns nullopt for anything but the canonical shape, and the caller must
// then decode the body with protobuf's JSON parser: a malformed body, a
// non-object tool, a "function" or "parameters" or "chat_template_kwargs" of an
// unexpected kind (null is fine -- both parsers leave the field unset), a
// duplicate "tools" or "chat_template_kwargs" key (the two parsers disagree on
// which wins), or a root key with an uppercase letter (protobuf's parser also
// accepts lowerCamelCase field aliases; json2pb does not).
std::optional<ChatStructSpans> locate_chat_struct_members(
    std::string_view json);

}  // namespace xllm

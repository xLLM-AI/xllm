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

#include <cstdint>
#include <string_view>

namespace xllm {

// Whether a request body needs the DOM-based preprocessing pass at all.
//
// The passes in completion_json_parser / chat_json_parser exist to rewrite the
// handful of shapes the proto schema cannot express: an array prompt,
// array-valued message content, an object tool_choice. For every other body
// they parse the whole request into a nlohmann DOM, conclude there is nothing
// to do, and hand the original bytes back -- after which the proto parser
// parses the same bytes a second time. On a 64K prompt that discarded pass
// measures ~740us against ~86us for the json2pb parse that actually fills the
// message.
//
// The predicates below answer "could this body possibly need rewriting?" by
// scanning the bytes, allocating nothing and building no nodes. Long string
// values are skipped with a memchr-backed search rather than decoded, so the
// scan costs a pass over the body instead of a parse of it.
//
// They are deliberately conservative: NEEDS_FULL_PARSE is the answer for
// anything the scan cannot fully account for, which keeps the DOM pass
// authoritative for rewriting, validation and error reporting. SKIP_PREPROCESS
// is returned only when that pass provably would have returned the body
// unchanged.
//
// The scan is not a JSON validator. It validates the structure it walks -- the
// root object's members, and for chat the messages array and its elements'
// members -- and gives up on anything unexpected there. Inside values it skips
// over it only tracks string literals and bracket matching, so a body that is
// malformed deep inside a skipped value may reach the proto parser and be
// rejected there instead of here. Both paths reject the request; only the
// message differs.
enum class JsonFastPath : int8_t {
  // The preprocessing pass would return the body unchanged; skip it.
  SKIP_PREPROCESS = 0,
  // Run the preprocessing pass: it may rewrite the body, it may reject it, or
  // the scan could not account for the input.
  NEEDS_FULL_PARSE = 1,
};

// preprocess_completion_prompt only rewrites an array prompt, so the pass can
// be skipped when the root object binds "prompt" to anything else, or does not
// bind it at all.
JsonFastPath completion_prompt_fast_path(std::string_view json);

// LlmChatJsonParser::preprocess acts when a message binds "content" to an array
// (collapsed into a string), when "tool_choice" is an object (normalised), or
// when an element of "messages" is not an object (rejected). The pass can be
// skipped when none of those hold.
JsonFastPath llm_chat_fast_path(std::string_view json);

}  // namespace xllm

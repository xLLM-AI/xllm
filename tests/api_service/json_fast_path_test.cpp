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

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace xllm {
namespace {

// Replicates the decision the DOM-based pass in completion_json_parser makes,
// so the scan is checked against the semantics it approximates rather than
// against itself. Returns false whenever that pass would rewrite or reject the
// body.
bool dom_leaves_completion_unchanged(const std::string& body) {
  const nlohmann::json json =
      nlohmann::json::parse(body, nullptr, /*allow_exceptions=*/false);
  if (json.is_discarded()) {
    return false;
  }
  return !json.contains("prompt") || !json["prompt"].is_array();
}

// Same, for LlmChatJsonParser::preprocess.
bool dom_leaves_llm_chat_unchanged(const std::string& body) {
  const nlohmann::json json =
      nlohmann::json::parse(body, nullptr, /*allow_exceptions=*/false);
  if (json.is_discarded()) {
    return false;
  }
  // An object tool_choice is normalised; any other non-string kind is rejected.
  if (json.contains("tool_choice") && !json["tool_choice"].is_string()) {
    return false;
  }
  if (!json.contains("messages") || !json["messages"].is_array()) {
    return true;
  }
  for (const auto& message : json["messages"]) {
    // A non-object message is rejected.
    if (!message.is_object()) {
      return false;
    }
    // Array content is collapsed into a string.
    if (message.contains("content") && message["content"].is_array()) {
      return false;
    }
  }
  return true;
}

std::string deeply_nested_prompt_body() {
  // Nested past kMaxScanDepth, which the scan refuses to account for.
  std::string body = R"({"prompt":"x","deep":)";
  body.append(80, '[');
  body.append(80, ']');
  body += "}";
  return body;
}

// Bodies the scan must clear, so that the optimisation is not silently dead.
std::vector<std::string> completion_skippable_bodies() {
  return {
      R"({"prompt":"hello","max_tokens":8})",
      R"({"max_tokens":8})",
      R"({"prompt":785})",
      R"({"prompt":null})",
      R"({"prompt":true})",
      R"({"prompt":-1.5e3})",
      R"({})",
      // Escaped quotes inside the prompt.
      R"({"prompt":"he said \"hi\""})",
      // A trailing escaped backslash: the final quote is not escaped.
      R"({"prompt":"ends with a backslash \\"})",
      // A prompt whose text looks like an array-valued prompt member.
      R"({"prompt":"looks like \"prompt\":[1,2] but is text"})",
      // Whitespace everywhere, plus an array in an unrelated member.
      "{ \"prompt\" : \"x\" , \"stop\" : [\"a\",\"b\"] }",
      // An array prompt nested one level down must not be mistaken for the
      // root member.
      R"({"nested":{"prompt":[1,2]},"prompt":"x"})",
      "{\"prompt\":\"x\"}\n  ",
  };
}

// Bodies the scan must hand over to the DOM pass.
std::vector<std::string> completion_full_parse_bodies() {
  return {
      R"({"prompt":[785,785]})",
      R"({"prompt":[]})",
      R"({"prompt" : [1]})",
      R"({"prompt":[785],"token_ids":[1]})",
      // The last duplicate wins, so this is an array prompt.
      R"({"prompt":"x","prompt":[1]})",
      // An escaped key is not decoded, so the scan cannot rule it out.
      R"({"pro\u006dpt":[1]})",
      "{not valid json",
      "",
      "[1,2]",
      R"({"prompt":"x"} trailing)",
      R"({"prompt":"unterminated)",
      // Mismatched brackets inside a skipped value.
      R"({"a":[1,2},"prompt":"x"})",
      deeply_nested_prompt_body(),
  };
}

std::vector<std::string> llm_chat_skippable_bodies() {
  return {
      R"({"messages":[{"role":"user","content":"Hello"}]})",
      R"({"model":"test"})",
      R"({"messages":[]})",
      R"({"messages":"not an array"})",
      R"({"messages":[{"role":"user","content":"has \"quotes\" and [brackets]"}]})",
      R"({"tool_choice":"auto","messages":[{"role":"user","content":"hi"}]})",
      R"({"messages":[{"role":"user","content":"hi"},)"
      R"({"role":"assistant","content":"yo"}]})",
      // An array-valued content nested inside another member of the message
      // is not the member the pass acts on.
      R"({"messages":[{"role":"user","content":"hi",)"
      R"("extra":{"content":["nested"]}}]})",
      R"({"messages":[{"role":"user","content":"hi"}],)"
      R"("tools":[{"type":"function","function":{"name":"f"}}]})",
      R"({"messages":[{"role":"user"}]})",
  };
}

std::vector<std::string> llm_chat_full_parse_bodies() {
  return {
      R"({"messages":[{"role":"user","content":[{"type":"text","text":"Hi"}]}]})",
      R"({"messages":["not an object"]})",
      R"({"messages":[{"content":[]}]})",
      R"({"tool_choice":{"type":"function","function":{"name":"submit"}},)"
      R"("messages":[{"role":"user","content":"hi"}]})",
      R"({"tool_choice":123,"messages":[{"role":"user","content":"hi"}]})",
      R"({"tool_choice":null})",
      "not valid json",
      "",
      // Trailing comma in the messages array.
      R"({"messages":[{"role":"user","content":"hi"},]})",
      // Second message carries array content.
      R"({"messages":[{"content":"hi"},{"content":[{"type":"text"}]}]})",
  };
}

}  // namespace

TEST(JsonFastPathTest, CompletionSkipsBodiesWithoutAnArrayPrompt) {
  for (const std::string& body : completion_skippable_bodies()) {
    EXPECT_EQ(completion_prompt_fast_path(body), JsonFastPath::SKIP_PREPROCESS)
        << "body: " << body;
  }
}

TEST(JsonFastPathTest, CompletionDefersBodiesItCannotRuleOut) {
  for (const std::string& body : completion_full_parse_bodies()) {
    EXPECT_EQ(completion_prompt_fast_path(body), JsonFastPath::NEEDS_FULL_PARSE)
        << "body: " << body;
  }
}

TEST(JsonFastPathTest, LlmChatSkipsBodiesNeedingNoRewrite) {
  for (const std::string& body : llm_chat_skippable_bodies()) {
    EXPECT_EQ(llm_chat_fast_path(body), JsonFastPath::SKIP_PREPROCESS)
        << "body: " << body;
  }
}

TEST(JsonFastPathTest, LlmChatDefersBodiesItCannotRuleOut) {
  for (const std::string& body : llm_chat_full_parse_bodies()) {
    EXPECT_EQ(llm_chat_fast_path(body), JsonFastPath::NEEDS_FULL_PARSE)
        << "body: " << body;
  }
}

// The contract that makes the fast path safe: whenever it fires, the DOM pass
// it replaces would have returned the body unchanged. The converse is not
// required -- the scan is allowed to defer a body the DOM pass would have left
// alone.
TEST(JsonFastPathTest, CompletionSkipImpliesTheDomPassWouldNotRewrite) {
  std::vector<std::string> bodies = completion_skippable_bodies();
  for (const std::string& body : completion_full_parse_bodies()) {
    bodies.emplace_back(body);
  }
  for (const std::string& body : bodies) {
    if (completion_prompt_fast_path(body) != JsonFastPath::SKIP_PREPROCESS) {
      continue;
    }
    EXPECT_TRUE(dom_leaves_completion_unchanged(body))
        << "fast path fired on a body the DOM pass would have touched: "
        << body;
  }
}

TEST(JsonFastPathTest, LlmChatSkipImpliesTheDomPassWouldNotRewrite) {
  std::vector<std::string> bodies = llm_chat_skippable_bodies();
  for (const std::string& body : llm_chat_full_parse_bodies()) {
    bodies.emplace_back(body);
  }
  for (const std::string& body : bodies) {
    if (llm_chat_fast_path(body) != JsonFastPath::SKIP_PREPROCESS) {
      continue;
    }
    EXPECT_TRUE(dom_leaves_llm_chat_unchanged(body))
        << "fast path fired on a body the DOM pass would have touched: "
        << body;
  }
}

// A long prompt is the case the fast path exists for: it must be cleared
// without the scan tripping over the payload.
TEST(JsonFastPathTest, CompletionSkipsLongPrompts) {
  std::string body = R"({"model":"m","prompt":")";
  body.append(64 * 1024, 'x');
  body += R"(","max_tokens":128})";
  EXPECT_EQ(completion_prompt_fast_path(body), JsonFastPath::SKIP_PREPROCESS);
  EXPECT_TRUE(dom_leaves_completion_unchanged(body));
}

}  // namespace xllm

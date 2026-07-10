/* Copyright 2025-2026 The xLLM Authors.

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

#include "api_service/chat_json_parser.h"

#include <google/protobuf/util/json_util.h>
#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "anthropic.pb.h"
#include "chat.pb.h"

namespace xllm {

class PreprocessChatJsonTest : public ::testing::Test {
 protected:
  void expect_success(const std::string& input,
                      const ChatJsonParser& parser,
                      const std::string& expected_output) {
    auto [status, result] = parser.preprocess(input);
    ASSERT_TRUE(status.ok()) << "Unexpected error: " << status.message();
    auto result_json = nlohmann::json::parse(result);
    auto expected_json = nlohmann::json::parse(expected_output);
    EXPECT_EQ(result_json, expected_json);
  }

  void expect_error(const std::string& input,
                    const ChatJsonParser& parser,
                    const std::string& expected_error_substring) {
    auto [status, result] = parser.preprocess(input);
    ASSERT_FALSE(status.ok()) << "Expected error but got success";
    EXPECT_NE(status.message().find(expected_error_substring),
              std::string::npos)
        << "Error message '" << status.message()
        << "' does not contain expected substring '" << expected_error_substring
        << "'";
  }
};

// =============================================================================
// Basic functionality tests
// =============================================================================

TEST_F(PreprocessChatJsonTest, PassThroughNonArrayContent) {
  std::string input = R"({
    "messages": [{"role": "user", "content": "Hello"}]
  })";
  std::string expected_vlm_output = R"({
    "messages": [{
      "role": "user",
      "content": [{"type": "text", "text": "Hello"}]
    }]
  })";
  LlmChatJsonParser llm_parser;
  VlmChatJsonParser vlm_parser;
  expect_success(input, llm_parser, input);
  expect_success(input, vlm_parser, expected_vlm_output);
}

TEST_F(PreprocessChatJsonTest, PassThroughNoMessages) {
  // JSON without messages field should pass through
  std::string input = R"({"model": "test"})";
  LlmChatJsonParser llm_parser;
  expect_success(input, llm_parser, input);
}

TEST_F(PreprocessChatJsonTest, CombineTextArrayIntoString) {
  // Array of text items should be combined into single string for
  // non-multimodal
  std::string input = R"({
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Hello"},
        {"type": "text", "text": "World"}
      ]
    }]
  })";
  std::string expected = R"({
    "messages": [{"role": "user", "content": "Hello\nWorld"}]
  })";
  LlmChatJsonParser llm_parser;
  VlmChatJsonParser vlm_parser;
  expect_success(input, llm_parser, expected);
  // For multimodal, array is preserved (not combined)
  expect_success(input, vlm_parser, input);
}

TEST_F(PreprocessChatJsonTest, SingleTextItemCombined) {
  // Single text item in array should be converted to string for non-multimodal
  std::string input = R"({
    "messages": [{
      "role": "user",
      "content": [{"type": "text", "text": "Hello"}]
    }]
  })";
  std::string expected = R"({
    "messages": [{"role": "user", "content": "Hello"}]
  })";
  LlmChatJsonParser llm_parser;
  VlmChatJsonParser vlm_parser;
  expect_success(input, llm_parser, expected);
  // For multimodal, array is preserved
  expect_success(input, vlm_parser, input);
}

// =============================================================================
// Multimodal content tests (Issue #801)
// =============================================================================

TEST_F(PreprocessChatJsonTest, ImageUrlPassesThroughOnMultimodal) {
  // image_url content should pass through unchanged on multimodal endpoint
  std::string input = R"({
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}},
        {"type": "text", "text": "What is this?"}
      ]
    }]
  })";
  VlmChatJsonParser vlm_parser;
  // Should pass through unchanged for multimodal
  expect_success(input, vlm_parser, input);
}

TEST_F(PreprocessChatJsonTest, ImageUrlErrorsOnTextOnly) {
  // image_url content should error on text-only endpoint with helpful message
  std::string input = R"({
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}},
        {"type": "text", "text": "What is this?"}
      ]
    }]
  })";
  LlmChatJsonParser llm_parser;
  expect_error(input, llm_parser, "multimodal backend");
  expect_error(input, llm_parser, "-backend vlm");
}

TEST_F(PreprocessChatJsonTest, MultipleMessagesWithMixedContent) {
  // Multiple messages: some text-only, some with images
  // On multimodal, all arrays are preserved (no combining)
  std::string input = R"({
    "messages": [
      {
        "role": "system",
        "content": [{"type": "text", "text": "You are helpful."}]
      },
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "data:image/png;base64,xyz"}},
          {"type": "text", "text": "Describe this image"}
        ]
      }
    ]
  })";
  VlmChatJsonParser vlm_parser;
  // On multimodal: all arrays preserved unchanged
  expect_success(input, vlm_parser, input);
}

// =============================================================================
// Error handling tests
// =============================================================================

TEST_F(PreprocessChatJsonTest, InvalidJsonReturnsError) {
  std::string input = "not valid json";
  LlmChatJsonParser llm_parser;
  expect_error(input, llm_parser, "Invalid JSON");
}

TEST_F(PreprocessChatJsonTest, NonObjectMessageReturnsError) {
  std::string input = R"({"messages": ["not an object"]})";
  LlmChatJsonParser llm_parser;
  expect_error(input, llm_parser, "must be an object");
}

TEST_F(PreprocessChatJsonTest, NonObjectContentItemReturnsError) {
  std::string input = R"({
    "messages": [{"role": "user", "content": ["not an object"]}]
  })";
  LlmChatJsonParser llm_parser;
  expect_error(input, llm_parser, "must be an object");
}

TEST_F(PreprocessChatJsonTest, MissingTextFieldReturnsError) {
  std::string input = R"({
    "messages": [{"role": "user", "content": [{"type": "text"}]}]
  })";
  LlmChatJsonParser llm_parser;
  expect_error(input, llm_parser, "Missing or invalid 'text' field");
}

TEST_F(PreprocessChatJsonTest, NonStringTextFieldReturnsError) {
  std::string input = R"({
    "messages": [{"role": "user", "content": [{"type": "text", "text": 123}]}]
  })";
  LlmChatJsonParser llm_parser;
  expect_error(input, llm_parser, "Missing or invalid 'text' field");
}

TEST_F(PreprocessChatJsonTest, MalformedTextInMultimodalContent) {
  // Multimodal mode skips parsing entirely - validation happens downstream
  std::string input = R"({
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "..."}},
        {"type": "text"}
      ]
    }]
  })";
  VlmChatJsonParser vlm_parser;
  // Should pass through unchanged without validation
  expect_success(input, vlm_parser, input);
}

// =============================================================================
// Edge cases
// =============================================================================

TEST_F(PreprocessChatJsonTest, EmptyContentArray) {
  // Empty content array - should result in empty string for non-multimodal
  std::string input = R"({
    "messages": [{"role": "user", "content": []}]
  })";
  std::string expected = R"({
    "messages": [{"role": "user", "content": ""}]
  })";
  LlmChatJsonParser llm_parser;
  VlmChatJsonParser vlm_parser;
  expect_success(input, llm_parser, expected);
  // For multimodal, empty array is preserved
  expect_success(input, vlm_parser, input);
}

TEST_F(PreprocessChatJsonTest, PreservesOtherFields) {
  // Other fields in the request should be preserved
  std::string input = R"({
    "model": "test-model",
    "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
    "temperature": 0.7,
    "max_tokens": 100
  })";
  std::string expected = R"({
    "model": "test-model",
    "messages": [{"role": "user", "content": "Hi"}],
    "temperature": 0.7,
    "max_tokens": 100
  })";
  LlmChatJsonParser llm_parser;
  VlmChatJsonParser vlm_parser;
  expect_success(input, llm_parser, expected);
  // For multimodal, array is preserved
  expect_success(input, vlm_parser, input);
}

TEST_F(PreprocessChatJsonTest, OpenAIObjectToolChoiceRemapped) {
  std::string input = R"({
    "model": "test-model",
    "messages": [{"role": "user", "content": "Submit the result."}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "submit",
        "description": "Submit the final answer.",
        "parameters": {"type": "object", "properties": {}}
      }
    }],
    "tool_choice": {
      "type": "function",
      "function": {"name": "submit"}
    }
  })";

  LlmChatJsonParser parser;
  auto [status, processed_json] = parser.preprocess(input);
  ASSERT_TRUE(status.ok()) << "Unexpected error: " << status.message();

  proto::ChatRequest request;
  google::protobuf::util::JsonParseOptions options;
  options.ignore_unknown_fields = true;
  auto parse_status = google::protobuf::util::JsonStringToMessage(
      processed_json, &request, options);
  ASSERT_TRUE(parse_status.ok()) << parse_status.ToString();
  ASSERT_TRUE(request.has_tool_choice());

  nlohmann::json expected_tool_choice = {{"type", "function"},
                                         {"function", {{"name", "submit"}}}};
  EXPECT_EQ(nlohmann::json::parse(request.tool_choice()), expected_tool_choice);
}

TEST_F(PreprocessChatJsonTest, InvalidObjectToolChoiceReturnsError) {
  std::string input = R"({
    "messages": [{"role": "user", "content": "Hello"}],
    "tool_choice": {
      "type": "function",
      "function": {}
    }
  })";

  LlmChatJsonParser parser;
  expect_error(input, parser, "function.name");
}

TEST_F(PreprocessChatJsonTest, UnknownContentTypeOnMultimodal) {
  // Unknown content types should pass through on multimodal
  std::string input = R"({
    "messages": [{
      "role": "user",
      "content": [{"type": "video", "video": {"url": "..."}}]
    }]
  })";
  VlmChatJsonParser vlm_parser;
  expect_success(input, vlm_parser, input);
}

TEST_F(PreprocessChatJsonTest, UnknownContentTypeErrorsOnTextOnly) {
  // Unknown content types should error on text-only with helpful message
  std::string input = R"({
    "messages": [{
      "role": "user",
      "content": [{"type": "video", "video": {"url": "..."}}]
    }]
  })";
  LlmChatJsonParser llm_parser;
  expect_error(input, llm_parser, "multimodal backend");
}

// =============================================================================
// Anthropic parser tests
// =============================================================================

TEST_F(PreprocessChatJsonTest, AnthropicStringContentRemapped) {
  std::string input = R"({
    "messages": [{"role": "user", "content": "Hello"}]
  })";
  std::string expected = R"({
    "messages": [{"role": "user", "content_string": "Hello"}]
  })";
  AnthropicChatJsonParser parser;
  expect_success(input, parser, expected);
}

TEST_F(PreprocessChatJsonTest, AnthropicArrayContentRemapped) {
  std::string input = R"({
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Hello"},
        {"type": "image", "source": {"data": "abc"}}
      ]
    }]
  })";
  std::string expected = R"({
    "messages": [{
      "role": "user",
      "content_blocks": {
        "blocks": [
          {"type": "text", "text": "Hello"},
          {"type": "image", "source": {"data": "abc"}}
        ]
      }
    }]
  })";
  AnthropicChatJsonParser parser;
  expect_success(input, parser, expected);
}

TEST_F(PreprocessChatJsonTest, AnthropicToolResultRemapped) {
  std::string input = R"({
    "messages": [{
      "role": "user",
      "content": [{
        "type": "tool_result",
        "tool_use_id": "call_123",
        "content": "total 1\nfile.txt"
      }]
    }]
  })";

  AnthropicChatJsonParser parser;
  auto [status, processed_json] = parser.preprocess(input);
  ASSERT_TRUE(status.ok()) << "Unexpected error: " << status.message();

  proto::AnthropicMessagesRequest request;
  google::protobuf::util::JsonParseOptions options;
  options.ignore_unknown_fields = true;
  auto parse_status = google::protobuf::util::JsonStringToMessage(
      processed_json, &request, options);
  ASSERT_TRUE(parse_status.ok()) << parse_status.ToString();

  ASSERT_EQ(request.messages_size(), 1);
  const auto& blocks = request.messages(0).content_blocks().blocks();
  ASSERT_EQ(blocks.size(), 1);
  EXPECT_EQ(blocks[0].type(), "tool_result");
  ASSERT_TRUE(blocks[0].has_id());
  EXPECT_EQ(blocks[0].id(), "call_123");
  ASSERT_TRUE(blocks[0].has_content_string());
  EXPECT_EQ(blocks[0].content_string(), "total 1\nfile.txt");
}

TEST_F(PreprocessChatJsonTest, AnthropicToolResultListRemapped) {
  std::string input = R"({
    "messages": [{
      "role": "user",
      "content": [{
        "type": "tool_result",
        "tool_use_id": "call_123",
        "content": [
          {"type": "text", "text": "total 1"},
          {"type": "text", "text": "file.txt"}
        ]
      }]
    }]
  })";

  AnthropicChatJsonParser parser;
  auto [status, processed_json] = parser.preprocess(input);
  ASSERT_TRUE(status.ok()) << "Unexpected error: " << status.message();

  nlohmann::json processed = nlohmann::json::parse(processed_json);
  const nlohmann::json& block =
      processed["messages"][0]["content_blocks"]["blocks"][0];
  const nlohmann::json& items = block["content_list"]["items"];
  ASSERT_TRUE(items.is_array());
  ASSERT_EQ(items.size(), 2);
  EXPECT_EQ(items[0]["type"], "text");
  EXPECT_EQ(items[0]["text"], "total 1");
  EXPECT_EQ(items[1]["type"], "text");
  EXPECT_EQ(items[1]["text"], "file.txt");

  proto::AnthropicMessagesRequest request;
  google::protobuf::util::JsonParseOptions options;
  options.ignore_unknown_fields = true;
  auto parse_status = google::protobuf::util::JsonStringToMessage(
      processed_json, &request, options);
  ASSERT_TRUE(parse_status.ok()) << parse_status.ToString();

  ASSERT_EQ(request.messages_size(), 1);
  const auto& blocks = request.messages(0).content_blocks().blocks();
  ASSERT_EQ(blocks.size(), 1);
  EXPECT_EQ(blocks[0].type(), "tool_result");
  ASSERT_TRUE(blocks[0].has_id());
  EXPECT_EQ(blocks[0].id(), "call_123");
  ASSERT_TRUE(blocks[0].has_content_list());
  ASSERT_EQ(blocks[0].content_list().items_size(), 2);
}

TEST_F(PreprocessChatJsonTest, AnthropicSystemStringRemapped) {
  std::string input = R"({
    "system": "You are helpful.",
    "messages": [{"role": "user", "content": "Hi"}]
  })";
  std::string expected = R"({
    "system_string": "You are helpful.",
    "messages": [{"role": "user", "content_string": "Hi"}]
  })";
  AnthropicChatJsonParser parser;
  expect_success(input, parser, expected);
}

TEST_F(PreprocessChatJsonTest, AnthropicSystemArrayRemapped) {
  std::string input = R"({
    "system": [{"type": "text", "text": "You are helpful."}],
    "messages": [{"role": "user", "content": "Hi"}]
  })";
  std::string expected = R"({
    "system_blocks": {"blocks": [{"type": "text", "text": "You are helpful."}]},
    "messages": [{"role": "user", "content_string": "Hi"}]
  })";
  AnthropicChatJsonParser parser;
  expect_success(input, parser, expected);
}

TEST_F(PreprocessChatJsonTest, AnthropicNoContentNoSystem) {
  std::string input = R"({"model": "claude-3"})";
  AnthropicChatJsonParser parser;
  expect_success(input, parser, input);
}

TEST_F(PreprocessChatJsonTest, AnthropicInvalidJsonReturnsError) {
  std::string input = "not valid json";
  AnthropicChatJsonParser parser;
  expect_error(input, parser, "Invalid JSON");
}

TEST_F(PreprocessChatJsonTest, AnthropicPreservesOtherFields) {
  std::string input = R"({
    "model": "claude-3",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello"}]
  })";
  std::string expected = R"({
    "model": "claude-3",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content_string": "Hello"}]
  })";
  AnthropicChatJsonParser parser;
  expect_success(input, parser, expected);
}

TEST_F(PreprocessChatJsonTest, AnthropicIgnoreEosParsedAfterRemap) {
  std::string input = R"({
    "model": "claude-3",
    "max_tokens": 1024,
    "ignore_eos": true,
    "messages": [{"role": "user", "content": "Hello"}]
  })";

  AnthropicChatJsonParser parser;
  auto [status, processed_json] = parser.preprocess(input);
  ASSERT_TRUE(status.ok()) << "Unexpected error: " << status.message();

  proto::AnthropicMessagesRequest request;
  google::protobuf::util::JsonParseOptions options;
  options.ignore_unknown_fields = true;
  auto parse_status = google::protobuf::util::JsonStringToMessage(
      processed_json, &request, options);
  ASSERT_TRUE(parse_status.ok()) << parse_status.ToString();

  EXPECT_TRUE(request.has_ignore_eos());
  EXPECT_TRUE(request.ignore_eos());
  ASSERT_EQ(request.messages_size(), 1);
  EXPECT_EQ(request.messages(0).content_string(), "Hello");
}

}  // namespace xllm

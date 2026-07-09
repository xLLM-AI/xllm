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

#include "api_service/completion_json_parser.h"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

namespace xllm {

namespace {

void expect_error(const std::string& input,
                  const std::string& expected_message) {
  auto [status, result] = preprocess_completion_prompt(input);
  ASSERT_FALSE(status.ok()) << "Expected error but got success";
  EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(status.message(), expected_message);
}

}  // namespace

// String prompt passes through with bytes untouched.
TEST(PreprocessCompletionPromptTest, StringPromptPassThroughUnchanged) {
  std::string input = R"({"prompt": "hello", "max_tokens": 8})";
  auto [status, result] = preprocess_completion_prompt(input);
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(result, input);
}

// Missing prompt passes through unchanged.
TEST(PreprocessCompletionPromptTest, MissingPromptPassThroughUnchanged) {
  std::string input = R"({"max_tokens": 8})";
  auto [status, result] = preprocess_completion_prompt(input);
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(result, input);
}

// Scalar integer prompt is left to json2pb, passed through unchanged.
TEST(PreprocessCompletionPromptTest, ScalarPromptPassThroughUnchanged) {
  std::string input = R"({"prompt": 785})";
  auto [status, result] = preprocess_completion_prompt(input);
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(result, input);
}

// Integer array prompt is moved into token_ids and prompt is cleared.
TEST(PreprocessCompletionPromptTest, IntegerArrayMovedToTokenIds) {
  std::string input = R"({"prompt": [785, 785], "max_tokens": 8})";
  auto [status, result] = preprocess_completion_prompt(input);
  ASSERT_TRUE(status.ok()) << status.message();
  auto json = nlohmann::json::parse(result);
  EXPECT_EQ(json["prompt"], "");
  EXPECT_EQ(json["token_ids"], (std::vector<int32_t>{785, 785}));
  EXPECT_EQ(json["max_tokens"], 8);
}

// Boundary int32 values are accepted.
TEST(PreprocessCompletionPromptTest, Int32BoundaryValuesAccepted) {
  std::string input = R"({"prompt": [2147483647, 0]})";
  auto [status, result] = preprocess_completion_prompt(input);
  ASSERT_TRUE(status.ok()) << status.message();
  auto json = nlohmann::json::parse(result);
  EXPECT_EQ(json["token_ids"], (std::vector<int32_t>{2147483647, 0}));
}

TEST(PreprocessCompletionPromptTest, EmptyArrayRejected) {
  expect_error(R"({"prompt": []})", "prompt array is empty");
}

TEST(PreprocessCompletionPromptTest, ArrayOfStringsRejected) {
  expect_error(R"({"prompt": ["a", "b"]})",
               "batch prompts (array of strings) are not supported");
}

TEST(PreprocessCompletionPromptTest, NestedArraysRejected) {
  expect_error(R"({"prompt": [[1, 2], [3, 4]]})",
               "batch prompts (nested arrays) are not supported");
}

TEST(PreprocessCompletionPromptTest, FloatElementRejected) {
  expect_error(R"({"prompt": [785.5]})",
               "prompt array must contain only integer token ids");
}

TEST(PreprocessCompletionPromptTest, ObjectElementRejected) {
  expect_error(R"({"prompt": [{"a": 1}]})",
               "prompt array must contain only integer token ids");
}

TEST(PreprocessCompletionPromptTest, MixedElementRejected) {
  expect_error(R"({"prompt": [1, "a"]})",
               "prompt array must contain only integer token ids");
}

TEST(PreprocessCompletionPromptTest, TokenIdExceedsInt32PositiveRejected) {
  expect_error(R"({"prompt": [2147483648]})", "token id exceeds int32 range");
}

TEST(PreprocessCompletionPromptTest, TokenIdExceedsInt32NegativeRejected) {
  expect_error(R"({"prompt": [-2147483649]})", "token id exceeds int32 range");
}

TEST(PreprocessCompletionPromptTest, PromptArrayAndTokenIdsRejected) {
  expect_error(R"({"prompt": [785], "token_ids": [1]})",
               "specify prompt as a token array or token_ids, not both");
}

TEST(PreprocessCompletionPromptTest, InvalidJsonRejected) {
  auto [status, result] = preprocess_completion_prompt("{not valid json");
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
}

}  // namespace xllm

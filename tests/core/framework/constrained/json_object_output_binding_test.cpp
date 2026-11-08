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

#include "core/framework/constrained/json_object_output_binding.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "core/framework/constrained/json_object_grammar.h"

namespace xllm {
namespace {

JsonObjectGrammar make_grammar() {
  return JsonObjectGrammar({"{", "}", "stop"},
                           /*stop_token_ids=*/{2});
}

TEST(JsonObjectOutputBindingTest, ResolvesReorderedAndInsertedRows) {
  JsonObjectGrammar grammar = make_grammar();
  std::vector<JsonObjectGrammarState> states = {grammar.initial_state(),
                                                grammar.initial_state()};
  std::vector<int32_t> output_rows;
  std::vector<JsonObjectOutputError> output_errors;
  std::string error;

  ASSERT_TRUE(detail::resolve_json_object_output_rows(states,
                                                      {"req-b", "req-new"},
                                                      {1, -1},
                                                      {"req-a", "req-b"},
                                                      &output_rows,
                                                      &output_errors,
                                                      &error))
      << error;
  EXPECT_EQ(output_rows, std::vector<int32_t>({1, -1}));
  EXPECT_TRUE(output_errors.empty());
}

TEST(JsonObjectOutputBindingTest, ReportsDuplicateConstrainedSequenceId) {
  JsonObjectGrammar grammar = make_grammar();
  std::vector<JsonObjectGrammarState> states = {grammar.initial_state()};
  std::vector<int32_t> output_rows;
  std::vector<JsonObjectOutputError> output_errors;
  std::string error;

  ASSERT_TRUE(detail::resolve_json_object_output_rows(states,
                                                      {"req-a"},
                                                      {0},
                                                      {"req-a", "req-a"},
                                                      &output_rows,
                                                      &output_errors,
                                                      &error))
      << error;
  EXPECT_EQ(output_rows, std::vector<int32_t>({-2}));
  ASSERT_EQ(output_errors.size(), 1u);
  EXPECT_NE(output_errors[0].message.find("duplicate"), std::string::npos);
}

TEST(JsonObjectOutputBindingTest, IgnoresDuplicateUnconstrainedRows) {
  std::vector<JsonObjectGrammarState> states = {JsonObjectGrammarState()};
  std::vector<int32_t> output_rows;
  std::vector<JsonObjectOutputError> output_errors;
  std::string error;

  ASSERT_TRUE(
      detail::resolve_json_object_output_rows(states,
                                              {"req-plain"},
                                              {-1},
                                              {"req-plain", "req-plain"},
                                              &output_rows,
                                              &output_errors,
                                              &error))
      << error;
  EXPECT_EQ(output_rows, std::vector<int32_t>({-1}));
  EXPECT_TRUE(output_errors.empty());
}

TEST(JsonObjectOutputBindingTest, RejectsEmptyConstrainedSequenceId) {
  JsonObjectGrammar grammar = make_grammar();
  std::vector<JsonObjectGrammarState> states = {grammar.initial_state()};
  std::vector<int32_t> output_rows;
  std::vector<JsonObjectOutputError> output_errors;
  std::string error;

  EXPECT_FALSE(detail::resolve_json_object_output_rows(
      states, {""}, {0}, {"req-a"}, &output_rows, &output_errors, &error));
  EXPECT_NE(error.find("non-empty"), std::string::npos);
}

TEST(JsonObjectOutputBindingTest, ReportsMissingExpectedOutputRow) {
  JsonObjectGrammar grammar = make_grammar();
  std::vector<JsonObjectGrammarState> states = {grammar.initial_state()};
  std::vector<int32_t> output_rows;
  std::vector<JsonObjectOutputError> output_errors;
  std::string error;

  ASSERT_TRUE(detail::resolve_json_object_output_rows(
      states, {"req-a"}, {0}, {"req-b"}, &output_rows, &output_errors, &error))
      << error;
  EXPECT_EQ(output_rows, std::vector<int32_t>({-2}));
  ASSERT_EQ(output_errors.size(), 1u);
  EXPECT_NE(output_errors[0].message.find("missing"), std::string::npos);
}

TEST(JsonObjectOutputBindingTest, ReportsUnexpectedOutputRow) {
  JsonObjectGrammar grammar = make_grammar();
  std::vector<JsonObjectGrammarState> states = {grammar.initial_state()};
  std::vector<int32_t> output_rows;
  std::vector<JsonObjectOutputError> output_errors;
  std::string error;

  ASSERT_TRUE(detail::resolve_json_object_output_rows(states,
                                                      {"req-b"},
                                                      {-1},
                                                      {"req-a", "req-b"},
                                                      &output_rows,
                                                      &output_errors,
                                                      &error))
      << error;
  EXPECT_EQ(output_rows, std::vector<int32_t>({-2}));
  ASSERT_EQ(output_errors.size(), 1u);
  EXPECT_NE(output_errors[0].message.find("unexpected"), std::string::npos);
}

TEST(JsonObjectOutputBindingTest, RejectsInvalidNegativePriorOutputRow) {
  JsonObjectGrammar grammar = make_grammar();
  std::vector<JsonObjectGrammarState> states = {grammar.initial_state()};
  std::vector<int32_t> output_rows;
  std::vector<JsonObjectOutputError> output_errors;
  std::string error;

  EXPECT_FALSE(detail::resolve_json_object_output_rows(states,
                                                       {"req-a"},
                                                       {-2},
                                                       {"req-a"},
                                                       &output_rows,
                                                       &output_errors,
                                                       &error));
  EXPECT_NE(error.find("-1 or non-negative"), std::string::npos);
}

TEST(JsonObjectOutputBindingTest, KeepsRowErrorsInStateOrder) {
  JsonObjectGrammar grammar = make_grammar();
  std::vector<JsonObjectGrammarState> states = {grammar.initial_state(),
                                                grammar.initial_state(),
                                                grammar.initial_state()};
  std::vector<int32_t> output_rows;
  std::vector<JsonObjectOutputError> output_errors;
  std::string error;

  ASSERT_TRUE(detail::resolve_json_object_output_rows(
      states,
      {"duplicated", "unexpected", "missing"},
      {0, -1, 3},
      {"duplicated", "duplicated", "unexpected"},
      &output_rows,
      &output_errors,
      &error))
      << error;
  EXPECT_EQ(output_rows, std::vector<int32_t>({-2, -2, -2}));
  ASSERT_EQ(output_errors.size(), 3u);
  EXPECT_EQ(output_errors[0].sample_sequence_id, "duplicated");
  EXPECT_EQ(output_errors[0].message, "duplicate prior sampled output rows");
  EXPECT_EQ(output_errors[1].sample_sequence_id, "unexpected");
  EXPECT_EQ(output_errors[1].message, "unexpected prior sampled output row");
  EXPECT_EQ(output_errors[2].sample_sequence_id, "missing");
  EXPECT_EQ(output_errors[2].message, "missing prior sampled output row");
}

TEST(JsonObjectOutputBindingTest, ValidatesPriorRowsBeforeConstrainedIdentity) {
  JsonObjectGrammar grammar = make_grammar();
  std::vector<JsonObjectGrammarState> states = {grammar.initial_state()};
  std::vector<int32_t> output_rows;
  std::vector<JsonObjectOutputError> output_errors;
  std::string error;

  EXPECT_FALSE(detail::resolve_json_object_output_rows(
      states, {""}, {-2}, {}, &output_rows, &output_errors, &error));
  EXPECT_EQ(error, "JSON prior output row must be -1 or non-negative");
  EXPECT_TRUE(output_rows.empty());
  EXPECT_TRUE(output_errors.empty());
}

}  // namespace
}  // namespace xllm

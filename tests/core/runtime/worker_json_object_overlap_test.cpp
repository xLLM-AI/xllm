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

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "core/framework/sampling/json_object_grammar.h"
#include "core/runtime/json_object_output_rows.h"

namespace xllm {
namespace {

JsonObjectGrammar make_grammar() {
  return JsonObjectGrammar({"{", "}", "stop"},
                           /*stop_token_ids=*/{2});
}

TEST(WorkerJsonObjectOverlapTest, FindsStableRequestIdOverlap) {
  EXPECT_TRUE(
      detail::has_request_id_overlap({"req-a", "req-b"}, {"req-a", "req-b"}));
  EXPECT_TRUE(
      detail::has_request_id_overlap({"req-b", "req-a"}, {"req-a", "req-b"}));
}

TEST(WorkerJsonObjectOverlapTest, FindsOverlapAfterInsertedFirstRequest) {
  EXPECT_TRUE(
      detail::has_request_id_overlap({"req-new", "req-b"}, {"req-a", "req-b"}));
}

TEST(WorkerJsonObjectOverlapTest, HandlesMissingDuplicateAndEmptyRequestIds) {
  EXPECT_FALSE(
      detail::has_request_id_overlap({"req-c", "req-d"}, {"req-a", "req-b"}));
  EXPECT_TRUE(
      detail::has_request_id_overlap({"req-b", "req-b"}, {"req-a", "req-b"}));
  EXPECT_TRUE(detail::has_request_id_overlap({}, {"req-a"}));
  EXPECT_TRUE(detail::has_request_id_overlap({"req-a"}, {}));
  EXPECT_TRUE(
      detail::has_request_id_overlap({"", "req-b"}, {"req-a", "req-b"}));
  EXPECT_FALSE(detail::has_request_id_overlap({"", ""}, {"req-a"}));
}

TEST(WorkerJsonObjectOverlapTest, ResolvesReorderedAndInsertedRows) {
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

TEST(WorkerJsonObjectOverlapTest, ReportsDuplicateConstrainedSequenceId) {
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

TEST(WorkerJsonObjectOverlapTest, IgnoresDuplicateUnconstrainedRows) {
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

TEST(WorkerJsonObjectOverlapTest, RejectsEmptyConstrainedSequenceId) {
  JsonObjectGrammar grammar = make_grammar();
  std::vector<JsonObjectGrammarState> states = {grammar.initial_state()};
  std::vector<int32_t> output_rows;
  std::vector<JsonObjectOutputError> output_errors;
  std::string error;

  EXPECT_FALSE(detail::resolve_json_object_output_rows(
      states, {""}, {0}, {"req-a"}, &output_rows, &output_errors, &error));
  EXPECT_NE(error.find("non-empty"), std::string::npos);
}

TEST(WorkerJsonObjectOverlapTest, ReportsMissingExpectedOutputRow) {
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

TEST(WorkerJsonObjectOverlapTest, ReportsUnexpectedOutputRow) {
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

TEST(WorkerJsonObjectOverlapTest, RejectsInvalidNegativePriorOutputRow) {
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

TEST(WorkerJsonObjectOverlapTest, SanitizesOnlyFailedRowPlaceholder) {
  torch::Tensor token_ids = torch::tensor({-1, 17, -2}, torch::kInt);
  std::string error;

  ASSERT_TRUE(detail::sanitize_json_object_error_token_ids(
      &token_ids,
      {"req-failed#0", "req-healthy#0"},
      {0, 1},
      {{"req-failed#0", "row mismatch"}},
      &error))
      << error;
  EXPECT_TRUE(torch::equal(token_ids, torch::tensor({0, 17, -2})));
}

TEST(WorkerJsonObjectOverlapTest, RejectsAmbiguousFailedRowPlaceholder) {
  torch::Tensor token_ids = torch::tensor({-1, 17, -1}, torch::kInt);
  std::string error;

  EXPECT_FALSE(detail::sanitize_json_object_error_token_ids(
      &token_ids,
      {"req-failed#0"},
      {0},
      {{"req-failed#0", "row mismatch"}},
      &error));
  EXPECT_NE(error.find("exactly one"), std::string::npos);
  EXPECT_TRUE(torch::equal(token_ids, torch::tensor({-1, 17, -1})));
}

}  // namespace
}  // namespace xllm

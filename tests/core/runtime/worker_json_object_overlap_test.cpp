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

TEST(WorkerJsonObjectOverlapTest, ResolvesReorderedAndInsertedRows) {
  JsonObjectGrammar grammar = make_grammar();
  std::vector<JsonObjectGrammarState> states = {grammar.initial_state(),
                                                grammar.initial_state()};
  std::vector<int32_t> output_rows;
  std::string error;

  ASSERT_TRUE(detail::resolve_json_object_output_rows(
      states, {"req-b", "req-new"}, {"req-a", "req-b"}, &output_rows, &error))
      << error;
  EXPECT_EQ(output_rows, std::vector<int32_t>({1, -1}));
}

TEST(WorkerJsonObjectOverlapTest, RejectsDuplicateConstrainedSequenceId) {
  JsonObjectGrammar grammar = make_grammar();
  std::vector<JsonObjectGrammarState> states = {grammar.initial_state()};
  std::vector<int32_t> output_rows;
  std::string error;

  EXPECT_FALSE(detail::resolve_json_object_output_rows(
      states, {"req-a"}, {"req-a", "req-a"}, &output_rows, &error));
  EXPECT_NE(error.find("duplicate"), std::string::npos);
}

TEST(WorkerJsonObjectOverlapTest, IgnoresDuplicateUnconstrainedRows) {
  std::vector<JsonObjectGrammarState> states = {JsonObjectGrammarState()};
  std::vector<int32_t> output_rows;
  std::string error;

  ASSERT_TRUE(detail::resolve_json_object_output_rows(
      states, {"req-plain"}, {"req-plain", "req-plain"}, &output_rows, &error))
      << error;
  EXPECT_EQ(output_rows, std::vector<int32_t>({-1}));
}

TEST(WorkerJsonObjectOverlapTest, RejectsEmptyConstrainedSequenceId) {
  JsonObjectGrammar grammar = make_grammar();
  std::vector<JsonObjectGrammarState> states = {grammar.initial_state()};
  std::vector<int32_t> output_rows;
  std::string error;

  EXPECT_FALSE(detail::resolve_json_object_output_rows(
      states, {""}, {"req-a"}, &output_rows, &error));
  EXPECT_NE(error.find("non-empty"), std::string::npos);
}

}  // namespace
}  // namespace xllm

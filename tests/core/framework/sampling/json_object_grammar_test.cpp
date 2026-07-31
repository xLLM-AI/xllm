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

#include "core/framework/sampling/json_object_grammar.h"

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

namespace xllm {
namespace {

JsonObjectGrammar make_grammar(bool add_model_only_token = false) {
  std::vector<std::string> token_pieces = {
      "{", "}", "\"", "a", "b", ":", ",", "[",  "]", "true", "false", "null",
      "1", "-", "0",  ".", "2", "e", " ", "\\", "n", "u",    "0041",  "x"};
  if (add_model_only_token) {
    token_pieces.emplace_back();
  }
  return JsonObjectGrammar(std::move(token_pieces),
                           {add_model_only_token ? 25 : 24});
}

TEST(JsonObjectGrammarTest, AcceptsNestedValuesAndEscapes) {
  JsonObjectGrammar grammar = make_grammar();
  JsonObjectGrammarState state = grammar.initial_state();

  EXPECT_TRUE(state.accept_piece("{\"a\":[true, {\"b\":\"x\\n"));
  EXPECT_TRUE(state.accept_piece("\"}]}"));
  EXPECT_TRUE(state.is_complete());
  EXPECT_TRUE(state.can_accept_piece(" "));
  EXPECT_FALSE(state.can_accept_piece("x"));
}

TEST(JsonObjectGrammarTest, RejectsInvalidRootAndObjectSyntax) {
  JsonObjectGrammar grammar = make_grammar();
  JsonObjectGrammarState state = grammar.initial_state();

  EXPECT_FALSE(state.accept_piece("["));
  EXPECT_FALSE(state.is_valid());

  state = grammar.initial_state();
  EXPECT_TRUE(state.accept_piece("{\"a\""));
  EXPECT_FALSE(state.can_accept_piece("}"));
  EXPECT_TRUE(state.can_accept_piece(":"));

  state = grammar.initial_state();
  EXPECT_FALSE(state.accept_piece("{\"a\":1,}"));

  state = grammar.initial_state();
  EXPECT_FALSE(state.accept_piece("{\"a\":[1,]}"));
}

TEST(JsonObjectGrammarTest, SupportsNumbersLiteralsAndWhitespace) {
  JsonObjectGrammar grammar = make_grammar();
  JsonObjectGrammarState state = grammar.initial_state();

  EXPECT_TRUE(state.accept_piece("{ \"a\": -1.2e+2, \"b\": null}"));
  EXPECT_TRUE(state.is_complete());

  state = grammar.initial_state();
  EXPECT_TRUE(state.accept_piece("{\"a\":0}"));
  EXPECT_FALSE(state.can_accept_piece("1"));
}

TEST(JsonObjectGrammarTest, TemporaryAdvanceDoesNotCommitState) {
  JsonObjectGrammar grammar = make_grammar();
  JsonObjectGrammarState state = grammar.initial_state();

  EXPECT_TRUE(state.can_accept_piece("{"));
  EXPECT_TRUE(state.accept_piece("{"));
  EXPECT_TRUE(state.can_accept_piece("}"));
  EXPECT_FALSE(state.is_complete());
  EXPECT_TRUE(state.accept_piece("\"a\":1}"));
  EXPECT_TRUE(state.is_complete());
}

TEST(JsonObjectGrammarTest, MtpDraftStatesAccumulateAcceptedTokens) {
  JsonObjectGrammar grammar = make_grammar();
  std::vector<JsonObjectGrammarState> original_states = {
      grammar.initial_state()};

  // The second draft row must be built from S+d0, not from the original S.
  const std::vector<JsonObjectGrammarState> after_d0 =
      advance_json_object_states(original_states, {0});
  const std::vector<JsonObjectGrammarState> cumulative_states =
      advance_json_object_states(after_d0, {2});
  const std::vector<JsonObjectGrammarState> reset_states =
      advance_json_object_states(original_states, {2});

  EXPECT_TRUE(cumulative_states[0].can_accept_token(/*d2=*/3));
  EXPECT_FALSE(reset_states[0].can_accept_token(/*d2=*/3));
}

TEST(JsonObjectGrammarTest, SnapshotRestoresCommittedState) {
  JsonObjectGrammar grammar = make_grammar();
  JsonObjectGrammarState state = grammar.initial_state();

  EXPECT_TRUE(state.accept_token(0));
  EXPECT_TRUE(state.accept_token(2));
  EXPECT_TRUE(state.accept_token(3));
  EXPECT_TRUE(state.accept_token(2));

  const JsonObjectGrammarSnapshot snapshot = state.snapshot();
  ASSERT_TRUE(snapshot.enabled);
  EXPECT_FALSE(snapshot.reasoning_enabled);
  EXPECT_EQ(snapshot.token_ids, std::vector<int32_t>({0, 2, 3, 2}));

  JsonObjectGrammarState restored = grammar.restore_state(snapshot);
  EXPECT_TRUE(restored.is_valid());
  EXPECT_TRUE(restored.can_accept_piece(":1}"));
}

TEST(JsonObjectGrammarTest, ReasoningIsUnconstrainedUntilEndMarker) {
  JsonObjectGrammar grammar({"{", "}", "reasoning", "<think>", "</think>"},
                            /*stop_token_ids=*/{1},
                            {3, 4});
  JsonObjectGrammarState state =
      grammar.initial_state(/*reasoning_phase=*/true);

  EXPECT_TRUE(state.in_reasoning());
  EXPECT_EQ(grammar.allowed_token_ids(state).size(), grammar.vocab_size());
  EXPECT_TRUE(state.accept_token(2));
  EXPECT_TRUE(state.accept_token(3));
  EXPECT_TRUE(state.accept_token(4));
  EXPECT_FALSE(state.in_reasoning());
  EXPECT_TRUE(state.can_accept_token(0));
  EXPECT_FALSE(state.can_accept_token(1));
}

TEST(JsonObjectGrammarTest, MaskHasNoUnrestrictedFailureFallback) {
  JsonObjectGrammar grammar = make_grammar();
  JsonObjectGrammarState state = grammar.initial_state();
  torch::Tensor mask = grammar.build_filter_mask(state);

  EXPECT_EQ(mask.size(0), static_cast<int64_t>(grammar.vocab_size()));
  EXPECT_EQ(mask.index({0}).item<float>(), 0.0F);
  EXPECT_LT(mask.index({1}).item<float>(), -1.0F);
}

TEST(JsonObjectGrammarTest, KeepsModelOnlyTokensMasked) {
  JsonObjectGrammar grammar = make_grammar(/*add_model_only_token=*/true);
  JsonObjectGrammarState state = grammar.initial_state();
  torch::Tensor mask = grammar.build_filter_mask(state);

  EXPECT_EQ(mask.size(0), 25);
  EXPECT_LT(mask.index({24}).item<float>(), -1.0F);
}

TEST(JsonObjectGrammarTest, BuildsMixedBatchMask) {
  JsonObjectGrammar grammar = make_grammar();
  std::vector<JsonObjectGrammarState> states = {grammar.initial_state(),
                                                JsonObjectGrammarState()};

  torch::Tensor mask = build_json_object_filter_mask(states);

  ASSERT_EQ(mask.sizes(), torch::IntArrayRef({2, 24}));
  EXPECT_EQ(mask.index({0, 0}).item<float>(), 0.0F);
  EXPECT_LT(mask.index({0, 1}).item<float>(), -1.0F);
  EXPECT_EQ(mask.index({1, 0}).item<float>(), 0.0F);
  EXPECT_EQ(mask.index({1, 1}).item<float>(), 0.0F);
}

TEST(JsonObjectGrammarTest, StopsAtObjectCompletionWithMultipleStopTokens) {
  JsonObjectGrammar grammar({"{", "}", "\"", "a", ":", "1", " ", "", ""},
                            /*stop_token_ids=*/{7, 8});
  JsonObjectGrammarState state = grammar.initial_state();

  ASSERT_TRUE(state.accept_piece("{\"a\":1}"));
  EXPECT_TRUE(state.can_accept_token(/*stop_token_id=*/7));
  EXPECT_TRUE(state.can_accept_token(/*stop_token_id=*/8));
  EXPECT_FALSE(state.can_accept_token(/*trailing_space_token_id=*/6));
}

}  // namespace
}  // namespace xllm

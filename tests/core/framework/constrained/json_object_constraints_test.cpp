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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "core/framework/constrained/json_object_grammar.h"

namespace xllm {
namespace {

TEST(JsonObjectConstraintsTest, AdvancesStatesWithReorderedTokenRows) {
  JsonObjectGrammar grammar({"{", "}", "\"a\"", ":", "1", "stop"},
                            /*stop_token_ids=*/{5});
  std::vector<JsonObjectGrammarState> states = {grammar.initial_state(),
                                                grammar.initial_state(),
                                                JsonObjectGrammarState(),
                                                grammar.initial_state()};
  ASSERT_TRUE(states[1].accept_token(/*open_object=*/0));

  const auto violations =
      apply_json_constraints(states,
                             torch::tensor({1, 1, 0}, torch::kInt64),
                             /*token_ids=*/{2, 0, 1, -1});

  EXPECT_TRUE(violations.empty());
  EXPECT_EQ(states[0].snapshot().token_ids, std::vector<int32_t>({0}));
  EXPECT_EQ(states[1].snapshot().token_ids, std::vector<int32_t>({0, 1}));
  EXPECT_TRUE(states[1].is_complete());
  EXPECT_FALSE(states[2].initialized());
  EXPECT_TRUE(states[3].snapshot().token_ids.empty());
}

TEST(JsonObjectConstraintsTest, BuildsCurrentMasksWithoutPriorTokens) {
  JsonObjectGrammar grammar({"{", "}", "stop"}, /*stop_token_ids=*/{2});
  std::vector<JsonObjectGrammarState> states = {grammar.initial_state(),
                                                JsonObjectGrammarState(),
                                                grammar.initial_state()};
  ASSERT_TRUE(states[2].accept_token(/*open_object=*/0));
  const uint64_t initial_fingerprint = states[0].fingerprint();
  const uint64_t continued_fingerprint = states[2].fingerprint();

  const auto violations = apply_json_constraints(
      states, torch::Tensor(), /*token_ids=*/{-1, -1, -1});

  EXPECT_TRUE(violations.empty());
  EXPECT_EQ(states[0].fingerprint(), initial_fingerprint);
  EXPECT_EQ(states[2].fingerprint(), continued_fingerprint);
  EXPECT_EQ(states[2].snapshot().token_ids, std::vector<int32_t>({0}));
}

TEST(JsonObjectConstraintsTest, IgnoresNegativeOneDimensionalTokens) {
  JsonObjectGrammar grammar({"{", "}", "stop"}, /*stop_token_ids=*/{2});
  std::vector<JsonObjectGrammarState> states = {grammar.initial_state(),
                                                grammar.initial_state()};

  const auto violations =
      apply_json_constraints(states,
                             torch::tensor({-1, -2}, torch::kInt64),
                             /*token_ids=*/{0, 1});

  EXPECT_TRUE(violations.empty());
  EXPECT_TRUE(states[0].initialized());
  EXPECT_TRUE(states[1].initialized());
  EXPECT_TRUE(states[0].snapshot().token_ids.empty());
  EXPECT_TRUE(states[1].snapshot().token_ids.empty());
}

TEST(JsonObjectConstraintsTest, StopsAtFirstNegativeSpeculativeToken) {
  JsonObjectGrammar grammar({"{", "}", "stop"}, /*stop_token_ids=*/{2});
  std::vector<JsonObjectGrammarState> states = {grammar.initial_state(),
                                                grammar.initial_state()};
  const torch::Tensor accepted_tokens =
      torch::tensor({{0, -2}, {1, 1}, {-1, 0}, {0, 0}}, torch::kInt64)
          .transpose(0, 1);
  ASSERT_FALSE(accepted_tokens.is_contiguous());

  const auto violations =
      apply_json_constraints(states, accepted_tokens, /*token_ids=*/{0, 1});

  EXPECT_TRUE(violations.empty());
  EXPECT_TRUE(states[0].is_complete());
  EXPECT_EQ(states[0].snapshot().token_ids, std::vector<int32_t>({0, 1}));
  EXPECT_TRUE(states[1].initialized());
  EXPECT_TRUE(states[1].snapshot().token_ids.empty());
}

TEST(JsonObjectConstraintsTest, ReportsInvalidOneDimensionalToken) {
  JsonObjectGrammar grammar({"{", "}", "stop"}, /*stop_token_ids=*/{2});
  std::vector<JsonObjectGrammarState> states = {grammar.initial_state(),
                                                grammar.initial_state()};
  ASSERT_TRUE(states[0].accept_token(/*open_object=*/0));
  const uint64_t expected_fingerprint = states[1].fingerprint();

  const auto violations =
      apply_json_constraints(states,
                             torch::tensor({1, 1}, torch::kInt64),
                             /*token_ids=*/{0, 1});

  ASSERT_EQ(violations.size(), 1);
  EXPECT_EQ(violations[0].sequence_index, 1);
  EXPECT_EQ(violations[0].token_offset, 0);
  EXPECT_EQ(violations[0].token_id, 1);
  EXPECT_EQ(violations[0].committed_tokens, 0);
  EXPECT_EQ(violations[0].state_fingerprint, expected_fingerprint);
  EXPECT_TRUE(states[0].is_complete());
  EXPECT_EQ(states[0].snapshot().token_ids, std::vector<int32_t>({0, 1}));
  EXPECT_FALSE(states[1].initialized());
}

TEST(JsonObjectConstraintsTest,
     ReportsPartialProgressWithoutBlockingOtherStates) {
  JsonObjectGrammar grammar({"{", "}", "\"a\"", ":", "1", "stop"},
                            /*stop_token_ids=*/{5});
  std::vector<JsonObjectGrammarState> states = {grammar.initial_state(),
                                                grammar.initial_state()};
  ASSERT_TRUE(states[0].accept_token(/*open_object=*/0));
  JsonObjectGrammarState expected_state = grammar.initial_state();
  ASSERT_TRUE(expected_state.accept_token(/*open_object=*/0));
  ASSERT_TRUE(expected_state.accept_token(/*key=*/2));
  const uint64_t expected_fingerprint = expected_state.fingerprint();

  const auto violations = apply_json_constraints(
      states,
      torch::tensor({{0, 1, -1}, {2, 2, -1}}, torch::kInt64),
      /*token_ids=*/{1, 0});

  ASSERT_EQ(violations.size(), 1);
  EXPECT_EQ(violations[0].sequence_index, 0);
  EXPECT_EQ(violations[0].token_offset, 1);
  EXPECT_EQ(violations[0].token_id, 2);
  EXPECT_EQ(violations[0].committed_tokens, 2);
  EXPECT_EQ(violations[0].state_fingerprint, expected_fingerprint);
  EXPECT_FALSE(states[0].initialized());
  EXPECT_TRUE(states[1].is_complete());
  EXPECT_EQ(states[1].snapshot().token_ids, std::vector<int32_t>({0, 1}));
}

}  // namespace
}  // namespace xllm

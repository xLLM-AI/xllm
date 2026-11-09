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

#include "core/framework/speculative/mtp_json_object_state.h"

#include <gtest/gtest.h>

#include <vector>

#include "core/framework/constrained/json_object_grammar.h"

namespace xllm {
namespace {

JsonObjectGrammar make_mtp_grammar() {
  return JsonObjectGrammar({"{", "\"a\"", ":", "1", "}", "stop"},
                           /*stop_token_ids=*/{5});
}

bool can_accept_token(const JsonObjectGrammarState& state, int32_t token_id) {
  JsonObjectGrammarState probe = state;
  return probe.accept_token(token_id);
}

TEST(MtpJsonObjectStateTest, BuildsSequenceMajorRowsForMixedAcceptedDrafts) {
  JsonObjectGrammar grammar = make_mtp_grammar();
  const std::vector<JsonObjectGrammarState> initial_states = {
      grammar.initial_state(), JsonObjectGrammarState()};
  const std::vector<std::vector<int32_t>> draft_token_ids = {
      {0, 4}, {1, 4}, {2, 4}};

  const auto validation_states =
      detail::replay_draft_states(initial_states, draft_token_ids);

  ASSERT_EQ(validation_states.size(), 8u);
  EXPECT_EQ(validation_states[0].fingerprint(),
            initial_states[0].fingerprint());
  for (size_t row = 4; row < validation_states.size(); ++row) {
    EXPECT_FALSE(validation_states[row].initialized());
  }
  EXPECT_EQ(initial_states[0].snapshot().token_ids, std::vector<int32_t>{});
}

TEST(MtpJsonObjectStateTest, FreezesFirstInvalidDraftAndItsSuffix) {
  JsonObjectGrammar grammar = make_mtp_grammar();
  const std::vector<JsonObjectGrammarState> initial_states = {
      grammar.initial_state(), JsonObjectGrammarState()};
  const std::vector<std::vector<int32_t>> draft_token_ids = {
      {4, 0}, {-1, 0}, {-1, 0}};

  const auto validation_states =
      detail::replay_draft_states(initial_states, draft_token_ids);

  ASSERT_EQ(validation_states.size(), 8u);
  for (size_t row = 0; row < 4; ++row) {
    EXPECT_EQ(validation_states[row].fingerprint(),
              initial_states[0].fingerprint());
    EXPECT_TRUE(validation_states[row].is_valid());
  }
  for (size_t row = 4; row < validation_states.size(); ++row) {
    EXPECT_FALSE(validation_states[row].initialized());
  }
}

TEST(MtpJsonObjectStateTest, PreservesAcceptedPrefixAfterMiddleInvalidDraft) {
  JsonObjectGrammar grammar = make_mtp_grammar();
  const std::vector<JsonObjectGrammarState> initial_states = {
      grammar.initial_state()};
  const std::vector<std::vector<int32_t>> draft_token_ids = {{0}, {2}, {-1}};

  const auto validation_states =
      detail::replay_draft_states(initial_states, draft_token_ids);

  ASSERT_EQ(validation_states.size(), 4u);
  EXPECT_EQ(validation_states[0].fingerprint(),
            initial_states[0].fingerprint());
  EXPECT_TRUE(validation_states[1].is_valid());
  EXPECT_NE(validation_states[1].fingerprint(),
            initial_states[0].fingerprint());
  EXPECT_EQ(validation_states[2].fingerprint(),
            validation_states[1].fingerprint());
  EXPECT_EQ(validation_states[3].fingerprint(),
            validation_states[1].fingerprint());
}

TEST(MtpJsonObjectStateTest, FiveDraftRowsReplayFromCommittedState) {
  JsonObjectGrammar grammar = make_mtp_grammar();
  const std::vector<JsonObjectGrammarState> initial_states = {
      grammar.initial_state()};
  const std::vector<int32_t> draft_tokens = {0, 1, 2, 3, 4};
  const std::vector<std::vector<int32_t>> draft_token_ids = {
      {0}, {1}, {2}, {3}, {4}};

  const std::vector<JsonObjectGrammarState> validation_states =
      detail::replay_draft_states(initial_states, draft_token_ids);

  ASSERT_EQ(validation_states.size(), 6u);
  JsonObjectGrammarState committed_state = initial_states[0];
  for (size_t token_idx = 0; token_idx < draft_tokens.size(); ++token_idx) {
    EXPECT_EQ(validation_states[token_idx].fingerprint(),
              committed_state.fingerprint());
    ASSERT_TRUE(can_accept_token(validation_states[token_idx],
                                 draft_tokens[token_idx]));
    ASSERT_TRUE(committed_state.accept_token(draft_tokens[token_idx]));
  }
  EXPECT_EQ(validation_states.back().fingerprint(),
            committed_state.fingerprint());
  EXPECT_TRUE(can_accept_token(validation_states.back(), /*stop=*/5));
  EXPECT_TRUE(committed_state.accept_token(/*stop=*/5));
}

TEST(MtpJsonObjectStateTest,
     ReasoningDisabledValidationRowsRejectReasoningEndMarker) {
  JsonObjectGrammar grammar(
      {"{", "\"a\"", ":", "1", "}", "stop", " thinking", " response"},
      /*stop_token_ids=*/{5},
      /*reasoning_end_token_ids=*/{7});
  const std::vector<JsonObjectGrammarState> initial_states = {
      grammar.initial_state(/*reasoning_phase=*/false)};
  const std::vector<int32_t> draft_tokens = {0, 1, 2, 3, 4};
  const std::vector<std::vector<int32_t>> draft_token_ids = {
      {0}, {1}, {2}, {3}, {4}};

  const std::vector<JsonObjectGrammarState> validation_states =
      detail::replay_draft_states(initial_states, draft_token_ids);

  ASSERT_EQ(validation_states.size(), 6u);
  for (const JsonObjectGrammarState& state : validation_states) {
    EXPECT_FALSE(state.in_reasoning());
    EXPECT_FALSE(can_accept_token(state, /*reasoning_end=*/7));
  }
}

TEST(MtpJsonObjectStateTest, ReplaysFiveStepMixedAcceptedOutputRows) {
  JsonObjectGrammar grammar = make_mtp_grammar();
  const std::vector<JsonObjectGrammarState> initial_states = {
      grammar.initial_state(), JsonObjectGrammarState()};
  const int64_t accepted_tokens[] = {
      0,
      1,
      2,
      3,
      4,
      5,
      5,
      4,
      3,
      2,
      1,
      0,
  };

  const std::vector<JsonTokenMismatch> mismatches =
      detail::find_json_token_mismatches(
          initial_states, accepted_tokens, /*num_rows=*/2, /*row_width=*/6);

  EXPECT_TRUE(mismatches.empty());
}

TEST(MtpJsonObjectStateTest, ReportsFirstAcceptedOutputMismatchPerSequence) {
  JsonObjectGrammar grammar = make_mtp_grammar();
  JsonObjectGrammarState committed_state = grammar.initial_state();
  ASSERT_TRUE(committed_state.accept_token(/*open_object=*/0));
  const std::vector<JsonObjectGrammarState> initial_states = {
      committed_state, JsonObjectGrammarState()};
  const int64_t accepted_tokens[] = {
      1,
      4,
      3,
      -1,
      -1,
      -1,
      5,
      4,
      3,
      2,
      1,
      0,
  };

  const std::vector<JsonTokenMismatch> mismatches =
      detail::find_json_token_mismatches(
          initial_states, accepted_tokens, /*num_rows=*/2, /*row_width=*/6);

  ASSERT_EQ(mismatches.size(), 1u);
  EXPECT_EQ(mismatches[0].sequence_index, 0u);
  EXPECT_EQ(mismatches[0].token_offset, 1u);
  EXPECT_EQ(mismatches[0].token_id, 4);
  EXPECT_EQ(mismatches[0].committed_tokens, 2u);
}

TEST(MtpJsonObjectStateTest, AcceptedOutputStopsAtAnyNegativePadding) {
  JsonObjectGrammar grammar = make_mtp_grammar();
  const std::vector<JsonObjectGrammarState> initial_states = {
      grammar.initial_state()};
  const int64_t accepted_tokens[] = {0, -2};

  const std::vector<JsonTokenMismatch> mismatches =
      detail::find_json_token_mismatches(
          initial_states, accepted_tokens, /*num_rows=*/1, /*row_width=*/2);
  EXPECT_TRUE(mismatches.empty());
}

}  // namespace
}  // namespace xllm

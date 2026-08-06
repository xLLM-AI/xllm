/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/runtime/mtp_json_object_state.h"

#include <gtest/gtest.h>

#include <vector>

#include "core/framework/sampling/json_object_grammar.h"

namespace xllm {
namespace {

JsonObjectGrammar make_mtp_grammar() {
  return JsonObjectGrammar({"{", "\"a\"", ":", "1", "}", "stop"},
                           /*stop_token_ids=*/{5});
}

TEST(MtpJsonObjectStateTest, BuildsSequenceMajorRowsForMixedAcceptedDrafts) {
  JsonObjectGrammar grammar = make_mtp_grammar();
  const std::vector<JsonObjectGrammarState> initial_states = {
      grammar.initial_state(), JsonObjectGrammarState()};
  std::vector<JsonObjectGrammarState> current_states = initial_states;
  std::vector<uint8_t> invalid_suffix(2, 0);
  detail::JsonDraftValidationScratch scratch;

  EXPECT_FALSE(detail::append_json_draft_step(
      current_states, invalid_suffix, {0, 4}, scratch));
  EXPECT_FALSE(detail::append_json_draft_step(
      current_states, invalid_suffix, {1, 4}, scratch));
  EXPECT_FALSE(detail::append_json_draft_step(
      current_states, invalid_suffix, {2, 4}, scratch));

  std::vector<uint8_t> invalid_draft;
  const auto validation_states = detail::build_json_validation_states(
      initial_states, scratch, invalid_draft);

  ASSERT_EQ(validation_states.size(), 8u);
  EXPECT_EQ(validation_states[0].fingerprint(),
            initial_states[0].fingerprint());
  EXPECT_EQ(validation_states[1].fingerprint(),
            scratch.states_after[0][0].fingerprint());
  EXPECT_EQ(validation_states[2].fingerprint(),
            scratch.states_after[1][0].fingerprint());
  EXPECT_EQ(validation_states[3].fingerprint(),
            scratch.states_after[2][0].fingerprint());
  for (size_t row = 4; row < validation_states.size(); ++row) {
    EXPECT_FALSE(validation_states[row].initialized());
  }
  EXPECT_EQ(invalid_draft, std::vector<uint8_t>({0, 0, 0, 0, 0, 0}));
  EXPECT_EQ(initial_states[0].snapshot().token_ids, std::vector<int32_t>({}));
}

TEST(MtpJsonObjectStateTest, FreezesFirstInvalidDraftAndItsSuffix) {
  JsonObjectGrammar grammar = make_mtp_grammar();
  const std::vector<JsonObjectGrammarState> initial_states = {
      grammar.initial_state(), JsonObjectGrammarState()};
  std::vector<JsonObjectGrammarState> current_states = initial_states;
  std::vector<uint8_t> invalid_suffix(2, 0);
  detail::JsonDraftValidationScratch scratch;

  EXPECT_TRUE(detail::append_json_draft_step(
      current_states, invalid_suffix, {4, 0}, scratch));
  EXPECT_FALSE(detail::append_json_draft_step(
      current_states, invalid_suffix, {-1, 0}, scratch));
  EXPECT_FALSE(detail::append_json_draft_step(
      current_states, invalid_suffix, {-1, 0}, scratch));

  std::vector<uint8_t> invalid_draft;
  const auto validation_states = detail::build_json_validation_states(
      initial_states, scratch, invalid_draft);

  EXPECT_EQ(invalid_draft, std::vector<uint8_t>({1, 1, 1, 0, 0, 0}));
  for (size_t row = 0; row < 4; ++row) {
    EXPECT_EQ(validation_states[row].fingerprint(),
              initial_states[0].fingerprint());
  }
}

TEST(MtpJsonObjectStateTest, PreservesAcceptedPrefixAfterMiddleInvalidDraft) {
  JsonObjectGrammar grammar = make_mtp_grammar();
  const std::vector<JsonObjectGrammarState> initial_states = {
      grammar.initial_state()};
  std::vector<JsonObjectGrammarState> current_states = initial_states;
  std::vector<uint8_t> invalid_suffix(1, 0);
  detail::JsonDraftValidationScratch scratch;

  EXPECT_FALSE(detail::append_json_draft_step(
      current_states, invalid_suffix, {0}, scratch));
  EXPECT_TRUE(detail::append_json_draft_step(
      current_states, invalid_suffix, {2}, scratch));
  EXPECT_FALSE(detail::append_json_draft_step(
      current_states, invalid_suffix, {-1}, scratch));

  std::vector<uint8_t> invalid_draft;
  const auto validation_states = detail::build_json_validation_states(
      initial_states, scratch, invalid_draft);

  EXPECT_EQ(invalid_draft, std::vector<uint8_t>({0, 1, 1}));
  ASSERT_EQ(validation_states.size(), 4u);
  EXPECT_EQ(validation_states[1].fingerprint(),
            scratch.states_after[0][0].fingerprint());
  EXPECT_EQ(validation_states[2].fingerprint(),
            scratch.states_after[0][0].fingerprint());
  EXPECT_EQ(validation_states[3].fingerprint(),
            scratch.states_after[0][0].fingerprint());
}

TEST(MtpJsonObjectStateTest, CopiesContiguousHostDraftTokensInBulk) {
  const int64_t token_ids[] = {0, 4, -1, 5};

  EXPECT_EQ(detail::copy_json_draft_token_ids(token_ids, 4),
            std::vector<int32_t>({0, 4, -1, 5}));
  EXPECT_TRUE(detail::copy_json_draft_token_ids(nullptr, 0).empty());
}

}  // namespace
}  // namespace xllm

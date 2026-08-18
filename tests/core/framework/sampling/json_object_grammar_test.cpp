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
#include <thread>
#include <utility>
#include <vector>

#include "core/common/metrics.h"
#include "core/framework/tokenizer/tokenizer.h"

namespace xllm {
namespace {

class LossySingleTokenTokenizer final : public Tokenizer {
 public:
  std::string decode(const Slice<int32_t>& ids,
                     bool skip_special_tokens) const override {
    (void)skip_special_tokens;
    std::string decoded;
    for (const int32_t id : ids) {
      if (id == 2) {
        decoded += "\xEF\xBF\xBD";
      } else {
        decoded += decode_token(id);
      }
    }
    return decoded;
  }

  std::string decode_token(int32_t id) const override {
    switch (id) {
      case 0:
        return "{";
      case 1:
        return "\"";
      case 2:
        return "a\"";
      case 3:
        return ":";
      case 4:
        return "1";
      case 5:
        return "}";
      case 6:
        return "<eos>";
      default:
        return "";
    }
  }

  std::string id_to_token(int32_t id) const override {
    return decode_token(id);
  }

  size_t vocab_size() const override { return 7; }
};

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
  const uint64_t fingerprint = state.fingerprint();
  EXPECT_FALSE(state.can_accept_piece("]"));
  EXPECT_EQ(state.fingerprint(), fingerprint);
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
  const uint64_t fingerprint = state.fingerprint();

  JsonObjectGrammarState restored = grammar.restore_state(snapshot);
  EXPECT_TRUE(restored.is_valid());
  EXPECT_EQ(restored.fingerprint(), fingerprint);
  EXPECT_TRUE(restored.can_accept_piece(":1}"));
}

TEST(JsonObjectGrammarTest, ReasoningIsUnconstrainedUntilEndMarker) {
  JsonObjectGrammar grammar({"{", "}", "reasoning", "<think>", "</think>"},
                            /*stop_token_ids=*/{1},
                            {3, 4});
  JsonObjectGrammarState state =
      grammar.initial_state(/*reasoning_phase=*/true);

  EXPECT_TRUE(state.in_reasoning());
  EXPECT_EQ(grammar.allowed_token_ids(state),
            std::vector<int32_t>({0, 2, 3, 4}));
  const torch::Tensor mask = grammar.build_filter_mask(state);
  EXPECT_EQ(mask.index({0}).item<float>(), 0.0F);
  EXPECT_LT(mask.index({1}).item<float>(), -1.0F);
  EXPECT_EQ(mask.index({2}).item<float>(), 0.0F);
  EXPECT_EQ(mask.index({3}).item<float>(), 0.0F);
  EXPECT_EQ(mask.index({4}).item<float>(), 0.0F);
  EXPECT_TRUE(state.accept_token(2));
  EXPECT_TRUE(state.accept_token(3));
  EXPECT_TRUE(state.accept_token(4));
  EXPECT_FALSE(state.in_reasoning());
  EXPECT_TRUE(state.can_accept_token(0));
  EXPECT_FALSE(state.can_accept_token(1));
}

TEST(JsonObjectGrammarTest, ReasoningDisabledRejectsEndMarkerAtJsonStart) {
  JsonObjectGrammar grammar({"{", "}", "reasoning", "<think>", "</think>"},
                            /*stop_token_ids=*/{1},
                            {3, 4});
  JsonObjectGrammarState state = grammar.initial_state();

  EXPECT_FALSE(state.in_reasoning());
  EXPECT_FALSE(state.can_accept_token(/*reasoning_end=*/4));
  EXPECT_TRUE(state.can_accept_token(/*open_object=*/0));
}

TEST(JsonObjectGrammarTest, RejectsNonEmptyStopTokensBeforeRootCompletion) {
  JsonObjectGrammar grammar({"{",
                             "\"",
                             "a",
                             ":",
                             "1",
                             "x",
                             " ",
                             "e",
                             "}",
                             "tru",
                             "<think>",
                             "</think>"},
                            /*stop_token_ids=*/{5, 6, 7},
                            /*reasoning_end_token_ids=*/{10, 11});

  JsonObjectGrammarState reasoning_state =
      grammar.initial_state(/*reasoning_phase=*/true);
  EXPECT_FALSE(reasoning_state.can_accept_token(/*stop_x=*/5));

  JsonObjectGrammarState key_state = grammar.initial_state();
  ASSERT_TRUE(key_state.accept_piece("{\"a"));
  EXPECT_FALSE(key_state.can_accept_token(/*stop_x=*/5));

  JsonObjectGrammarState string_value_state = grammar.initial_state();
  ASSERT_TRUE(string_value_state.accept_piece("{\"a\":\""));
  EXPECT_FALSE(string_value_state.can_accept_token(/*stop_x=*/5));

  JsonObjectGrammarState number_state = grammar.initial_state();
  ASSERT_TRUE(number_state.accept_piece("{\"a\":1"));
  EXPECT_FALSE(number_state.can_accept_token(/*stop_space=*/6));

  JsonObjectGrammarState literal_state = grammar.initial_state();
  ASSERT_TRUE(literal_state.accept_piece("{\"a\":tru"));
  EXPECT_FALSE(literal_state.can_accept_token(/*stop_e=*/7));
}

TEST(JsonObjectGrammarTest, AcceptsNonEmptyStopTokenAfterRootCompletion) {
  JsonObjectGrammar grammar({"{", "}", "\"", "a", ":", "1", "stop"},
                            /*stop_token_ids=*/{6});
  JsonObjectGrammarState state = grammar.initial_state();

  ASSERT_TRUE(state.accept_piece("{\"a\":1}"));
  EXPECT_TRUE(state.can_accept_token(/*stop_token_id=*/6));
}

TEST(JsonObjectGrammarTest, MatcherFingerprintIgnoresCommittedHistory) {
  JsonObjectGrammar grammar = make_grammar();
  JsonObjectGrammarState via_tokens = grammar.initial_state();
  ASSERT_TRUE(via_tokens.accept_token(/*{=*/0));
  ASSERT_TRUE(via_tokens.accept_token(/*"=*/2));
  ASSERT_TRUE(via_tokens.accept_token(/*a=*/3));
  ASSERT_TRUE(via_tokens.accept_token(/*"=*/2));

  JsonObjectGrammarState via_piece = grammar.initial_state();
  ASSERT_TRUE(via_piece.accept_piece("{\"a\""));

  EXPECT_EQ(via_tokens.fingerprint(), via_piece.fingerprint());
  EXPECT_EQ(grammar.allowed_token_ids(via_tokens),
            grammar.allowed_token_ids(via_piece));

  JsonObjectGrammarState restored =
      grammar.restore_state(via_tokens.snapshot());
  EXPECT_EQ(restored.fingerprint(), via_piece.fingerprint());
}

TEST(JsonObjectGrammarTest, BitmaskMatchesFloatMaskAndCaches) {
  JsonObjectGrammar grammar = make_grammar();
  JsonObjectGrammarState state = grammar.initial_state();
  ASSERT_TRUE(state.accept_piece("{\"a\":"));

  const std::vector<uint32_t> bitmask = grammar.allowed_token_bitmask(state);
  ASSERT_EQ(bitmask.size(), grammar.bitmask_num_words());
  EXPECT_EQ(grammar.allowed_token_bitmask(state), bitmask);

  const torch::Tensor float_mask = grammar.build_filter_mask(state);
  const torch::Tensor packed = grammar.build_filter_bitmask(state);
  ASSERT_EQ(
      packed.sizes(),
      torch::IntArrayRef({static_cast<int64_t>(grammar.bitmask_num_words())}));

  torch::Tensor logits =
      torch::zeros({1, static_cast<int64_t>(grammar.vocab_size())});
  apply_token_bitmask_inplace(logits, packed.unsqueeze(0));
  for (int64_t token_id = 0;
       token_id < static_cast<int64_t>(grammar.vocab_size());
       ++token_id) {
    EXPECT_NEAR(logits.index({0, token_id}).item<float>(),
                float_mask.index({token_id}).item<float>(),
                1.0e-3F);
  }
}

TEST(JsonObjectGrammarTest, BuildsMixedBatchBitmask) {
  JsonObjectGrammar grammar = make_grammar();
  std::vector<JsonObjectGrammarState> states = {grammar.initial_state(),
                                                JsonObjectGrammarState()};

  torch::Tensor bitmask = build_json_object_filter_bitmask(states);
  ASSERT_EQ(bitmask.sizes(),
            torch::IntArrayRef(
                {2, static_cast<int64_t>(grammar.bitmask_num_words())}));
  // Constrained row allows '{'.
  EXPECT_NE(bitmask.index({0, 0}).item<int32_t>() & 0x1, 0);
  // Unconstrained row is all-ones.
  EXPECT_EQ(bitmask.index({1, 0}).item<int32_t>(), -1);
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

TEST(JsonObjectGrammarTest, BuildsMaskWithMixedReasoningDefinitions) {
  const std::vector<std::string> token_pieces = {
      "{", "}", "reasoning", "<think>", "</think>"};
  JsonObjectGrammar json_grammar(token_pieces, /*stop_token_ids=*/{1});
  JsonObjectGrammar reasoning_grammar(token_pieces,
                                      /*stop_token_ids=*/{1},
                                      /*reasoning_end_token_ids=*/{3, 4});
  std::vector<JsonObjectGrammarState> states = {
      json_grammar.initial_state(),
      reasoning_grammar.initial_state(/*reasoning_phase=*/true),
      JsonObjectGrammarState(),
      reasoning_grammar.initial_state(/*reasoning_phase=*/true),
      json_grammar.initial_state()};

  torch::Tensor mask = build_json_object_filter_mask(states);

  ASSERT_EQ(mask.sizes(), torch::IntArrayRef({5, 5}));
  EXPECT_EQ(mask.index({0, 0}).item<float>(), 0.0F);
  EXPECT_LT(mask.index({0, 2}).item<float>(), -1.0F);
  EXPECT_EQ(mask.index({1, 2}).item<float>(), 0.0F);
  EXPECT_LT(mask.index({1, 1}).item<float>(), -1.0F);
  EXPECT_EQ(mask.index({2, 1}).item<float>(), 0.0F);
  EXPECT_EQ(mask.index({3, 2}).item<float>(), 0.0F);
  EXPECT_LT(mask.index({4, 2}).item<float>(), -1.0F);

  const torch::Tensor bitmask = build_json_object_filter_bitmask(states);
  ASSERT_EQ(bitmask.sizes(), torch::IntArrayRef({5, 1}));
  torch::Tensor masked_logits = torch::zeros({5, 5});
  apply_token_bitmask_inplace(masked_logits, bitmask);
  EXPECT_TRUE(torch::equal(masked_logits, mask));
}

TEST(JsonObjectGrammarTest, ReusesMaskForTransitionEquivalentStates) {
  JsonObjectGrammar grammar = make_grammar();
  JsonObjectGrammarState first = grammar.initial_state();
  JsonObjectGrammarState second = grammar.initial_state();
  ASSERT_TRUE(first.accept_token(/*open_object=*/0));
  ASSERT_TRUE(first.accept_token(/*quote=*/2));
  ASSERT_TRUE(first.accept_token(/*key_a=*/3));
  ASSERT_TRUE(second.accept_token(/*open_object=*/0));
  ASSERT_TRUE(second.accept_token(/*quote=*/2));
  ASSERT_TRUE(second.accept_token(/*key_b=*/4));
  ASSERT_NE(first.snapshot().token_ids, second.snapshot().token_ids);

  const double hits_before =
      COUNTER_json_object_mask_cache_hits_total.get_value();
  const torch::Tensor first_mask = grammar.build_filter_mask(first);
  const torch::Tensor second_mask = grammar.build_filter_mask(second);

  EXPECT_TRUE(torch::equal(first_mask, second_mask));
  EXPECT_GT(COUNTER_json_object_mask_cache_hits_total.get_value(), hits_before);
}

TEST(JsonObjectGrammarTest, ReusesMaskAfterDifferentCompletedLiterals) {
  JsonObjectGrammar grammar({"{", "}", "true", "false", "stop"},
                            /*stop_token_ids=*/{4});
  JsonObjectGrammarState true_value = grammar.initial_state();
  JsonObjectGrammarState false_value = grammar.initial_state();
  ASSERT_TRUE(true_value.accept_piece("{\"a\":true}"));
  ASSERT_TRUE(false_value.accept_piece("{\"a\":false}"));

  const double hits_before =
      COUNTER_json_object_mask_cache_hits_total.get_value();
  const torch::Tensor true_mask = grammar.build_filter_mask(true_value);
  const torch::Tensor false_mask = grammar.build_filter_mask(false_value);

  EXPECT_TRUE(torch::equal(true_mask, false_mask));
  EXPECT_GT(COUNTER_json_object_mask_cache_hits_total.get_value(), hits_before);
}

TEST(JsonObjectGrammarTest, DoesNotAliasDifferentTransitionStates) {
  JsonObjectGrammar grammar = make_grammar();
  JsonObjectGrammarState initial = grammar.initial_state();
  JsonObjectGrammarState in_object = grammar.initial_state();
  ASSERT_TRUE(in_object.accept_piece("{"));

  const torch::Tensor initial_mask = grammar.build_filter_mask(initial);
  const torch::Tensor object_mask = grammar.build_filter_mask(in_object);
  const torch::Tensor cached_initial_mask = grammar.build_filter_mask(initial);

  EXPECT_FALSE(torch::equal(initial_mask, object_mask));
  EXPECT_TRUE(torch::equal(initial_mask, cached_initial_mask));
}

TEST(JsonObjectGrammarTest, BuildsCachedMasksConcurrently) {
  JsonObjectGrammar grammar = make_grammar();
  const JsonObjectGrammarState state = grammar.initial_state();
  constexpr size_t kThreadCount = 8;
  std::vector<torch::Tensor> masks(kThreadCount);
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);
  for (size_t thread_idx = 0; thread_idx < kThreadCount; ++thread_idx) {
    threads.emplace_back([&grammar, &masks, &state, thread_idx]() {
      masks[thread_idx] = grammar.build_filter_mask(state);
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }

  for (size_t mask_idx = 1; mask_idx < masks.size(); ++mask_idx) {
    EXPECT_TRUE(torch::equal(masks[0], masks[mask_idx]));
  }
}

TEST(JsonObjectGrammarTest, RemainsCorrectAfterMaskCacheEviction) {
  JsonObjectGrammar grammar({"{", "\"a\"", ":", "[", "]", "}", "0"});
  JsonObjectGrammarState initial = grammar.initial_state();
  const torch::Tensor initial_mask = grammar.build_filter_mask(initial);
  EXPECT_EQ(initial_mask.index({0}).item<float>(), 0.0F);
  EXPECT_LT(initial_mask.index({3}).item<float>(), -1.0F);

  JsonObjectGrammarState nested = grammar.initial_state();
  ASSERT_TRUE(nested.accept_token(/*open_object=*/0));
  ASSERT_TRUE(nested.accept_token(/*key=*/1));
  ASSERT_TRUE(nested.accept_token(/*colon=*/2));
  JsonObjectGrammarState oldest_nested;
  for (size_t depth = 0; depth < 63; ++depth) {
    ASSERT_TRUE(nested.accept_token(/*open_array=*/3));
    if (depth == 0) {
      oldest_nested = nested;
    }
    const torch::Tensor mask = grammar.build_filter_mask(nested);
    EXPECT_EQ(mask.size(0), static_cast<int64_t>(grammar.vocab_size()));
  }

  // Promote the initial state before inserting the 65th distinct state.
  const double hits_before_touch =
      COUNTER_json_object_mask_cache_hits_total.get_value();
  grammar.build_filter_mask(initial);
  EXPECT_GT(COUNTER_json_object_mask_cache_hits_total.get_value(),
            hits_before_touch);

  ASSERT_TRUE(nested.accept_token(/*open_array=*/3));
  grammar.build_filter_mask(nested);

  const double misses_before_hot_lookup =
      COUNTER_json_object_mask_cache_misses_total.get_value();
  grammar.build_filter_mask(initial);
  EXPECT_EQ(COUNTER_json_object_mask_cache_misses_total.get_value(),
            misses_before_hot_lookup);

  grammar.build_filter_mask(oldest_nested);
  EXPECT_GT(COUNTER_json_object_mask_cache_misses_total.get_value(),
            misses_before_hot_lookup);
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

TEST(JsonObjectGrammarTest, UsesStableDecodeTokenPiecesForGrammar) {
  LossySingleTokenTokenizer tokenizer;
  std::string error;
  std::shared_ptr<const JsonObjectGrammar> grammar =
      JsonObjectGrammar::create_from_tokenizer(tokenizer,
                                               /*eos_token_id=*/6,
                                               /*stop_token_ids=*/{},
                                               /*model_vocab_size=*/7,
                                               /*reasoning_enabled=*/false,
                                               &error);
  ASSERT_NE(grammar, nullptr) << error;

  JsonObjectGrammarState state = grammar->initial_state();
  ASSERT_TRUE(state.accept_token(0));
  ASSERT_TRUE(state.accept_token(1));
  ASSERT_TRUE(state.accept_token(2));

  EXPECT_TRUE(state.can_accept_token(3));
  EXPECT_FALSE(state.can_accept_piece("x"));
}

TEST(JsonObjectGrammarTest, TrialAcceptIgnoresCommittedTokenHistorySize) {
  JsonObjectGrammar grammar = make_grammar();
  JsonObjectGrammarState state = grammar.initial_state();
  ASSERT_TRUE(state.accept_piece("{\"a\":"));

  // Grow committed history with digit tokens; can_accept must stay equivalent
  // without depending on O(generated) vector copies.
  constexpr int32_t kDigitOneTokenId = 12;
  for (int32_t i = 0; i < 64; ++i) {
    ASSERT_TRUE(state.accept_token(kDigitOneTokenId));
  }
  EXPECT_TRUE(state.can_accept_token(kDigitOneTokenId));
  EXPECT_TRUE(state.can_accept_piece("}"));
  EXPECT_FALSE(state.can_accept_piece("x"));
}

}  // namespace
}  // namespace xllm

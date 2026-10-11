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

#include "core/framework/batch/sampling_input_builder.h"

#include <gtest/gtest.h>

#include <cmath>

#include "core/framework/request/sequence.h"
#include "core/framework/request/stopping_checker.h"

namespace xllm {
namespace {

Sequence make_sequence(RequestSamplingParam& sampling,
                       StoppingChecker& stopping,
                       size_t index,
                       const std::vector<int32_t>& prompt) {
  SequenceParams params;
  params.seq_capacity = prompt.size() + 16;
  params.sampling_param = &sampling;
  params.stopping_checker = &stopping;
  IncrementalDecoder decoder("",
                             prompt.size(),
                             /*echo=*/false,
                             /*skip_special_tokens=*/true);
  return Sequence(index, prompt, {}, {}, std::move(decoder), params);
}

SamplingParameters build_row(const RequestSamplingParam& params,
                             const Sequence& sequence) {
  SamplingInputBuilder builder;
  builder.append(&params,
                 /*token_index=*/0,
                 &sequence.token_to_count_map(),
                 /*excluded_counts=*/nullptr,
                 /*sample=*/true,
                 &sequence);
  return builder.build();
}

TEST(SamplingInputBuilderTest, MasksStopsUntilMinimumAndBadWordCompletions) {
  RequestSamplingParam params;
  params.vocab_size = 8;
  params.min_tokens = 2;
  params.all_stop_token_ids = {7};
  params.bad_words_token_ids = {{2, 3}, {4}};
  StoppingChecker stopping;
  Sequence sequence = make_sequence(params, stopping, 0, {1, 2});
  auto first = build_row(params, sequence);
  EXPECT_EQ(first.logits_bias[0][3].item<float>(), 0.0F);
  EXPECT_TRUE(std::isinf(first.logits_bias[0][4].item<float>()));
  EXPECT_TRUE(std::isinf(first.logits_bias[0][7].item<float>()));
  sequence.kv_state().set_kv_cache_tokens_num(sequence.num_prompt_tokens());
  sequence.append_token(Token(2));
  auto second = build_row(params, sequence);
  EXPECT_TRUE(std::isinf(second.logits_bias[0][3].item<float>()));
  EXPECT_TRUE(std::isinf(second.logits_bias[0][7].item<float>()));
  sequence.append_token(Token(5));
  auto third = build_row(params, sequence);
  EXPECT_EQ(third.logits_bias[0][3].item<float>(), 0.0F);
  EXPECT_EQ(third.logits_bias[0][7].item<float>(), 0.0F);
}

TEST(SamplingInputBuilderTest, MergesConstraintsAndIndependentSeedOffsets) {
  RequestSamplingParam constrained;
  constrained.vocab_size = 8;
  constrained.seed = 42;
  constrained.allowed_token_ids = std::vector<int32_t>{2, 3};
  constrained.logit_bias = {{1, 100.0F}, {2, 5.0F}};
  StoppingChecker stopping;
  Sequence sequence = make_sequence(constrained, stopping, 2, {1});
  sequence.kv_state().set_kv_cache_tokens_num(sequence.num_prompt_tokens());
  sequence.append_token(Token(3));
  RequestSamplingParam plain;
  SamplingInputBuilder first;
  first.append(&plain, /*token_index=*/0);
  SamplingInputBuilder second;
  second.append(&constrained,
                /*token_index=*/0,
                /*counts=*/nullptr,
                /*excluded_counts=*/nullptr,
                /*sample=*/true,
                &sequence);
  first.merge(std::move(second), /*token_offset=*/1);
  auto result = first.build();
  EXPECT_EQ(result.logits_bias[0].sum().item<float>(), 0.0F);
  EXPECT_TRUE(std::isinf(result.logits_bias[1][1].item<float>()));
  EXPECT_EQ(result.logits_bias[1][2].item<float>(), 5.0F);
  EXPECT_EQ(result.seeds[0].item<int64_t>(), -1);
  EXPECT_EQ(result.seeds[1].item<int64_t>(), 42);
  EXPECT_EQ(result.seed_offsets[1].item<int64_t>(), (int64_t{2} << 32) + 1);
}

TEST(SamplingInputBuilderTest, PromptAffectsRepetitionButNotOutputCounts) {
  RequestSamplingParam params;
  params.repetition_penalty = 1.2F;
  params.frequency_penalty = 0.5F;
  StoppingChecker stopping;
  Sequence sequence = make_sequence(params, stopping, 0, {1, 1, 2});
  sequence.kv_state().set_kv_cache_tokens_num(sequence.num_prompt_tokens());
  sequence.append_token(Token(2));
  sequence.append_token(Token(3));
  auto result = build_row(params, sequence);
  ASSERT_EQ(result.unique_token_ids_lens[0].item<int32_t>(), 3);
  for (int64_t column = 0; column < 3; ++column) {
    const int64_t token = result.unique_token_ids[0][column].item<int64_t>();
    const int32_t count = result.unique_token_counts[0][column].item<int32_t>();
    EXPECT_EQ(count, token == 1 ? 0 : 1);
  }
}

}  // namespace
}  // namespace xllm

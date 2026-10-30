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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "framework/request/incremental_decoder.h"
#include "framework/request/request.h"
#include "framework/request/request_state.h"
#include "framework/request/sequence.h"

namespace xllm {
namespace {

Sequence make_decode_ready_sequence(bool enable_schedule_overlap) {
  static RequestSamplingParam sampling_param;
  static StoppingChecker stopping_checker;

  SequenceParams params;
  params.seq_capacity = 8;
  params.echo = false;
  params.skip_special_tokens = true;
  params.streaming = false;
  params.enable_schedule_overlap = enable_schedule_overlap;
  params.rec_type = RecType::kNone;
  params.bos_token_id = 0;
  params.request_id = "generated_tokens_req";
  params.sampling_param = &sampling_param;
  params.stopping_checker = &stopping_checker;

  std::vector<int32_t> prompt_token_ids = {1, 2, 3};
  IncrementalDecoder decoder(
      /*prompt=*/"prompt",
      /*num_prompt_tokens=*/prompt_token_ids.size(),
      /*echo=*/params.echo,
      /*skip_special_tokens=*/params.skip_special_tokens);
  Sequence sequence(/*index=*/0,
                    prompt_token_ids,
                    /*input_embedding=*/torch::Tensor(),
                    /*mm_data=*/MMData(),
                    decoder,
                    params);
  // Move the sequence out of the prefill stage so append_token / the decode
  // path treats new tokens as generated decode tokens.
  sequence.kv_state().set_kv_cache_tokens_num(sequence.num_prompt_tokens());
  return sequence;
}

}  // namespace

TEST(SequenceGeneratedTokensTest, CountsRealAppendedTokens) {
  Sequence sequence = make_decode_ready_sequence(
      /*enable_schedule_overlap=*/false);
  EXPECT_EQ(sequence.generated_tokens_since_latency(), 0u);

  sequence.append_token(Token(10));
  sequence.append_token(Token(11));

  EXPECT_EQ(sequence.generated_tokens_since_latency(), 2u);
}

TEST(SequenceGeneratedTokensTest, TbtResetsGeneratedTokenCount) {
  Sequence sequence = make_decode_ready_sequence(
      /*enable_schedule_overlap=*/false);
  sequence.append_token(Token(10));
  sequence.append_token(Token(11));
  ASSERT_EQ(sequence.generated_tokens_since_latency(), 2u);

  sequence.tbt(absl::Now());
  EXPECT_EQ(sequence.generated_tokens_since_latency(), 0u);

  sequence.append_token(Token(12));
  EXPECT_EQ(sequence.generated_tokens_since_latency(), 1u);
}

TEST(SequenceGeneratedTokensTest, ForkPreservesPendingLatencyObservation) {
  for (bool enable_schedule_overlap : {false, true}) {
    SCOPED_TRACE(enable_schedule_overlap);
    Sequence sequence = make_decode_ready_sequence(enable_schedule_overlap);
    const absl::Time start = absl::FromUnixSeconds(100);
    sequence.tbt_microseconds(start);
    if (enable_schedule_overlap) {
      sequence.append_token(Token(-1));
      sequence.update_last_step_token(Token(10), /*token_offset=*/0);
    } else {
      sequence.append_token(Token(10));
    }
    ASSERT_TRUE(sequence.is_first_token());
    ASSERT_EQ(sequence.generated_tokens_since_latency(), 1u);

    std::unique_ptr<Sequence> fork = sequence.fork(/*index=*/1);
    EXPECT_TRUE(fork->is_first_token());
    EXPECT_EQ(fork->generated_tokens_since_latency(), 1u);
    EXPECT_EQ(fork->tbt_microseconds(start + absl::Seconds(7)), 7000000);
    EXPECT_EQ(fork->generated_tokens_since_latency(), 0u);
    EXPECT_EQ(sequence.generated_tokens_since_latency(), 1u);
    EXPECT_EQ(sequence.tbt_microseconds(start + absl::Seconds(9)), 9000000);
  }
}

TEST(SequenceGeneratedTokensTest, IgnoresOverlapFakeTokens) {
  Sequence sequence = make_decode_ready_sequence(
      /*enable_schedule_overlap=*/true);
  // Under schedule overlap a placeholder token id (< 0) is appended and the
  // real token is committed later via update_last_step_token. Fake tokens must
  // not inflate the committed-token count used for amortized TPOT.
  sequence.append_token(Token(-1));
  sequence.append_token(Token(-1));

  EXPECT_EQ(sequence.generated_tokens_since_latency(), 0u);
}

TEST(SequenceGeneratedTokensTest, CountsOverlapCommittedTokens) {
  Sequence sequence = make_decode_ready_sequence(
      /*enable_schedule_overlap=*/true);
  EXPECT_EQ(sequence.generated_tokens_since_latency(), 0u);

  sequence.update_last_step_token(Token(10), /*token_offset=*/0);
  sequence.update_last_step_token(Token(11), /*token_offset=*/0);

  EXPECT_EQ(sequence.generated_tokens_since_latency(), 2u);
}

class RequestGeneratedTokenUsageTest
    : public ::testing::TestWithParam<std::tuple<bool, int32_t, int32_t>> {};

TEST_P(RequestGeneratedTokenUsageTest, CountsOnlyCommittedTokens) {
  const auto [enable_schedule_overlap, generated_tokens, placeholder_tokens] =
      GetParam();
  RequestState state(
      "prompt",
      std::vector<int32_t>{1, 2, 3},
      RequestSamplingParam{},
      SchedulerParam{},
      StoppingChecker{},
      /*seq_capacity=*/16,
      /*n=*/1,
      /*best_of=*/1,
      /*logprobs=*/false,
      /*stream=*/false,
      /*echo=*/false,
      /*skip_special_tokens=*/true,
      enable_schedule_overlap,
      [](const RequestOutput&) { return true; },
      OutputsFunc{});
  Request request("generated-token-usage", "", "", std::move(state));
  auto& sequence = *request.sequences().front();
  sequence.kv_state().set_kv_cache_tokens_num(sequence.num_prompt_tokens());
  for (int32_t token_index = 0; token_index < generated_tokens; ++token_index) {
    sequence.append_token(Token(10 + token_index));
  }
  for (int32_t token_index = 0; token_index < placeholder_tokens;
       ++token_index) {
    sequence.append_token(Token(-1));
  }

  Tokenizer tokenizer;
  const RequestOutput output = request.generate_output(tokenizer);
  ASSERT_TRUE(output.usage.has_value());
  EXPECT_EQ(output.usage->num_prompt_tokens, 3);
  EXPECT_EQ(output.usage->num_generated_tokens, generated_tokens);
  EXPECT_EQ(output.usage->num_total_tokens, 3 + generated_tokens);
}

INSTANTIATE_TEST_SUITE_P(OverlapPlaceholders,
                         RequestGeneratedTokenUsageTest,
                         ::testing::Values(std::make_tuple(false, 0, 0),
                                           std::make_tuple(false, 2, 0),
                                           std::make_tuple(true, 0, 0),
                                           std::make_tuple(true, 0, 1),
                                           std::make_tuple(true, 0, 2),
                                           std::make_tuple(true, 2, 0),
                                           std::make_tuple(true, 2, 1),
                                           std::make_tuple(true, 2, 2)));

}  // namespace xllm

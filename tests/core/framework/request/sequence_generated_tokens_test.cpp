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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "core/framework/request/request.h"
#include "framework/request/incremental_decoder.h"
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

class UsageTokenizer final : public Tokenizer {
 public:
  std::string decode(const Slice<int32_t>& ids,
                     bool /*skip_special_tokens*/) const override {
    return std::string(ids.size(), 'A');
  }
};

std::unique_ptr<Request> make_usage_request(size_t max_tokens, bool overlap) {
  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(max_tokens);
  const std::vector<int32_t> prompt_tokens = {1};
  RequestState state(
      "A",
      prompt_tokens,
      sampling_param,
      SchedulerParam{},
      stopping_checker,
      /*seq_capacity=*/max_tokens + 8,
      /*n=*/1,
      /*best_of=*/1,
      /*logprobs=*/false,
      /*stream=*/false,
      /*echo=*/false,
      /*skip_special_tokens=*/true,
      overlap,
      [](const RequestOutput&) { return true; },
      OutputsFunc{});
  auto request = std::make_unique<Request>("usage-test", "", "", state);
  request->sequences().front()->kv_state().set_kv_cache_tokens_num(1);
  return request;
}

class RequestCommittedUsageTest
    : public ::testing::TestWithParam<std::tuple<size_t, int32_t>> {};

TEST_P(RequestCommittedUsageTest, CountsCommittedTokensAtLengthBoundary) {
  const auto [max_tokens, pending_placeholders] = GetParam();
  auto request = make_usage_request(max_tokens, /*overlap=*/true);
  Sequence& sequence = *request->sequences().front();
  for (size_t i = 0; i < max_tokens; ++i) {
    sequence.append_token(Token(-1));
    sequence.update_last_step_token(Token(65));
  }
  for (int32_t i = 0; i < pending_placeholders; ++i) {
    sequence.append_token(Token(-1));
  }
  ASSERT_TRUE(sequence.finished());
  EXPECT_EQ(sequence.finish_reason(), FinishReason::LENGTH);
  UsageTokenizer tokenizer;
  const RequestOutput output = request->generate_output(tokenizer);
  ASSERT_TRUE(output.usage.has_value());
  EXPECT_EQ(output.usage->num_generated_tokens, max_tokens);
  EXPECT_EQ(output.usage->num_total_tokens, max_tokens + 1);
  ASSERT_EQ(output.outputs.size(), 1);
  EXPECT_EQ(output.outputs.front().token_ids.size(), max_tokens);
}

INSTANTIATE_TEST_SUITE_P(OverlapBoundaries,
                         RequestCommittedUsageTest,
                         ::testing::Combine(::testing::Values(size_t{1},
                                                              size_t{263}),
                                            ::testing::Values(0, 1, 2)));

TEST(RequestCommittedUsageTest, CancelledPlaceholdersDoNotCountAsOutput) {
  auto request = make_usage_request(/*max_tokens=*/1, /*overlap=*/true);
  request->sequences().front()->append_token(Token(-1));
  request->sequences().front()->append_token(Token(-1));
  request->set_cancel();
  UsageTokenizer tokenizer;
  const RequestOutput output = request->generate_output(tokenizer);
  ASSERT_TRUE(output.usage.has_value());
  EXPECT_EQ(output.usage->num_generated_tokens, 0);
}

TEST(RequestCommittedUsageTest, NonOverlapCountsEveryRealToken) {
  auto request = make_usage_request(/*max_tokens=*/1, /*overlap=*/false);
  request->sequences().front()->append_token(Token(65));
  UsageTokenizer tokenizer;
  const RequestOutput output = request->generate_output(tokenizer);
  ASSERT_TRUE(output.usage.has_value());
  EXPECT_EQ(output.usage->num_generated_tokens, 1);
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

}  // namespace xllm

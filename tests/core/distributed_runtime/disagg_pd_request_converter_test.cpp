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

#include "core/distributed_runtime/disagg_pd_request_converter.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <variant>
#include <vector>

namespace xllm {
namespace {

TEST(DisaggPDRequestConverterTest, PreservesSamplingValuesAndTokenOrder) {
  proto::DisaggRequest req;
  req.set_frequency_penalty(0.25f);
  req.set_presence_penalty(0.5f);
  req.set_repetition_penalty(1.25f);
  req.set_temperature(0.75f);
  req.set_top_p(0.875f);
  req.set_min_p(0.125f);
  req.set_seed(-23);
  req.set_min_tokens(3);
  req.set_vocab_size(1000);
  req.set_top_k(16);
  req.set_logprobs(true);
  req.set_top_logprobs(4);
  req.set_is_embeddings(true);
  req.set_json_object(true);
  (*req.mutable_logit_bias())[9] = -1.5f;
  (*req.mutable_logit_bias())[3] = 2.5f;
  req.add_allowed_token_ids(9);
  req.add_allowed_token_ids(3);
  req.add_allowed_token_ids(9);
  req.add_all_stop_token_ids(7);
  req.add_all_stop_token_ids(2);
  auto* first_bad_word = req.add_bad_words_token_ids();
  first_bad_word->add_seq_tokens(8);
  first_bad_word->add_seq_tokens(5);
  req.add_bad_words_token_ids()->add_seq_tokens(6);

  const RequestSamplingParam param = make_disagg_sampling_param(req);

  EXPECT_FLOAT_EQ(param.frequency_penalty, 0.25f);
  EXPECT_FLOAT_EQ(param.presence_penalty, 0.5f);
  EXPECT_FLOAT_EQ(param.repetition_penalty, 1.25f);
  EXPECT_FLOAT_EQ(param.temperature, 0.75f);
  EXPECT_FLOAT_EQ(param.top_p, 0.875f);
  EXPECT_FLOAT_EQ(param.min_p, 0.125f);
  ASSERT_TRUE(param.seed.has_value());
  EXPECT_EQ(param.seed.value(), -23);
  EXPECT_EQ(param.min_tokens, 3u);
  EXPECT_EQ(param.vocab_size, 1000);
  EXPECT_EQ(param.top_k, 16);
  EXPECT_TRUE(param.logprobs);
  EXPECT_EQ(param.top_logprobs, 4);
  EXPECT_TRUE(param.is_embeddings);
  EXPECT_TRUE(param.json_object);
  ASSERT_EQ(param.logit_bias.size(), 2u);
  EXPECT_FLOAT_EQ(param.logit_bias.at(9), -1.5f);
  EXPECT_FLOAT_EQ(param.logit_bias.at(3), 2.5f);
  ASSERT_TRUE(param.allowed_token_ids.has_value());
  EXPECT_EQ(param.allowed_token_ids.value(), (std::vector<int32_t>{9, 3, 9}));
  EXPECT_EQ(param.all_stop_token_ids, (std::vector<int32_t>{7, 2}));
  EXPECT_EQ(param.bad_words_token_ids,
            (std::vector<std::vector<int32_t>>{{8, 5}, {6}}));
  EXPECT_EQ(param.beam_width, 0);
  EXPECT_EQ(param.num_return_sequences, 0);
}

TEST(DisaggPDRequestConverterTest, KeepsWireDefaultsAndOptionalSeedPresence) {
  proto::DisaggRequest req;
  const RequestSamplingParam absent = make_disagg_sampling_param(req);

  EXPECT_FALSE(absent.seed.has_value());
  EXPECT_FALSE(absent.allowed_token_ids.has_value());
  EXPECT_FLOAT_EQ(absent.frequency_penalty, 0.0f);
  EXPECT_FLOAT_EQ(absent.presence_penalty, 0.0f);
  EXPECT_FLOAT_EQ(absent.repetition_penalty, 0.0f);
  EXPECT_FLOAT_EQ(absent.temperature, 0.0f);
  EXPECT_FLOAT_EQ(absent.top_p, 0.0f);
  EXPECT_FLOAT_EQ(absent.min_p, 0.0f);
  EXPECT_EQ(absent.min_tokens, 0u);
  EXPECT_EQ(absent.vocab_size, 0);
  EXPECT_EQ(absent.top_k, 0);
  EXPECT_FALSE(absent.logprobs);
  EXPECT_EQ(absent.top_logprobs, 0);
  EXPECT_FALSE(absent.is_embeddings);
  EXPECT_FALSE(absent.json_object);
  EXPECT_TRUE(absent.logit_bias.empty());
  EXPECT_TRUE(absent.all_stop_token_ids.empty());
  EXPECT_TRUE(absent.bad_words_token_ids.empty());

  req.set_seed(0);
  const RequestSamplingParam explicit_zero = make_disagg_sampling_param(req);
  ASSERT_TRUE(explicit_zero.seed.has_value());
  EXPECT_EQ(explicit_zero.seed.value(), 0);
}

TEST(DisaggPDRequestConverterTest, CopiesOnlineSchedulingFields) {
  proto::DisaggRequest req;
  req.set_priority(proto::Priority::HIGH);
  req.set_ttft_slo_ms(101);
  req.set_tpot_slo_ms(102);
  req.set_ttlt_slo_ms(103);
  req.set_tpot_priority_weight(4);
  req.set_ttft_priority_weight(5);
  req.set_ttlt_priority_weight(6);
  req.set_priority_weight(7);

  const SchedulerParam param = make_disagg_scheduler_param(req);

  EXPECT_FALSE(param.offline);
  EXPECT_EQ(param.priority, RequestPriority::HIGH);
  EXPECT_EQ(param.ttft_slo_ms, 101);
  EXPECT_EQ(param.tpot_slo_ms, 102);
  EXPECT_EQ(param.ttlt_slo_ms, 103);
  EXPECT_EQ(param.tpot_priority_weight, 4);
  EXPECT_EQ(param.ttft_priority_weight, 5);
  EXPECT_EQ(param.ttlt_priority_weight, 6);
  EXPECT_EQ(param.priority_weight, 7);
}

TEST(DisaggPDRequestConverterTest, KeepsOfflineSchedulingDefaults) {
  proto::DisaggRequest req;
  req.set_offline(true);
  req.set_priority(proto::Priority::LOW);
  req.set_ttft_slo_ms(101);
  req.set_tpot_slo_ms(102);
  req.set_ttlt_slo_ms(103);
  req.set_tpot_priority_weight(4);
  req.set_ttft_priority_weight(5);
  req.set_ttlt_priority_weight(6);
  req.set_priority_weight(7);

  const SchedulerParam param = make_disagg_scheduler_param(req);

  EXPECT_TRUE(param.offline);
  EXPECT_EQ(param.priority, RequestPriority::LOW);
  EXPECT_EQ(param.ttft_slo_ms, std::numeric_limits<int32_t>::max());
  EXPECT_EQ(param.tpot_slo_ms, std::numeric_limits<int32_t>::max());
  EXPECT_EQ(param.ttlt_slo_ms, std::numeric_limits<int32_t>::max());
  EXPECT_EQ(param.tpot_priority_weight, 1);
  EXPECT_EQ(param.ttft_priority_weight, 1);
  EXPECT_EQ(param.ttlt_priority_weight, 1);
  EXPECT_EQ(param.priority_weight, 1);
}

TEST(DisaggPDRequestConverterTest, UsesWireDefaultsForAbsentOnlineScheduling) {
  const SchedulerParam param =
      make_disagg_scheduler_param(proto::DisaggRequest{});

  EXPECT_FALSE(param.offline);
  EXPECT_EQ(param.priority, RequestPriority::DEFAULT);
  EXPECT_EQ(param.ttft_slo_ms, 0);
  EXPECT_EQ(param.tpot_slo_ms, 0);
  EXPECT_EQ(param.ttlt_slo_ms, 0);
  EXPECT_EQ(param.tpot_priority_weight, 0);
  EXPECT_EQ(param.ttft_priority_weight, 0);
  EXPECT_EQ(param.ttlt_priority_weight, 0);
  EXPECT_EQ(param.priority_weight, 0);
}

TEST(DisaggPDRequestConverterTest, PreservesStopSequenceOrderAndMetadata) {
  proto::DisaggRequest req;
  req.set_max_tokens(2);
  req.set_max_context_len(20);
  req.set_eos_token_id(2);
  req.set_ignore_eos(true);
  req.add_stop_token_ids(3);
  req.add_stop_token_ids(3);
  auto* first_stop = req.add_stop_sequences();
  first_stop->add_seq_tokens(4);
  first_stop->add_seq_tokens(5);
  first_stop->set_stop_string("first");
  auto* second_stop = req.add_stop_sequences();
  second_stop->add_seq_tokens(5);

  StoppingChecker checker = make_disagg_stopping_checker(req);

  EXPECT_EQ(checker.get_max_generated_tokens(), 2u);
  EXPECT_EQ(checker.get_max_context_len(), 20u);
  EXPECT_EQ(checker.get_eos_token(), 2);
  EXPECT_TRUE(checker.get_ignore_eos());
  EXPECT_EQ(checker.get_stop_tokens().size(), 1u);
  EXPECT_EQ(checker.get_stop_sequences(),
            (std::vector<std::vector<int32_t>>{{4, 5}, {5}}));
  EXPECT_EQ(checker.get_stop_strings(),
            (std::vector<std::string>{"first", ""}));
  EXPECT_EQ(checker.check(std::vector<int32_t>{1, 2},
                          /*num_prompt_tokens=*/1),
            FinishReason::NONE);

  size_t matched_stop_token_count = 0;
  StopReason stop_reason;
  EXPECT_EQ(checker.check(std::vector<int32_t>{1, 4, 5, -1},
                          /*num_prompt_tokens=*/1,
                          &matched_stop_token_count,
                          &stop_reason),
            FinishReason::STOP);
  EXPECT_EQ(matched_stop_token_count, 2u);
  ASSERT_TRUE(std::holds_alternative<std::string>(stop_reason));
  EXPECT_EQ(std::get<std::string>(stop_reason), "first");

  EXPECT_EQ(checker.check(std::vector<int32_t>{1, 4, 5},
                          /*num_prompt_tokens=*/2,
                          &matched_stop_token_count,
                          &stop_reason),
            FinishReason::STOP);
  EXPECT_EQ(matched_stop_token_count, 1u);

  EXPECT_EQ(checker.check(std::vector<int32_t>{1, 3},
                          /*num_prompt_tokens=*/1,
                          &matched_stop_token_count,
                          &stop_reason),
            FinishReason::STOP);
  ASSERT_TRUE(std::holds_alternative<int32_t>(stop_reason));
  EXPECT_EQ(std::get<int32_t>(stop_reason), 3);
}

TEST(DisaggPDRequestConverterTest, EnforcesMinTokensBeforeStopCriteria) {
  proto::DisaggRequest req;
  req.set_max_tokens(10);
  req.set_eos_token_id(2);
  req.set_min_tokens(2);
  req.add_stop_token_ids(3);
  req.add_stop_sequences()->add_seq_tokens(4);
  const StoppingChecker checker = make_disagg_stopping_checker(req);

  for (int32_t token : {2, 3, 4}) {
    EXPECT_EQ(checker.check(std::vector<int32_t>{1, token},
                            /*num_prompt_tokens=*/1),
              FinishReason::NONE);
    EXPECT_EQ(checker.check(std::vector<int32_t>{1, 5, token},
                            /*num_prompt_tokens=*/1),
              FinishReason::STOP);
  }
  EXPECT_EQ(checker.check(std::vector<int32_t>{1, 5, 6},
                          /*num_prompt_tokens=*/1),
            FinishReason::NONE);
}

TEST(DisaggPDRequestConverterTest, EnforcesGenerationAndContextLimits) {
  proto::DisaggRequest req;
  req.set_max_tokens(2);
  req.set_max_context_len(5);
  req.set_eos_token_id(2);
  const StoppingChecker generation_checker = make_disagg_stopping_checker(req);
  EXPECT_EQ(generation_checker.check(std::vector<int32_t>{1, 5, 6},
                                     /*num_prompt_tokens=*/1),
            FinishReason::LENGTH);

  req.set_max_tokens(10);
  const StoppingChecker context_checker = make_disagg_stopping_checker(req);
  EXPECT_EQ(context_checker.check(std::vector<int32_t>{1, 5, 6, 7, 8},
                                  /*num_prompt_tokens=*/4),
            FinishReason::LENGTH);
}

}  // namespace
}  // namespace xllm

/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/framework/request/stopping_checker.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <unordered_set>
#include <vector>

namespace xllm {
namespace {

TEST(StoppingCheckerTest, IgnoreEosSkipsOnlyEosToken) {
  StoppingChecker checker(
      /*max_generated_tokens=*/10,
      /*max_context_len=*/0,
      /*eos_token=*/2,
      /*ignore_eos=*/true,
      /*stop_tokens=*/std::unordered_set<int32_t>{},
      /*stop_sequences=*/std::vector<std::vector<int32_t>>{});
  const std::vector<int32_t> token_ids = {1, 2};

  EXPECT_EQ(checker.check(token_ids, /*num_prompt_tokens=*/1),
            FinishReason::NONE);
}

TEST(StoppingCheckerTest, IgnoreEosHonorsExplicitStopTokens) {
  StoppingChecker checker(/*max_generated_tokens=*/10,
                          /*max_context_len=*/0,
                          /*eos_token=*/2,
                          /*ignore_eos=*/true,
                          /*stop_tokens=*/{2, 3},
                          /*stop_sequences=*/{});
  EXPECT_EQ(checker.check(std::vector<int32_t>{1, 2}, 1), FinishReason::STOP);
  EXPECT_EQ(checker.check(std::vector<int32_t>{1, 3}, 1), FinishReason::STOP);
}

TEST(StoppingCheckerTest, MinTokensDelaysStopStringsAndTokens) {
  StoppingChecker checker(/*max_generated_tokens=*/4,
                          /*max_context_len=*/0,
                          /*eos_token=*/2,
                          /*ignore_eos=*/false,
                          /*stop_tokens=*/{3},
                          /*stop_sequences=*/{{4}},
                          /*stop_strings=*/{"stop"},
                          /*min_generated_tokens=*/2);
  EXPECT_EQ(checker.check(std::vector<int32_t>{1, 2}, 1), FinishReason::NONE);
  EXPECT_EQ(checker.check(std::vector<int32_t>{1, 3}, 1), FinishReason::NONE);
  EXPECT_EQ(checker.check(std::vector<int32_t>{1, 4}, 1), FinishReason::NONE);
  EXPECT_EQ(checker.check(std::vector<int32_t>{1, 5, 4}, 1),
            FinishReason::STOP);
}

TEST(StoppingCheckerTest, StopTokensStopWhenEosNotIgnored) {
  StoppingChecker checker(
      /*max_generated_tokens=*/10,
      /*max_context_len=*/0,
      /*eos_token=*/163585,
      /*ignore_eos=*/false,
      /*stop_tokens=*/std::unordered_set<int32_t>{163585, 163586},
      /*stop_sequences=*/std::vector<std::vector<int32_t>>{});

  EXPECT_EQ(checker.check(std::vector<int32_t>{1, 163586},
                          /*num_prompt_tokens=*/1),
            FinishReason::STOP);
}

TEST(StoppingCheckerTest, IgnoreEosStillStopsOnStopSequence) {
  // stop sequences come from the request's `stop` field, independent of
  // ignore_eos, so they keep stopping generation.
  StoppingChecker checker(
      /*max_generated_tokens=*/10,
      /*max_context_len=*/0,
      /*eos_token=*/2,
      /*ignore_eos=*/true,
      /*stop_tokens=*/std::unordered_set<int32_t>{},
      /*stop_sequences=*/std::vector<std::vector<int32_t>>{{4, 5}});
  const std::vector<int32_t> token_ids = {1, 4, 5};

  EXPECT_EQ(checker.check(token_ids, /*num_prompt_tokens=*/1),
            FinishReason::STOP);
}

TEST(StoppingCheckerTest, StopSequenceIgnoresOverlapPlaceholder) {
  StoppingChecker checker(
      /*max_generated_tokens=*/2,
      /*max_context_len=*/0,
      /*eos_token=*/2,
      /*ignore_eos=*/false,
      /*stop_tokens=*/std::unordered_set<int32_t>{},
      /*stop_sequences=*/std::vector<std::vector<int32_t>>{{4, 5}});
  const std::vector<int32_t> token_ids = {1, 4, 5, -1};
  size_t matched_stop_token_count = 0;

  EXPECT_EQ(checker.check(token_ids,
                          /*num_prompt_tokens=*/1,
                          &matched_stop_token_count),
            FinishReason::STOP);
  EXPECT_EQ(matched_stop_token_count, 2);
}

}  // namespace
}  // namespace xllm

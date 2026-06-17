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

TEST(StoppingCheckerTest, IgnoreEosStillStopsOnStopToken) {
  StoppingChecker checker(
      /*max_generated_tokens=*/10,
      /*max_context_len=*/0,
      /*eos_token=*/2,
      /*ignore_eos=*/true,
      /*stop_tokens=*/std::unordered_set<int32_t>{7},
      /*stop_sequences=*/std::vector<std::vector<int32_t>>{});
  const std::vector<int32_t> token_ids = {1, 7};

  EXPECT_EQ(checker.check(token_ids, /*num_prompt_tokens=*/1),
            FinishReason::STOP);
}

TEST(StoppingCheckerTest, IgnoreEosStillStopsOnStopSequence) {
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

}  // namespace
}  // namespace xllm

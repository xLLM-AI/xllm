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

#include <vector>

#include "core/util/utils.h"

namespace xllm::util {

TEST(VocabValidationTest, AllTokensInRangeReturnsNullopt) {
  const std::vector<int> token_ids = {0, 1, 785, 999};
  EXPECT_FALSE(
      find_out_of_vocab_token(token_ids, /*vocab_size=*/1000).has_value());
}

TEST(VocabValidationTest, EmptyTokensReturnsNullopt) {
  const std::vector<int> token_ids = {};
  EXPECT_FALSE(
      find_out_of_vocab_token(token_ids, /*vocab_size=*/1000).has_value());
}

TEST(VocabValidationTest, NegativeTokenIsFlagged) {
  const std::vector<int> token_ids = {5, -1, 7};
  const auto invalid = find_out_of_vocab_token(token_ids, /*vocab_size=*/1000);
  ASSERT_TRUE(invalid.has_value());
  EXPECT_EQ(invalid.value(), -1);
}

TEST(VocabValidationTest, TokenEqualToVocabSizeIsFlagged) {
  const std::vector<int> token_ids = {1000};
  const auto invalid = find_out_of_vocab_token(token_ids, /*vocab_size=*/1000);
  ASSERT_TRUE(invalid.has_value());
  EXPECT_EQ(invalid.value(), 1000);
}

TEST(VocabValidationTest, UpperAndLowerBoundaryAreInRange) {
  const std::vector<int> token_ids = {0, 999};
  EXPECT_FALSE(
      find_out_of_vocab_token(token_ids, /*vocab_size=*/1000).has_value());
}

TEST(VocabValidationTest, ReturnsFirstOutOfRangeToken) {
  const std::vector<int> token_ids = {2000, 3000};
  const auto invalid = find_out_of_vocab_token(token_ids, /*vocab_size=*/1000);
  ASSERT_TRUE(invalid.has_value());
  EXPECT_EQ(invalid.value(), 2000);
}

}  // namespace xllm::util

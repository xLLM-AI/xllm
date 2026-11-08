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

#include "core/runtime/decode_input_placeholders.h"

#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <unordered_set>

namespace xllm {
namespace {

TEST(DecodeInputPlaceholdersTest, LeavesInputsWithoutFailedSequencesUntouched) {
  torch::Tensor token_ids = torch::tensor({-1, 17}, torch::kInt32);
  const torch::Tensor original = token_ids;
  std::string error = "previous error";

  ASSERT_TRUE(detail::reset_failed_sequence_placeholders(
      &token_ids, {"healthy"}, {}, {}, &error));
  EXPECT_TRUE(error.empty());
  EXPECT_EQ(token_ids.data_ptr(), original.data_ptr());
  EXPECT_TRUE(torch::equal(token_ids, original));
}

TEST(DecodeInputPlaceholdersTest, LeavesUndefinedInputsUntouched) {
  torch::Tensor token_ids;
  std::string error = "previous error";

  ASSERT_TRUE(detail::reset_failed_sequence_placeholders(
      &token_ids, {"failed"}, {}, {"failed"}, &error));
  EXPECT_TRUE(error.empty());
  EXPECT_FALSE(token_ids.defined());
}

TEST(DecodeInputPlaceholdersTest,
     RejectsMisalignedMetadataBeforeChangingTokens) {
  torch::Tensor token_ids = torch::tensor({-1, 17}, torch::kInt32);
  const torch::Tensor original = token_ids;
  std::string error;

  EXPECT_FALSE(detail::reset_failed_sequence_placeholders(
      &token_ids, {"failed"}, {}, {"failed"}, &error));
  EXPECT_EQ(error, "overlap error sanitization metadata must align");
  EXPECT_EQ(token_ids.data_ptr(), original.data_ptr());
  EXPECT_TRUE(torch::equal(token_ids, original));
}

TEST(DecodeInputPlaceholdersTest, ResetsOnlyFailedRowPlaceholder) {
  torch::Tensor token_ids = torch::tensor({-1, 17, -2}, torch::kInt);
  std::string error;

  ASSERT_TRUE(detail::reset_failed_sequence_placeholders(
      &token_ids,
      {"req-failed#0", "req-healthy#0"},
      {0, 1},
      {"req-failed#0"},
      &error))
      << error;
  EXPECT_TRUE(torch::equal(token_ids, torch::tensor({0, 17, -2})));
}

TEST(DecodeInputPlaceholdersTest, RejectsAmbiguousFailedRowPlaceholder) {
  torch::Tensor token_ids = torch::tensor({-1, 17, -1}, torch::kInt);
  std::string error;

  EXPECT_FALSE(detail::reset_failed_sequence_placeholders(
      &token_ids, {"req-failed#0"}, {0}, {"req-failed#0"}, &error));
  EXPECT_NE(error.find("exactly one"), std::string::npos);
  EXPECT_TRUE(torch::equal(token_ids, torch::tensor({-1, 17, -1})));
}

TEST(DecodeInputPlaceholdersTest, ResetsMultipleRowsWithoutChangingAliases) {
  for (const torch::ScalarType dtype : {torch::kInt32, torch::kInt64}) {
    torch::Tensor token_ids = torch::tensor({{-1, 17}, {-3, -2}}, dtype);
    const torch::Tensor original = token_ids;
    std::string error;

    ASSERT_TRUE(detail::reset_failed_sequence_placeholders(
        &token_ids,
        {"failed-a", "healthy", "failed-b"},
        {0, 1, 2},
        {"failed-b", "failed-a", "failed-a"},
        &error))
        << error;
    EXPECT_TRUE(torch::equal(token_ids, torch::tensor({{0, 17}, {0, -2}})));
    EXPECT_TRUE(torch::equal(original, torch::tensor({{-1, 17}, {-3, -2}})));
    EXPECT_NE(token_ids.data_ptr(), original.data_ptr());
    EXPECT_EQ(token_ids.scalar_type(), dtype);
    EXPECT_TRUE(token_ids.device().is_cpu());
  }
}

TEST(DecodeInputPlaceholdersTest, KeepsEarlierResetWhenLaterRowFails) {
  torch::Tensor token_ids = torch::tensor({-1, -2, -2}, torch::kInt64);
  const torch::Tensor original = token_ids;
  std::string error;

  EXPECT_FALSE(
      detail::reset_failed_sequence_placeholders(&token_ids,
                                                 {"first", "ambiguous"},
                                                 {0, 1},
                                                 {"first", "ambiguous"},
                                                 &error));
  EXPECT_EQ(
      error,
      "failed overlap row placeholder must identify exactly one input token");
  EXPECT_TRUE(torch::equal(token_ids, torch::tensor({0, -2, -2})));
  EXPECT_TRUE(torch::equal(original, torch::tensor({-1, -2, -2})));
}

TEST(DecodeInputPlaceholdersTest, DoesNotResetPastFirstInvalidRow) {
  torch::Tensor token_ids = torch::tensor({-1, -1, -2}, torch::kInt32);
  const torch::Tensor original = token_ids;
  std::string error;

  EXPECT_FALSE(
      detail::reset_failed_sequence_placeholders(&token_ids,
                                                 {"ambiguous", "later"},
                                                 {0, 1},
                                                 {"ambiguous", "later"},
                                                 &error));
  EXPECT_TRUE(torch::equal(token_ids, original));
  EXPECT_EQ(token_ids.data_ptr(), original.data_ptr());
}

TEST(DecodeInputPlaceholdersTest, RejectsSecondUseOfResetPlaceholder) {
  torch::Tensor token_ids = torch::tensor({-1, 17}, torch::kInt32);
  const torch::Tensor original = token_ids;
  std::string error;

  EXPECT_FALSE(detail::reset_failed_sequence_placeholders(
      &token_ids, {"first", "second"}, {0, 0}, {"first", "second"}, &error));
  EXPECT_NE(error.find("exactly one"), std::string::npos);
  EXPECT_TRUE(torch::equal(token_ids, torch::tensor({0, 17})));
  EXPECT_TRUE(torch::equal(original, torch::tensor({-1, 17})));
}

TEST(DecodeInputPlaceholdersTest, LeavesUnmatchedAndNewFailedRowsUntouched) {
  torch::Tensor token_ids = torch::tensor({-1, 17}, torch::kInt64);
  const torch::Tensor original = token_ids;
  std::string error;

  ASSERT_TRUE(detail::reset_failed_sequence_placeholders(
      &token_ids, {"healthy", "new"}, {0, -1}, {"absent", "new"}, &error))
      << error;
  EXPECT_TRUE(torch::equal(token_ids, original));
  EXPECT_EQ(token_ids.data_ptr(), original.data_ptr());
}

TEST(DecodeInputPlaceholdersTest, ResetsStridedAndFloatingPointInputs) {
  torch::Tensor strided =
      torch::tensor({{-1, 17}, {-2, 19}}, torch::kInt32).t();
  ASSERT_FALSE(strided.is_contiguous());
  const torch::Tensor original = strided;
  std::string error;
  ASSERT_TRUE(detail::reset_failed_sequence_placeholders(
      &strided, {"failed"}, {0}, {"failed"}, &error))
      << error;
  EXPECT_TRUE(torch::equal(strided, torch::tensor({{0, -2}, {17, 19}})));
  EXPECT_TRUE(torch::equal(original, torch::tensor({{-1, -2}, {17, 19}})));

  torch::Tensor floating = torch::tensor({-1.0, 17.5}, torch::kFloat64);
  ASSERT_TRUE(detail::reset_failed_sequence_placeholders(
      &floating, {"failed"}, {0}, {"failed"}, &error))
      << error;
  EXPECT_TRUE(
      torch::equal(floating, torch::tensor({0.0, 17.5}, torch::kFloat64)));
  EXPECT_EQ(floating.scalar_type(), torch::kFloat64);
}

}  // namespace
}  // namespace xllm

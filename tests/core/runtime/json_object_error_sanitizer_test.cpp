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

#include <string>
#include <vector>

#include "core/runtime/json_object_output_binding.h"

namespace xllm {
namespace {

TEST(JsonObjectErrorSanitizerTest, SanitizesOnlyFailedRowPlaceholder) {
  torch::Tensor token_ids = torch::tensor({-1, 17, -2}, torch::kInt);
  std::string error;

  ASSERT_TRUE(detail::sanitize_json_object_error_token_ids(
      &token_ids,
      {"req-failed#0", "req-healthy#0"},
      {0, 1},
      {{"req-failed#0", "row mismatch"}},
      &error))
      << error;
  EXPECT_TRUE(torch::equal(token_ids, torch::tensor({0, 17, -2})));
}

TEST(JsonObjectErrorSanitizerTest, RejectsAmbiguousFailedRowPlaceholder) {
  torch::Tensor token_ids = torch::tensor({-1, 17, -1}, torch::kInt);
  std::string error;

  EXPECT_FALSE(detail::sanitize_json_object_error_token_ids(
      &token_ids,
      {"req-failed#0"},
      {0},
      {{"req-failed#0", "row mismatch"}},
      &error));
  EXPECT_NE(error.find("exactly one"), std::string::npos);
  EXPECT_TRUE(torch::equal(token_ids, torch::tensor({-1, 17, -1})));
}

TEST(JsonObjectErrorSanitizerTest,
     SanitizesMultipleRowsWithoutChangingAliases) {
  for (const torch::ScalarType dtype : {torch::kInt32, torch::kInt64}) {
    torch::Tensor token_ids = torch::tensor({{-1, 17}, {-3, -2}}, dtype);
    const torch::Tensor original = token_ids;
    std::string error;

    ASSERT_TRUE(detail::sanitize_json_object_error_token_ids(
        &token_ids,
        {"failed-a", "healthy", "failed-b"},
        {0, 1, 2},
        {{"failed-b", "invalid"},
         {"failed-a", "invalid"},
         {"failed-a", "repeated error"}},
        &error))
        << error;
    EXPECT_TRUE(torch::equal(token_ids, torch::tensor({{0, 17}, {0, -2}})));
    EXPECT_TRUE(torch::equal(original, torch::tensor({{-1, 17}, {-3, -2}})));
    EXPECT_NE(token_ids.data_ptr(), original.data_ptr());
    EXPECT_EQ(token_ids.scalar_type(), dtype);
    EXPECT_TRUE(token_ids.device().is_cpu());
  }
}

TEST(JsonObjectErrorSanitizerTest, KeepsEarlierSanitizationWhenLaterRowFails) {
  torch::Tensor token_ids = torch::tensor({-1, -2, -2}, torch::kInt64);
  const torch::Tensor original = token_ids;
  std::string error;

  EXPECT_FALSE(detail::sanitize_json_object_error_token_ids(
      &token_ids,
      {"first", "ambiguous"},
      {0, 1},
      {{"first", "invalid"}, {"ambiguous", "invalid"}},
      &error));
  EXPECT_EQ(
      error,
      "failed JSON row placeholder must identify exactly one input token");
  EXPECT_TRUE(torch::equal(token_ids, torch::tensor({0, -2, -2})));
  EXPECT_TRUE(torch::equal(original, torch::tensor({-1, -2, -2})));
}

TEST(JsonObjectErrorSanitizerTest, DoesNotSanitizePastFirstInvalidRow) {
  torch::Tensor token_ids = torch::tensor({-1, -1, -2}, torch::kInt32);
  const torch::Tensor original = token_ids;
  std::string error;

  EXPECT_FALSE(detail::sanitize_json_object_error_token_ids(
      &token_ids,
      {"ambiguous", "later"},
      {0, 1},
      {{"ambiguous", "invalid"}, {"later", "invalid"}},
      &error));
  EXPECT_TRUE(torch::equal(token_ids, original));
  EXPECT_EQ(token_ids.data_ptr(), original.data_ptr());
}

TEST(JsonObjectErrorSanitizerTest, RejectsSecondUseOfSanitizedPlaceholder) {
  torch::Tensor token_ids = torch::tensor({-1, 17}, torch::kInt32);
  const torch::Tensor original = token_ids;
  std::string error;

  EXPECT_FALSE(detail::sanitize_json_object_error_token_ids(
      &token_ids,
      {"first", "second"},
      {0, 0},
      {{"first", "invalid"}, {"second", "invalid"}},
      &error));
  EXPECT_NE(error.find("exactly one"), std::string::npos);
  EXPECT_TRUE(torch::equal(token_ids, torch::tensor({0, 17})));
  EXPECT_TRUE(torch::equal(original, torch::tensor({-1, 17})));
}

TEST(JsonObjectErrorSanitizerTest, LeavesUnmatchedAndNewFailedRowsUntouched) {
  torch::Tensor token_ids = torch::tensor({-1, 17}, torch::kInt64);
  const torch::Tensor original = token_ids;
  std::string error;

  ASSERT_TRUE(detail::sanitize_json_object_error_token_ids(
      &token_ids,
      {"healthy", "new"},
      {0, -1},
      {{"absent", "invalid"}, {"new", "invalid"}},
      &error))
      << error;
  EXPECT_TRUE(torch::equal(token_ids, original));
  EXPECT_EQ(token_ids.data_ptr(), original.data_ptr());
}

TEST(JsonObjectErrorSanitizerTest, SanitizesStridedAndFloatingPointInputs) {
  torch::Tensor strided =
      torch::tensor({{-1, 17}, {-2, 19}}, torch::kInt32).t();
  ASSERT_FALSE(strided.is_contiguous());
  const torch::Tensor original = strided;
  std::string error;
  ASSERT_TRUE(detail::sanitize_json_object_error_token_ids(
      &strided, {"failed"}, {0}, {{"failed", "invalid"}}, &error))
      << error;
  EXPECT_TRUE(torch::equal(strided, torch::tensor({{0, -2}, {17, 19}})));
  EXPECT_TRUE(torch::equal(original, torch::tensor({{-1, -2}, {17, 19}})));

  torch::Tensor floating = torch::tensor({-1.0, 17.5}, torch::kFloat64);
  ASSERT_TRUE(detail::sanitize_json_object_error_token_ids(
      &floating, {"failed"}, {0}, {{"failed", "invalid"}}, &error))
      << error;
  EXPECT_TRUE(
      torch::equal(floating, torch::tensor({0.0, 17.5}, torch::kFloat64)));
  EXPECT_EQ(floating.scalar_type(), torch::kFloat64);
}

}  // namespace
}  // namespace xllm

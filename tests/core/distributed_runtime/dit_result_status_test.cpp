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

#include <folly/ExceptionWrapper.h>
#include <folly/Try.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/distributed_runtime/dit_engine.h"
#include "core/runtime/dit_forward_params.h"
#include "core/runtime/forward_params.h"

namespace xllm {
namespace {

using RankResult = folly::Try<std::optional<RawForwardOutput>>;

std::vector<RankResult> successful_ranks() {
  std::vector<RankResult> results;
  results.reserve(16);
  for (size_t rank = 0; rank < 16; ++rank) {
    results.emplace_back(std::optional<RawForwardOutput>(std::in_place));
  }
  results.front().value()->dit_forward_output.text_output = {"driver"};
  return results;
}

TEST(DiTResultStatusTest, SixteenRanksAllowEmptySuccessFollowers) {
  auto results = successful_ranks();
  const DiTForwardOutput output = select_dit_driver_output(results);
  EXPECT_EQ(output.text_output, (std::vector<std::string>{"driver"}));
}

TEST(DiTResultStatusTest, FollowerFailureIsCheckedBeforeMovingDriverOutput) {
  auto results = successful_ranks();
  results[7].value()->dit_execution_status =
      Status(StatusCode::INVALID_ARGUMENT, "bad condition");
  try {
    select_dit_driver_output(results);
    FAIL() << "follower failure must reject the complete batch";
  } catch (const DiTForwardExecutionError& error) {
    EXPECT_EQ(error.status().code(), StatusCode::INVALID_ARGUMENT);
    EXPECT_NE(error.status().message().find("rank=7"), std::string::npos);
    EXPECT_NE(error.status().message().find("bad condition"),
              std::string::npos);
  }
  EXPECT_EQ(results.front().value()->dit_forward_output.text_output,
            (std::vector<std::string>{"driver"}));
}

TEST(DiTResultStatusTest, TypedFutureFailurePreservesCodeAndRank) {
  auto results = successful_ranks();
  results[7] =
      RankResult(folly::make_exception_wrapper<DiTForwardExecutionError>(
          Status(StatusCode::RESOURCE_EXHAUSTED, "allocation failed")));
  try {
    select_dit_driver_output(results);
    FAIL() << "expected typed exception";
  } catch (const DiTForwardExecutionError& error) {
    EXPECT_EQ(error.status().code(), StatusCode::RESOURCE_EXHAUSTED);
    EXPECT_NE(error.status().message().find("rank=7"), std::string::npos);
    EXPECT_NE(error.status().message().find("allocation failed"),
              std::string::npos);
  }
}

TEST(DiTResultStatusTest, UntypedFutureFailureBecomesUnknownWithRank) {
  auto results = successful_ranks();
  results[3] = RankResult(
      folly::make_exception_wrapper<std::runtime_error>("transport"));
  try {
    select_dit_driver_output(results);
    FAIL() << "expected normalized exception";
  } catch (const DiTForwardExecutionError& error) {
    EXPECT_EQ(error.status().code(), StatusCode::UNKNOWN);
    EXPECT_NE(error.status().message().find("rank=3"), std::string::npos);
    EXPECT_NE(error.status().message().find("transport"), std::string::npos);
  }
}

TEST(DiTResultStatusTest, EmptyRanksAndMissingTransportResultAreFailures) {
  std::vector<RankResult> empty;
  EXPECT_THROW(select_dit_driver_output(empty), DiTForwardExecutionError);
  auto results = successful_ranks();
  results[5] = RankResult(std::optional<RawForwardOutput>());
  try {
    select_dit_driver_output(results);
    FAIL() << "missing transport result is not a follower success";
  } catch (const DiTForwardExecutionError& error) {
    EXPECT_EQ(error.status().code(), StatusCode::UNKNOWN);
    EXPECT_NE(error.status().message().find("rank=5"), std::string::npos);
  }
}

}  // namespace
}  // namespace xllm

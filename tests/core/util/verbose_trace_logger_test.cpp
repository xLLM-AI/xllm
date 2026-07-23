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

#include "core/util/verbose_trace_logger.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {

TEST(VerboseTraceLogPathTest, SingleRankPreservesConfiguredPath) {
  EXPECT_EQ(resolve_verbose_trace_log_path(
                "log/verbose_trace.log", /*nnodes=*/1, /*node_rank=*/0),
            "log/verbose_trace.log");
}

TEST(VerboseTraceLogPathTest, MultiRankInsertsRankBeforeLastExtension) {
  EXPECT_EQ(resolve_verbose_trace_log_path(
                "log.v1/verbose.trace.log", /*nnodes=*/4, /*node_rank=*/2),
            "log.v1/verbose.trace_rank_2.log");
}

TEST(VerboseTraceLogPathTest, MultiRankSuffixesRankZero) {
  EXPECT_EQ(resolve_verbose_trace_log_path(
                "log/verbose_trace.log", /*nnodes=*/4, /*node_rank=*/0),
            "log/verbose_trace_rank_0.log");
}

TEST(VerboseTraceLogPathTest, MultiRankAppendsRankWithoutExtension) {
  EXPECT_EQ(resolve_verbose_trace_log_path(
                "log/verbose_trace", /*nnodes=*/4, /*node_rank=*/3),
            "log/verbose_trace_rank_3");
}

TEST(VerboseTraceLogPathTest, EmptyPathRemainsEmpty) {
  EXPECT_TRUE(resolve_verbose_trace_log_path("", /*nnodes=*/4, /*node_rank=*/1)
                  .empty());
}

}  // namespace
}  // namespace xllm

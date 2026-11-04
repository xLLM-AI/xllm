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

#include "core/runtime/minimax_h3_failure_injection.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {

TEST(MiniMaxH3FailureInjectionTest, NativeAndDisabledNeverInject) {
  EXPECT_FALSE(should_inject_minimax_h3_failure(
      /*qualified_profile=*/false, /*enabled=*/true, 7, 3, 7, 3));
  EXPECT_FALSE(should_inject_minimax_h3_failure(
      /*qualified_profile=*/true, /*enabled=*/false, 7, 3, 7, 3));
}

TEST(MiniMaxH3FailureInjectionTest, MatchesOnlySpecifiedFollowerAndRequest) {
  for (int32_t rank = 0; rank < 16; ++rank) {
    EXPECT_EQ(should_inject_minimax_h3_failure(
                  /*qualified_profile=*/true, /*enabled=*/true, 7, 3, rank, 3),
              rank == 7);
  }
  EXPECT_FALSE(should_inject_minimax_h3_failure(true, true, 7, 3, 7, 2));
  EXPECT_FALSE(should_inject_minimax_h3_failure(true, true, 7, 3, 7, 4));
  EXPECT_FALSE(should_inject_minimax_h3_failure(true, true, -1, 3, 7, 3));
  EXPECT_FALSE(should_inject_minimax_h3_failure(true, true, 16, 3, 16, 3));
  EXPECT_FALSE(should_inject_minimax_h3_failure(true, true, 7, 0, 7, 0));
}

}  // namespace
}  // namespace xllm

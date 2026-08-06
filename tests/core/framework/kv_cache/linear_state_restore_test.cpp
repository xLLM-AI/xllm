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

#include "framework/kv_cache/linear_state_restore.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace xllm {
namespace {

TEST(LinearStateRestoreTest, BuildsColdMaskForUncachedRow) {
  EXPECT_EQ(build_linear_state_mask(/*cached_tokens=*/{0}, /*active_rows=*/1),
            std::vector<int64_t>({0}));
}

TEST(LinearStateRestoreTest, BuildsWarmMaskForCachedRow) {
  EXPECT_EQ(build_linear_state_mask(/*cached_tokens=*/{8}, /*active_rows=*/1),
            std::vector<int64_t>({1}));
}

TEST(LinearStateRestoreTest, BuildsMixedWarmMask) {
  EXPECT_EQ(
      build_linear_state_mask(/*cached_tokens=*/{0, 8, -1}, /*active_rows=*/3),
      std::vector<int64_t>({0, 1, 0}));
}

TEST(LinearStateRestoreTest, RepeatsLogicalRowsForActiveRows) {
  EXPECT_EQ(build_linear_state_mask(/*cached_tokens=*/{0, 8},
                                    /*active_rows=*/6),
            std::vector<int64_t>({0, 0, 0, 1, 1, 1}));
}

TEST(LinearStateRestoreTest, RejectsEmptyCachedTokens) {
  EXPECT_DEATH(build_linear_state_mask(/*cached_tokens=*/{},
                                       /*active_rows=*/1),
               "cached_tokens must not be empty");
}

TEST(LinearStateRestoreTest, RejectsNonPositiveActiveRows) {
  EXPECT_DEATH(build_linear_state_mask(/*cached_tokens=*/{0},
                                       /*active_rows=*/0),
               "active_rows must be positive");
}

TEST(LinearStateRestoreTest, RejectsNonDivisibleActiveRows) {
  EXPECT_DEATH(build_linear_state_mask(/*cached_tokens=*/{0, 8},
                                       /*active_rows=*/3),
               "logical rows must evenly divide active rows");
}

TEST(LinearStateRestoreTest, ModelInputConversionPreservesValidityMask) {
  ModelInputParams input_params;
  input_params.linear_state_validity_mask = {0, 1, 1, 0};

  ModelInputParams converted = input_params.to(torch::Device(torch::kCPU));

  EXPECT_EQ(converted.linear_state_validity_mask,
            std::vector<int64_t>({0, 1, 1, 0}));
}

}  // namespace
}  // namespace xllm

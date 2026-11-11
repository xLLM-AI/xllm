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

#include "core/framework/eplb/eplb_options.h"

#include <gtest/gtest.h>

#include <limits>

namespace xllm {
namespace {

TEST(EplbOptionsTest, AcceptsDocumentedDefaults) {
  EplbOptions options;
  options.validate();
  EXPECT_EQ(options.eplb_policy_kind, "balanced");
}

TEST(EplbOptionsTest, RejectsInvalidValuesBeforeRuntime) {
  EXPECT_DEATH(
      {
        EplbOptions options;
        options.eplb_update_interval = -1;
        options.validate();
      },
      "eplb_update_interval");
  EXPECT_DEATH(
      {
        EplbOptions options;
        options.eplb_min_peak_load_improvement = 1.1;
        options.validate();
      },
      "eplb_min_peak_load_improvement");
}

TEST(EplbOptionsTest, AcceptsZeroRedundancyIntervalAndImprovement) {
  EplbOptions options;
  options.redundant_experts_num = 0;
  options.eplb_update_interval = 0;
  options.eplb_min_peak_load_improvement = 0.0;
  options.validate();
}

TEST(EplbOptionsTest, RejectsNegativeRedundancyAndNonpositiveTimeout) {
  EXPECT_DEATH(
      {
        EplbOptions options;
        options.redundant_experts_num = -1;
        options.validate();
      },
      "redundant_experts_num");
  for (int32_t timeout : {0, -1}) {
    EXPECT_DEATH(
        {
          EplbOptions options;
          options.eplb_prepare_timeout_seconds = timeout;
          options.validate();
        },
        "eplb_prepare_timeout_seconds");
  }
}

TEST(EplbOptionsTest, RejectsNonfiniteAndNegativeImprovement) {
  for (double improvement : {-0.01,
                             std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::infinity(),
                             -std::numeric_limits<double>::infinity()}) {
    EXPECT_DEATH(
        {
          EplbOptions options;
          options.eplb_min_peak_load_improvement = improvement;
          options.validate();
        },
        "eplb_min_peak_load_improvement");
  }
}

}  // namespace
}  // namespace xllm

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

#include "core/framework/speculative/speculative_profile_registry.h"

#include <gtest/gtest.h>

#include <optional>

namespace xllm {
namespace {

TEST(SpeculativeProfileRegistryTest, PredictsValidateTimeFromLinearTerms) {
  SpeculativeProfileRegistry& registry =
      SpeculativeProfileRegistry::get_instance();
  registry.reset_validate_time_predictor();

  SpeculativeProfileRegistry::ValidateTimePredictor predictor;
  predictor.intercept_ms = 1.0;
  predictor.batch_ms = 2.0;
  predictor.query_token_ms = 3.0;
  predictor.query_prefix_ms = 0.5;
  registry.set_validate_time_predictor(predictor);

  const std::optional<SpeculativeProfileRegistry::ValidateTimePredictor>
      registered_predictor = registry.validate_time_predictor();
  ASSERT_TRUE(registered_predictor.has_value());
  EXPECT_DOUBLE_EQ(registered_predictor->query_prefix_ms, 0.5);

  const double time_ms = registry.predict_validate_time_ms(
      /*batch_size=*/4, /*query_len=*/3, /*avg_prefix_len=*/5.0);

  EXPECT_DOUBLE_EQ(time_ms,
                   1.0 + 2.0 * 4.0 + 3.0 * 4.0 * 3.0 + 0.5 * 4.0 * 3.0 * 5.0);
  registry.reset_validate_time_predictor();
}

TEST(SpeculativeProfileRegistryTest, ReturnsZeroBeforeRegistration) {
  SpeculativeProfileRegistry& registry =
      SpeculativeProfileRegistry::get_instance();
  registry.reset_validate_time_predictor();

  EXPECT_DOUBLE_EQ(registry.predict_validate_time_ms(
                       /*batch_size=*/4,
                       /*query_len=*/3,
                       /*avg_prefix_len=*/128.0),
                   0.0);
}

}  // namespace
}  // namespace xllm

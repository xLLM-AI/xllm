/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/framework/config/dit_config.h"
#include "models/dit/utils/minimax_h3_rainfusion_context.h"

namespace xllm {
namespace {
TEST(H3RainfusionPolicyTest, InactiveRequestNeverSelectsKernel) {
  H3RainfusionRequestState state;
  state.policy = kH3SparseMiddleSteps;
  for (int64_t step = 0; step < 49; ++step) {
    EXPECT_EQ(active_h3_rainfusion_context(state, step), nullptr);
  }
  EXPECT_EQ(state.borrowed_kernel_context.launches, 0);
}
TEST(H3RainfusionPolicyTest, SelectsOnlyFrozenHalfOpenWindow) {
  H3RainfusionRequestState state;
  state.policy = kH3SparseMiddleSteps;
  state.ready = true;
  EXPECT_EQ(active_h3_rainfusion_context(state, 7), nullptr);
  EXPECT_EQ(active_h3_rainfusion_context(state, 8),
            &state.borrowed_kernel_context);
  EXPECT_EQ(active_h3_rainfusion_context(state, 35),
            &state.borrowed_kernel_context);
  EXPECT_EQ(active_h3_rainfusion_context(state, 36), nullptr);
  int64_t selected = 0;
  for (int64_t step = 0; step < 49; ++step) {
    selected += active_h3_rainfusion_context(state, step) != nullptr;
  }
  EXPECT_EQ(selected, 28);
  EXPECT_EQ(state.borrowed_kernel_context.launches, 0);
}
TEST(H3RainfusionPolicyTest, NewRequestDoesNotInheritLaunchesOrReadiness) {
  H3RainfusionRequestState first;
  first.ready = true;
  first.borrowed_kernel_context.launches = 1400;
  H3RainfusionRequestState second;
  EXPECT_FALSE(second.ready);
  EXPECT_FALSE(second.policy.enabled());
  EXPECT_EQ(second.borrowed_kernel_context.launches, 0);
  EXPECT_EQ(active_h3_rainfusion_context(second, 8), nullptr);
}
TEST(H3RainfusionPolicyTest, ResolvesEnabledEnvironmentAndDefaults) {
  EXPECT_EQ(resolve_h3_rainfusion_policy(false, "invalid").name,
            kH3SparseNone.name);
  EXPECT_EQ(resolve_h3_rainfusion_policy(true, nullptr).name,
            kH3SparseMiddleSteps.name);
  EXPECT_EQ(resolve_h3_rainfusion_policy(true, "sparse-middle-steps").name,
            kH3SparseMiddleSteps.name);
  EXPECT_EQ(resolve_h3_rainfusion_policy(true, "sparse-middle-layers").name,
            kH3SparseMiddleLayers.name);
  EXPECT_THROW(resolve_h3_rainfusion_policy(true, ""), std::invalid_argument);
  EXPECT_THROW(resolve_h3_rainfusion_policy(true, "invalid"),
               std::invalid_argument);
}

TEST(H3RainfusionPolicyTest, MiddleLayersUseHalfOpenLayerBoundsOnEveryStep) {
  H3RainfusionRequestState state;
  state.policy = kH3SparseMiddleLayers;
  state.ready = true;
  EXPECT_NE(active_h3_rainfusion_context(state, 0), nullptr);
  EXPECT_NE(active_h3_rainfusion_context(state, 48), nullptr);
  EXPECT_EQ(active_h3_rainfusion_context(state, 49), nullptr);
  EXPECT_FALSE(state.policy.includes_layer(1));
  EXPECT_TRUE(state.policy.includes_layer(2));
  EXPECT_TRUE(state.policy.includes_layer(39));
  EXPECT_FALSE(state.policy.includes_layer(40));
  EXPECT_EQ(state.policy.layer_count(0, 50), 38);
  EXPECT_EQ(state.policy.layer_count(0, 1), 0);
  EXPECT_EQ(state.policy.layer_count(0, 4), 2);
  EXPECT_EQ(state.policy.layer_count(38, 50), 2);
}

TEST(H3RainfusionPolicyTest, TypedOptionIsOffAndIndependentOfLegacySparse) {
  DiTConfig config;
  EXPECT_FALSE(config.dit_h3_rainfusion_enabled());
  config.dit_h3_rainfusion_enabled(true);
  EXPECT_TRUE(config.dit_h3_rainfusion_enabled());
  EXPECT_FALSE(config.dit_sparse_attention_enabled());
  EXPECT_EQ(config.dit_h3_execution_profile(), "native");
}
}  // namespace
}  // namespace xllm

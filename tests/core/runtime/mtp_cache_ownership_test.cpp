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

#include "core/framework/model/mtp_utils.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {

TEST(MtpCacheOwnershipTest, EnablesFenceForGlmEagerDpOverlap) {
  EXPECT_TRUE(requires_glm_mtp_cache_ownership_fence(
      /*schedule_overlap_enabled=*/true,
      /*num_speculative_tokens=*/3,
      /*dp_size=*/2,
      "glm_moe_dsa",
      /*graph_enabled=*/false));
}

TEST(MtpCacheOwnershipTest, DisablesFenceOutsideExactExecutionMode) {
  EXPECT_FALSE(requires_glm_mtp_cache_ownership_fence(
      /*schedule_overlap_enabled=*/false,
      /*num_speculative_tokens=*/3,
      /*dp_size=*/2,
      "glm_moe_dsa",
      /*graph_enabled=*/false));
  EXPECT_FALSE(requires_glm_mtp_cache_ownership_fence(
      /*schedule_overlap_enabled=*/true,
      /*num_speculative_tokens=*/0,
      /*dp_size=*/2,
      "glm_moe_dsa",
      /*graph_enabled=*/false));
  EXPECT_FALSE(requires_glm_mtp_cache_ownership_fence(
      /*schedule_overlap_enabled=*/true,
      /*num_speculative_tokens=*/3,
      /*dp_size=*/1,
      "glm_moe_dsa",
      /*graph_enabled=*/false));
  EXPECT_FALSE(requires_glm_mtp_cache_ownership_fence(
      /*schedule_overlap_enabled=*/true,
      /*num_speculative_tokens=*/3,
      /*dp_size=*/2,
      "qwen3_moe",
      /*graph_enabled=*/false));
  EXPECT_FALSE(requires_glm_mtp_cache_ownership_fence(
      /*schedule_overlap_enabled=*/true,
      /*num_speculative_tokens=*/3,
      /*dp_size=*/2,
      "glm_moe_dsa",
      /*graph_enabled=*/true));
}

}  // namespace
}  // namespace xllm

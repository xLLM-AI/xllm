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

#include "models/dit/utils/minimax_h3_qualified_execution_context.h"

namespace xllm {
namespace {

H3PackedLayout probe_layout() {
  H3PackedLayout layout;
  layout.aligned_length = 32;
  layout.used_length = 31;
  layout.img_pos = torch::tensor({0, 1, 6, 9, 15}, torch::kInt64);
  layout.update_mask =
      torch::tensor({false, true, true, true, true}, torch::kBool);
  layout.audio_pos = torch::tensor({16, 30}, torch::kInt64);
  layout.audio_update_mask = torch::tensor({true, true}, torch::kBool);
  return layout;
}

TEST(H3QualifiedCacheRowsTest, ConvertsGlobalTargetsUsingActualSpLocalRank) {
  const H3PackedLayout layout = probe_layout();
  for (const int32_t rank : {1, 3, 5, 7}) {
    const H3QualifiedCacheProbeRows rows = h3_qualified_cache_probe_rows(
        layout, rank, 8, torch::Device(torch::kCPU));
    EXPECT_EQ(rows.used_rows, rank == 7 ? 3 : 4);
    ASSERT_TRUE(rows.target_indices.defined());
    if (rank == 5) {
      EXPECT_EQ(rows.target_indices.numel(), 0);
    } else {
      EXPECT_TRUE(torch::equal(
          rows.target_indices,
          torch::tensor({int64_t{rank == 3 ? 3 : 2}}, torch::kInt64)));
    }
  }
}

TEST(H3QualifiedCacheRowsTest, RejectsGlobalEmptyTargetsAndAllPaddingShard) {
  H3PackedLayout layout = probe_layout();
  layout.used_length = 28;
  EXPECT_THROW(
      h3_qualified_cache_probe_rows(layout, 7, 8, torch::Device(torch::kCPU)),
      std::invalid_argument);
  layout = probe_layout();
  layout.update_mask.fill_(false);
  layout.audio_update_mask.fill_(false);
  EXPECT_THROW(
      h3_qualified_cache_probe_rows(layout, 0, 8, torch::Device(torch::kCPU)),
      std::invalid_argument);
}

TEST(H3QualifiedCacheRowsTest,
     DenseContextBorrowsWorldWithoutAllocatingProbeOrCache) {
  ProcessGroup world(0, 16, torch::Device(torch::kCPU));
  ParallelArgs parallel(0, 16, &world);
  H3QualifiedRequest request;
  DiTCacheConfig config;
  H3QualifiedExecutionContext execution = make_h3_qualified_execution_context(
      request, config, parallel, 49, torch::Device(torch::kCPU));
  EXPECT_EQ(execution.consensus_group, &world);
  EXPECT_EQ(execution.metrics.infer_steps, 49);
  EXPECT_EQ(execution.cache, nullptr);
  EXPECT_FALSE(execution.probe_rows.target_indices.defined());
  EXPECT_EQ(execution.metrics.executed_blocks, 0);
  EXPECT_TRUE(execution.metrics.hit_steps.empty());
}

}  // namespace
}  // namespace xllm

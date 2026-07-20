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

#include "core/platform/mlu/mlu_rdma_memory_plan.h"

#include <gtest/gtest.h>

#include <cstddef>

namespace xllm::mlu {
namespace {

TEST(MluRdmaMemoryPlanTest, AlignsSmallAllocationToWholeBlocks) {
  constexpr size_t kBlockCount = 2;
  constexpr size_t kBlockBytes = 96 * sizeof(float);
  constexpr size_t kLogicalBytes = kBlockCount * kBlockBytes;

  const RdmaMemoryPlan plan = make_rdma_memory_plan(kLogicalBytes, kBlockCount);

  EXPECT_EQ(plan.logical_bytes, kLogicalBytes);
  EXPECT_EQ(plan.block_bytes, kBlockBytes);
  EXPECT_EQ(plan.registered_bytes, 2097408U);
}

TEST(MluRdmaMemoryPlanTest, KeepsLogicalSizeAtAndAboveRegistrationMinimum) {
  constexpr size_t kBlockBytes = 16 * sizeof(float);
  constexpr size_t kBlocksAtMinimum = kMinRdmaRegisterBytes / kBlockBytes;

  const RdmaMemoryPlan at_minimum =
      make_rdma_memory_plan(kBlocksAtMinimum * kBlockBytes, kBlocksAtMinimum);
  EXPECT_EQ(at_minimum.registered_bytes, kMinRdmaRegisterBytes);

  const size_t blocks_above_minimum = kBlocksAtMinimum + 1;
  const size_t bytes_above_minimum = blocks_above_minimum * kBlockBytes;
  const RdmaMemoryPlan above_minimum =
      make_rdma_memory_plan(bytes_above_minimum, blocks_above_minimum);
  EXPECT_EQ(above_minimum.registered_bytes, bytes_above_minimum);
}

}  // namespace
}  // namespace xllm::mlu

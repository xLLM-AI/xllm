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

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "core/framework/allocator/torch/mapped_virtual_memory_tensor_view_internal.h"

namespace xllm {
namespace {

constexpr size_t kRegionSize = 4096;
constexpr size_t kElementSize = 2;

TEST(MappedVirtualMemoryTensorViewTest, AcceptsAlignedRangesAtRegionBounds) {
  EXPECT_EQ(detail::checked_tensor_view_numel(
                kRegionSize, kElementSize, /*offset=*/2, {2, 3}),
            6);
  EXPECT_EQ(detail::checked_tensor_view_numel(
                kRegionSize, kElementSize, kRegionSize - kElementSize, {1}),
            1);
  EXPECT_EQ(detail::checked_tensor_view_numel(
                kRegionSize, kElementSize, kRegionSize, {0}),
            0);
}

TEST(MappedVirtualMemoryTensorViewTest, ScalarShapeRequiresOneElement) {
  EXPECT_EQ(detail::checked_tensor_view_numel(
                kElementSize, kElementSize, /*offset=*/0, {}),
            1);
}

TEST(MappedVirtualMemoryTensorViewTest,
     ZeroDimensionAvoidsElementCountOverflow) {
  EXPECT_EQ(detail::checked_tensor_view_numel(
                kRegionSize,
                kElementSize,
                kRegionSize,
                {std::numeric_limits<int64_t>::max(), 3, 0}),
            0);
}

class MappedVirtualMemoryTensorViewDeathTest : public ::testing::Test {
 protected:
  void SetUp() override { GTEST_FLAG_SET(death_test_style, "threadsafe"); }
};

TEST_F(MappedVirtualMemoryTensorViewDeathTest, RejectsOutOfBoundsOffset) {
  EXPECT_DEATH(detail::checked_tensor_view_numel(
                   kRegionSize, kElementSize, kRegionSize + kElementSize, {0}),
               "Tensor byte offset is out of bounds");
}

TEST_F(MappedVirtualMemoryTensorViewDeathTest, RejectsMisalignedOffset) {
  EXPECT_DEATH(detail::checked_tensor_view_numel(
                   kRegionSize, kElementSize, /*offset=*/1, {1}),
               "Tensor byte offset is not aligned");
}

TEST_F(MappedVirtualMemoryTensorViewDeathTest, RejectsNegativeDimension) {
  EXPECT_DEATH(detail::checked_tensor_view_numel(
                   kRegionSize, kElementSize, /*offset=*/0, {-1}),
               "Tensor dimensions must be nonnegative");
}

TEST_F(MappedVirtualMemoryTensorViewDeathTest, RejectsViewExceedingRegion) {
  EXPECT_DEATH(detail::checked_tensor_view_numel(
                   kRegionSize, kElementSize, kRegionSize - kElementSize, {2}),
               "Tensor view exceeds its virtual memory region");
}

TEST_F(MappedVirtualMemoryTensorViewDeathTest, RejectsElementCountOverflow) {
  EXPECT_DEATH(detail::checked_tensor_view_numel(
                   kRegionSize,
                   kElementSize,
                   /*offset=*/0,
                   {std::numeric_limits<int64_t>::max(), 3}),
               "num_elems");
}

}  // namespace
}  // namespace xllm

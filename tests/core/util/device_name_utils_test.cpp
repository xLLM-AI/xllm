/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/platform/device_name_utils.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "core/platform/device.h"
#include "core/platform/platform.h"

namespace xllm {
namespace {

struct RankMapping {
  int32_t node_rank;
  int32_t expected_device_idx;
};

class MultiNodeDeviceRankTest : public testing::TestWithParam<RankMapping> {};

TEST_P(MultiNodeDeviceRankTest, MapsGlobalRankToLocalLogicalDevice) {
  const RankMapping param = GetParam();

  EXPECT_EQ(DeviceNameUtils::get_device_idx(param.node_rank,
                                            /*nnodes=*/16,
                                            /*visible_device_count=*/8),
            param.expected_device_idx);
}

INSTANTIATE_TEST_SUITE_P(
    TwoNodesWithEightDevicesEach,
    MultiNodeDeviceRankTest,
    testing::Values(RankMapping{/*node_rank=*/0, /*expected_device_idx=*/0},
                    RankMapping{/*node_rank=*/7, /*expected_device_idx=*/7},
                    RankMapping{/*node_rank=*/8, /*expected_device_idx=*/0},
                    RankMapping{/*node_rank=*/15, /*expected_device_idx=*/7}));

TEST(DeviceNameUtilsTest, RejectsNegativeGlobalRank) {
  EXPECT_DEATH(DeviceNameUtils::get_device_idx(/*node_rank=*/-1,
                                               /*nnodes=*/16,
                                               /*visible_device_count=*/8),
               "node_rank");
}

TEST(DeviceNameUtilsTest, RejectsRankOutsideWorld) {
  EXPECT_DEATH(DeviceNameUtils::get_device_idx(/*node_rank=*/16,
                                               /*nnodes=*/16,
                                               /*visible_device_count=*/8),
               "node_rank");
}

TEST(DeviceNameUtilsTest, RejectsNoVisibleDevice) {
  EXPECT_DEATH(DeviceNameUtils::get_device_idx(/*node_rank=*/0,
                                               /*nnodes=*/1,
                                               /*visible_device_count=*/0),
               "accelerator device");
  EXPECT_DEATH(DeviceNameUtils::get_device_idx(/*node_rank=*/0,
                                               /*nnodes=*/1,
                                               /*visible_device_count=*/-1),
               "accelerator device");
}

TEST(DeviceNameUtilsTest, MapsAnyValidRankToOnlyVisibleDevice) {
  EXPECT_EQ(DeviceNameUtils::get_device_idx(/*node_rank=*/7,
                                            /*nnodes=*/8,
                                            /*visible_device_count=*/1),
            0);
}

TEST(DeviceNameUtilsTest, PreservesGlobalRankInput) {
  int32_t node_rank = 8;

  EXPECT_EQ(DeviceNameUtils::get_device_idx(node_rank,
                                            /*nnodes=*/16,
                                            /*visible_device_count=*/8),
            0);
  EXPECT_EQ(node_rank, 8);
}

TEST(DeviceNameUtilsTest, ParseGeneratedDeviceString) {
  const std::vector<torch::Device> devices =
      DeviceNameUtils::parse_devices(Platform::type_str() + ":2");

  ASSERT_EQ(devices.size(), 1);
  EXPECT_EQ(devices[0].type(), Platform::type_torch());
  EXPECT_EQ(devices[0].index(), 2);
}

}  // namespace
}  // namespace xllm

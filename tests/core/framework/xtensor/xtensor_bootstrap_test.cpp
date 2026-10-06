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
#include <torch/types.h>

#include "core/framework/config/distributed_config.h"
#include "core/framework/xtensor/xtensor_allocator.h"

namespace xllm {
namespace {

TEST(XTensorBootstrapTest, RejectsMissingLocalDeviceBeforeStartingServers) {
  xtensor::Options options;
  EXPECT_DEATH(XTensorAllocator::get_instance().setup_multi_node_xtensor_dist(
                   options, "127.0.0.1:0", /*dp_size=*/1),
               "exactly one local device per process");
}

TEST(XTensorBootstrapTest, RejectsMultipleLocalDevicesBeforeStartingServers) {
  xtensor::Options options;
  options.devices({torch::Device(torch::kCPU), torch::Device(torch::kCPU)});
  EXPECT_DEATH(XTensorAllocator::get_instance().setup_multi_node_xtensor_dist(
                   options, "127.0.0.1:0", /*dp_size=*/1),
               "exactly one local device per process");
}

TEST(XTensorBootstrapTest, RejectsNonpositiveDpSizeBeforeDivision) {
  xtensor::Options options;
  options.devices({torch::Device(torch::kCPU)});
  EXPECT_DEATH(XTensorAllocator::get_instance().setup_multi_node_xtensor_dist(
                   options, "127.0.0.1:0", /*dp_size=*/0),
               "dp_size must be positive");
  EXPECT_DEATH(XTensorAllocator::get_instance().setup_multi_node_xtensor_dist(
                   options, "127.0.0.1:0", /*dp_size=*/-1),
               "dp_size must be positive");
}

TEST(XTensorBootstrapTest, RejectsInvalidNodeRankBeforeStartingServers) {
  xtensor::Options options;
  options.devices({torch::Device(torch::kCPU)});
  EXPECT_DEATH(
      {
        DistributedConfig::get_instance().nnodes(1).node_rank(1);
        XTensorAllocator::get_instance().setup_multi_node_xtensor_dist(
            options, "127.0.0.1:0", /*dp_size=*/1);
      },
      "distributed_config.node_rank.*distributed_config.nnodes");
}

TEST(XTensorBootstrapTest, RejectsNondivisibleTopologyBeforeStartingServers) {
  xtensor::Options options;
  options.devices({torch::Device(torch::kCPU)});
  EXPECT_DEATH(
      {
        DistributedConfig::get_instance().nnodes(3).node_rank(0);
        XTensorAllocator::get_instance().setup_multi_node_xtensor_dist(
            options, "127.0.0.1:0", /*dp_size=*/2);
      },
      "world_size must be divisible by dp_size");
}

}  // namespace
}  // namespace xllm

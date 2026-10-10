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
#include "core/framework/virtual_memory/virtual_memory_manager.h"

namespace xllm {
namespace {

TEST(VirtualMemoryBootstrapTest,
     RejectsMissingLocalDeviceBeforeStartingServers) {
  VirtualMemoryOptions options;
  EXPECT_DEATH(
      VirtualMemoryManager::get_instance().setup_multi_node_virtual_memory_dist(
          options, "127.0.0.1:0", /*dp_size=*/1),
      "exactly one local device per process");
}

TEST(VirtualMemoryBootstrapTest,
     RejectsMultipleLocalDevicesBeforeStartingServers) {
  VirtualMemoryOptions options;
  options.devices({torch::Device(torch::kCPU), torch::Device(torch::kCPU)});
  EXPECT_DEATH(
      VirtualMemoryManager::get_instance().setup_multi_node_virtual_memory_dist(
          options, "127.0.0.1:0", /*dp_size=*/1),
      "exactly one local device per process");
}

TEST(VirtualMemoryBootstrapTest, RejectsNonpositiveDpSizeBeforeDivision) {
  VirtualMemoryOptions options;
  options.devices({torch::Device(torch::kCPU)});
  EXPECT_DEATH(
      VirtualMemoryManager::get_instance().setup_multi_node_virtual_memory_dist(
          options, "127.0.0.1:0", /*dp_size=*/0),
      "dp_size must be positive");
  EXPECT_DEATH(
      VirtualMemoryManager::get_instance().setup_multi_node_virtual_memory_dist(
          options, "127.0.0.1:0", /*dp_size=*/-1),
      "dp_size must be positive");
}

TEST(VirtualMemoryBootstrapTest, RejectsInvalidNodeRankBeforeStartingServers) {
  VirtualMemoryOptions options;
  options.devices({torch::Device(torch::kCPU)});
  EXPECT_DEATH(
      {
        DistributedConfig::get_instance().nnodes(1).node_rank(1);
        VirtualMemoryManager::get_instance()
            .setup_multi_node_virtual_memory_dist(
                options, "127.0.0.1:0", /*dp_size=*/1);
      },
      "distributed_config.node_rank.*distributed_config.nnodes");
}

TEST(VirtualMemoryBootstrapTest,
     RejectsNondivisibleTopologyBeforeStartingServers) {
  VirtualMemoryOptions options;
  options.devices({torch::Device(torch::kCPU)});
  EXPECT_DEATH(
      {
        DistributedConfig::get_instance().nnodes(3).node_rank(0);
        VirtualMemoryManager::get_instance()
            .setup_multi_node_virtual_memory_dist(
                options, "127.0.0.1:0", /*dp_size=*/2);
      },
      "world_size must be divisible by dp_size");
}

}  // namespace
}  // namespace xllm

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

#include "core/distributed_runtime/model_memory_manager.h"
#include "core/framework/config/distributed_config.h"

namespace xllm {
namespace {

TEST(ModelMemoryBootstrapTest, RejectsMissingLocalDeviceBeforeStartingServers) {
  ModelMemoryOptions options;
  EXPECT_DEATH(
      ModelMemoryManager::get_instance().setup_multi_node_model_memory_dist(
          options, "127.0.0.1:0", /*dp_size=*/1),
      "exactly one local device per process");
}

TEST(ModelMemoryBootstrapTest,
     RejectsMultipleLocalDevicesBeforeStartingServers) {
  ModelMemoryOptions options;
  options.devices({torch::Device(torch::kCPU), torch::Device(torch::kCPU)});
  EXPECT_DEATH(
      ModelMemoryManager::get_instance().setup_multi_node_model_memory_dist(
          options, "127.0.0.1:0", /*dp_size=*/1),
      "exactly one local device per process");
}

TEST(ModelMemoryBootstrapTest, RejectsNonpositiveDpSizeBeforeDivision) {
  ModelMemoryOptions options;
  options.devices({torch::Device(torch::kCPU)});
  EXPECT_DEATH(
      ModelMemoryManager::get_instance().setup_multi_node_model_memory_dist(
          options, "127.0.0.1:0", /*dp_size=*/0),
      "dp_size must be positive");
  EXPECT_DEATH(
      ModelMemoryManager::get_instance().setup_multi_node_model_memory_dist(
          options, "127.0.0.1:0", /*dp_size=*/-1),
      "dp_size must be positive");
}

TEST(ModelMemoryBootstrapTest, RejectsInvalidNodeRankBeforeStartingServers) {
  ModelMemoryOptions options;
  options.devices({torch::Device(torch::kCPU)});
  EXPECT_DEATH(
      {
        DistributedConfig::get_instance().nnodes(1).node_rank(1);
        ModelMemoryManager::get_instance().setup_multi_node_model_memory_dist(
            options, "127.0.0.1:0", /*dp_size=*/1);
      },
      "distributed_config.node_rank.*distributed_config.nnodes");
}

TEST(ModelMemoryBootstrapTest,
     RejectsNondivisibleTopologyBeforeStartingServers) {
  ModelMemoryOptions options;
  options.devices({torch::Device(torch::kCPU)});
  EXPECT_DEATH(
      {
        DistributedConfig::get_instance().nnodes(3).node_rank(0);
        ModelMemoryManager::get_instance().setup_multi_node_model_memory_dist(
            options, "127.0.0.1:0", /*dp_size=*/2);
      },
      "world_size must be divisible by dp_size");
}

}  // namespace
}  // namespace xllm

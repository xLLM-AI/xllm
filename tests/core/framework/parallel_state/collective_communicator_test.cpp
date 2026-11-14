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

#include "core/framework/parallel_state/collective_communicator.h"

#include <gtest/gtest.h>

#include "core/framework/config/eplb_config.h"

namespace xllm {
namespace {

TEST(CollectiveCommunicatorTest,
     RetainsInjectedEplbConfigAcrossParallelCopies) {
  EPLBConfig& global_config = EPLBConfig::get_instance();
  const bool previous_enabled = global_config.enable_eplb();
  global_config.enable_eplb(false);
  EplbWorkerConfig config;
  config.enabled = true;
  config.redundant_experts_num = 4;
  config.use_decode_only_load = true;
  CollectiveCommunicator communicator(/*global_rank=*/0,
                                      /*world_size=*/2,
                                      /*dp_size=*/1,
                                      /*ep_size=*/2,
                                      /*cp_size=*/1,
                                      config);
  config.enabled = false;
  config.redundant_experts_num = 0;
  config.use_decode_only_load = false;
  ParallelArgs worker_args = *communicator.parallel_args();
  global_config.enable_eplb(previous_enabled);

  EXPECT_TRUE(worker_args.eplb_worker_config().enabled);
  EXPECT_EQ(worker_args.eplb_worker_config().redundant_experts_num, 4);
  EXPECT_TRUE(worker_args.eplb_worker_config().use_decode_only_load);

  worker_args.eplb_worker_config(config);
  EXPECT_TRUE(communicator.parallel_args()->eplb_worker_config().enabled);
  EXPECT_EQ(
      communicator.parallel_args()->eplb_worker_config().redundant_experts_num,
      4);
}

TEST(CollectiveCommunicatorTest, ExplicitDisabledConfigOverridesGlobalEnable) {
  EPLBConfig& global_config = EPLBConfig::get_instance();
  const bool previous_enabled = global_config.enable_eplb();
  global_config.enable_eplb(true);
  CollectiveCommunicator communicator(/*global_rank=*/0,
                                      /*world_size=*/2,
                                      /*dp_size=*/1,
                                      /*ep_size=*/2,
                                      /*cp_size=*/1,
                                      EplbWorkerConfig{});
  global_config.enable_eplb(previous_enabled);

  const EplbWorkerConfig& config =
      communicator.parallel_args()->eplb_worker_config();
  EXPECT_FALSE(config.enabled);
  EXPECT_EQ(config.redundant_experts_num, 1);
  EXPECT_FALSE(config.use_decode_only_load);
}

}  // namespace
}  // namespace xllm

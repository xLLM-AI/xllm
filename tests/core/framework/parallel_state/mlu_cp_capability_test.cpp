/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gtest/gtest.h>

#include "common/options.h"
#include "distributed_runtime/master.h"
#include "framework/config/parallel_config.h"
#include "models/model_registry.h"

namespace xllm {
namespace {

Options make_cp_options(int32_t world_size) {
  Options options;
  options.cp_size(2)
      .dp_size(1)
      .ep_size(world_size)
      .task_type("generate")
      .instance_role(InstanceRole::PREFILL)
      .speculative_algorithm("Suffix");
  ParallelConfig::get_instance().kv_split_size(1);
  return options;
}

TEST(MluCpCapabilityTest, RegistersSupportedModels) {
  EXPECT_TRUE(is_mlu_model_cp_capable("deepseek_v4"));
  EXPECT_EQ(ModelRegistry::get_cp_sharding_mode("deepseek_v4"),
            CpShardingMode::MODEL);
  EXPECT_FALSE(is_mlu_model_cp_capable("deepseek_v32"));
  EXPECT_FALSE(is_mlu_model_cp_capable("glm_moe_dsa"));
  EXPECT_EQ(ModelRegistry::get_cp_sharding_mode("deepseek_v32"),
            CpShardingMode::NONE);
  EXPECT_EQ(ModelRegistry::get_cp_sharding_mode("glm_moe_dsa"),
            CpShardingMode::NONE);
}

TEST(MluCpCapabilityTest, RejectsUnsupportedModels) {
  EXPECT_FALSE(is_mlu_model_cp_capable("deepseek_v4_mtp"));
  EXPECT_FALSE(is_mlu_model_cp_capable("deepseek_v3"));
  EXPECT_FALSE(is_mlu_model_cp_capable("qwen3"));
  EXPECT_FALSE(is_mlu_model_cp_capable("definitely_not_a_model"));
}

TEST(MluCpCapabilityTest, SharedRegistryDoesNotLeakBackendCapability) {
  ModelRegistry::register_cp_sharding_mode("npu_only_fixture",
                                           CpShardingMode::MODEL);
  EXPECT_FALSE(is_mlu_model_cp_capable("npu_only_fixture"));
}

TEST(MluCpCapabilityTest, AcceptsOrthogonalDeepseekV4TargetAndSuffix) {
  constexpr int32_t kWorldSize = 8;
  const Options options = make_cp_options(kWorldSize);

  EXPECT_FALSE(
      validate_model_cp(options, EngineType::LLM, "deepseek_v4", kWorldSize)
          .has_value());
  EXPECT_FALSE(
      validate_model_cp(options, EngineType::SSM, "deepseek_v4", kWorldSize)
          .has_value());
}

TEST(MluCpCapabilityTest, RejectsLegacyModelsAndDraftTargets) {
  constexpr int32_t kWorldSize = 8;
  const Options options = make_cp_options(kWorldSize);

  EXPECT_TRUE(
      validate_model_cp(options, EngineType::LLM, "deepseek_v32", kWorldSize)
          .has_value());
  EXPECT_TRUE(
      validate_model_cp(options, EngineType::LLM, "glm_moe_dsa", kWorldSize)
          .has_value());
  EXPECT_TRUE(
      validate_model_cp(options, EngineType::LLM, "deepseek_v4_mtp", kWorldSize)
          .has_value());
}

TEST(MluCpCapabilityTest, RejectsUnsupportedTopologyAndSpeculation) {
  constexpr int32_t kWorldSize = 8;

  Options invalid_world = make_cp_options(kWorldSize);
  EXPECT_TRUE(validate_model_cp(invalid_world,
                                EngineType::LLM,
                                "deepseek_v4",
                                /*global_world_size=*/7)
                  .has_value());

  Options invalid_ep = make_cp_options(kWorldSize);
  invalid_ep.ep_size(4);
  EXPECT_TRUE(
      validate_model_cp(invalid_ep, EngineType::LLM, "deepseek_v4", kWorldSize)
          .has_value());

  Options invalid_spec = make_cp_options(kWorldSize);
  invalid_spec.speculative_algorithm("MTP");
  EXPECT_TRUE(validate_model_cp(
                  invalid_spec, EngineType::SSM, "deepseek_v4", kWorldSize)
                  .has_value());

  Options invalid_kv_split = make_cp_options(kWorldSize);
  ParallelConfig::get_instance().kv_split_size(2);
  EXPECT_TRUE(validate_model_cp(
                  invalid_kv_split, EngineType::LLM, "deepseek_v4", kWorldSize)
                  .has_value());
  ParallelConfig::get_instance().kv_split_size(1);
}

}  // namespace
}  // namespace xllm

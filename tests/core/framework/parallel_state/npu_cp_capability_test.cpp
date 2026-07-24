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

#include <gtest/gtest.h>

#include <string>

#include "models/model_registry.h"

namespace xllm {
namespace {

TEST(NpuCpCapabilityTest, RegisteredCpCapableModels) {
  // The four models that opt into the NPU ATB model-side CP pipeline.
  EXPECT_TRUE(is_npu_model_cp_capable("deepseek_v32"));
  EXPECT_TRUE(is_npu_model_cp_capable("deepseek_v32_mtp"));
  EXPECT_TRUE(is_npu_model_cp_capable("glm_moe_dsa"));
  EXPECT_TRUE(is_npu_model_cp_capable("glm_moe_dsa_mtp"));
  // The registry must advertise NPU_MODEL for these and NONE for the rest.
  EXPECT_EQ(ModelRegistry::get_cp_sharding_mode("deepseek_v32"),
            CpShardingMode::NPU_MODEL);
  EXPECT_EQ(ModelRegistry::get_cp_sharding_mode("glm_moe_dsa_mtp"),
            CpShardingMode::NPU_MODEL);
}

TEST(NpuCpCapabilityTest, UnregisteredModelsAreNotCapable) {
  // deepseek_v3_mtp uses the DeepSeekV2 decoder without the V3.2 ATB CP
  // metadata/TP contract; it must NOT be advertised as CP-capable so that
  // validate_model_cp rejects deepseek_v3_mtp + cp_size>1 at startup.
  EXPECT_FALSE(is_npu_model_cp_capable("deepseek_v3_mtp"));
  EXPECT_FALSE(is_npu_model_cp_capable("deepseek_v3"));
  // TORCH-only / unrelated NPU models are not CP-capable.
  EXPECT_FALSE(is_npu_model_cp_capable("qwen3"));
  EXPECT_FALSE(is_npu_model_cp_capable("qwen3_atb"));
  EXPECT_FALSE(is_npu_model_cp_capable("deepseek_v4"));
  // Unknown model names default to NONE.
  EXPECT_FALSE(is_npu_model_cp_capable("definitely_not_a_model"));
  EXPECT_EQ(ModelRegistry::get_cp_sharding_mode("deepseek_v3_mtp"),
            CpShardingMode::NONE);
  EXPECT_EQ(ModelRegistry::get_cp_sharding_mode("definitely_not_a_model"),
            CpShardingMode::NONE);
}

TEST(NpuCpCapabilityTest, RegistrationIsIdempotent) {
  // Repeated calls must not flip the capability and must keep returning the
  // same result (std::call_once guards the one-shot registration).
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(is_npu_model_cp_capable("deepseek_v32"));
    EXPECT_FALSE(is_npu_model_cp_capable("deepseek_v3_mtp"));
  }
}

}  // namespace
}  // namespace xllm

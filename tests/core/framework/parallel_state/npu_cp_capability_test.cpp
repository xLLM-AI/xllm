/* Copyright 2025-2026 The xLLM Authors.

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

#include <cstdint>
#include <optional>
#include <string>

#include "core/distributed_runtime/master.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/util/scope_guard.h"
#include "models/model_registry.h"

namespace xllm {
namespace {

TEST(NpuCpCapabilityTest, RegisteredCpCapableModels) {
  // The models that opt into NPU model-side CP. deepseek_v32 / glm_moe_dsa
  // drive it through the ATB NpuCpPlan pipeline; deepseek_v4 owns its split
  // inside the model on the TORCH backend. Both are advertised here because
  // this is the master-side startup gate, not the worker-side sharding switch.
  EXPECT_TRUE(is_npu_model_cp_capable("deepseek_v32"));
  EXPECT_TRUE(is_npu_model_cp_capable("deepseek_v32_mtp"));
  EXPECT_TRUE(is_npu_model_cp_capable("deepseek_v4"));
  EXPECT_TRUE(is_npu_model_cp_capable("deepseek_v4_mtp"));
  EXPECT_TRUE(is_npu_model_cp_capable("glm_moe_dsa"));
  EXPECT_TRUE(is_npu_model_cp_capable("glm_moe_dsa_mtp"));
  // The registry must advertise MODEL for these and NONE for the rest.
  EXPECT_EQ(ModelRegistry::get_cp_sharding_mode("deepseek_v32"),
            CpShardingMode::MODEL);
  EXPECT_EQ(ModelRegistry::get_cp_sharding_mode("glm_moe_dsa_mtp"),
            CpShardingMode::MODEL);
}

TEST(NpuCpCapabilityTest, UnregisteredModelsAreNotCapable) {
  // deepseek_v3_mtp uses the DeepSeekV2 decoder without the V3.2 ATB CP
  // metadata/TP contract; it must NOT be advertised as CP-capable so that
  // validate_model_cp rejects deepseek_v3_mtp + cp_size>1 at startup.
  EXPECT_FALSE(is_npu_model_cp_capable("deepseek_v3_mtp"));
  EXPECT_FALSE(is_npu_model_cp_capable("deepseek_v3"));
  // Unrelated NPU models are not CP-capable.
  EXPECT_FALSE(is_npu_model_cp_capable("qwen3"));
  EXPECT_FALSE(is_npu_model_cp_capable("qwen3_atb"));
  // Hybrid linear attention models are the only ones the graph executor takes
  // through spec-verify chunked prefill; none of them is CP-capable, which is
  // what keeps that capture path CP-free.
  EXPECT_FALSE(is_npu_model_cp_capable("qwen3_next"));
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

TEST(NpuCpCapabilityTest, PythonDcpRejectsCrossDpGroupsBeforePcpEarlyReturn) {
  ModelConfig& model_config = ModelConfig::get_instance();
  ParallelConfig& parallel_config = ParallelConfig::get_instance();
  const std::string original_model_impl = model_config.model_impl();
  const int32_t original_kv_split_size = parallel_config.kv_split_size();
  ScopeGuard config_guard([&] {
    model_config.model_impl(original_model_impl);
    parallel_config.kv_split_size(original_kv_split_size);
  });
  model_config.model_impl("python");
  parallel_config.kv_split_size(2);
  Options options;
  options.cp_size(1).dp_size(2).instance_role(InstanceRole::DECODE);

  const std::optional<std::string> error = validate_model_cp(
      options, EngineType::LLM, "glm_moe_dsa", /*global_world_size=*/8);
  ASSERT_TRUE(error.has_value());
  EXPECT_NE(error->find("Python DCP requires dp_size == 1"), std::string::npos);
  EXPECT_NE(error->find("dp_size=2"), std::string::npos);
  EXPECT_NE(error->find("tp_size=4"), std::string::npos);
  EXPECT_NE(error->find("cp_size=1"), std::string::npos);
  EXPECT_NE(error->find("kv_split_size=2"), std::string::npos);

  model_config.model_impl("py");
  EXPECT_TRUE(validate_model_cp(options,
                                EngineType::LLM,
                                "glm_moe_dsa",
                                /*global_world_size=*/8)
                  .has_value());
  options.dp_size(1);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm_moe_dsa",
                                 /*global_world_size=*/8)
                   .has_value());
  options.dp_size(2);
  parallel_config.kv_split_size(1);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm_moe_dsa",
                                 /*global_world_size=*/8)
                   .has_value());
  parallel_config.kv_split_size(2);
  model_config.model_impl("native");
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm_moe_dsa",
                                 /*global_world_size=*/8)
                   .has_value());
}

TEST(NpuCpCapabilityTest, PythonCpAllowsAclGraphAndRejectsCompileBackends) {
  ExecutionConfig& execution_config = ExecutionConfig::get_instance();
  ModelConfig& model_config = ModelConfig::get_instance();
  ParallelConfig& parallel_config = ParallelConfig::get_instance();
  const std::string original_python_graph_backend =
      execution_config.python_graph_backend();
  const std::string original_model_impl = model_config.model_impl();
  const int32_t original_kv_split_size = parallel_config.kv_split_size();
  ScopeGuard config_guard([&] {
    parallel_config.kv_split_size(original_kv_split_size);
    model_config.model_impl(original_model_impl);
    execution_config.python_graph_backend(original_python_graph_backend);
  });
  execution_config.python_graph_backend("off");
  model_config.model_impl("python");
  parallel_config.kv_split_size(1);

  Options options;
  options.task_type("generate")
      .cp_size(4)
      .dp_size(1)
      .ep_size(16)
      .instance_role(InstanceRole::PREFILL)
      .enable_graph(true);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm_moe_dsa",
                                 /*global_world_size=*/16)
                   .has_value());

  options.enable_graph(false);
  execution_config.python_graph_backend("aclgraph");
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm_moe_dsa",
                                 /*global_world_size=*/16)
                   .has_value());

  execution_config.python_graph_backend("inductor");
  const std::optional<std::string> graph_error = std::optional<std::string>(
      "Python model-side CP requires Prefill to use EagerRunner; use "
      "--python_graph_backend=off or decode-only aclgraph");
  EXPECT_EQ(validate_model_cp(options,
                              EngineType::LLM,
                              "glm_moe_dsa",
                              /*global_world_size=*/16),
            graph_error);

  execution_config.python_graph_backend("off");
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm_moe_dsa",
                                 /*global_world_size=*/16)
                   .has_value());
}

TEST(NpuCpCapabilityTest, PythonCpPreservesQwenAndRestrictsGlm) {
  ExecutionConfig& execution_config = ExecutionConfig::get_instance();
  ModelConfig& model_config = ModelConfig::get_instance();
  ParallelConfig& parallel_config = ParallelConfig::get_instance();
  const std::string original_python_graph_backend =
      execution_config.python_graph_backend();
  const std::string original_model_impl = model_config.model_impl();
  const int32_t original_kv_split_size = parallel_config.kv_split_size();
  ScopeGuard config_guard([&] {
    parallel_config.kv_split_size(original_kv_split_size);
    model_config.model_impl(original_model_impl);
    execution_config.python_graph_backend(original_python_graph_backend);
  });
  execution_config.python_graph_backend("off");
  model_config.model_impl("python");
  parallel_config.kv_split_size(1);

  Options options;
  options.task_type("generate")
      .cp_size(4)
      .dp_size(1)
      .ep_size(16)
      .instance_role(InstanceRole::PREFILL)
      .enable_graph(false)
      .speculative_algorithm("MTP");
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "qwen3",
                                 /*global_world_size=*/16)
                   .has_value());
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::SSM,
                                 "qwen3",
                                 /*global_world_size=*/16)
                   .has_value());
  EXPECT_EQ(validate_model_cp(options,
                              EngineType::SSM,
                              "glm_moe_dsa",
                              /*global_world_size=*/16),
            std::optional<std::string>(
                "Python model-side CP does not support MTP speculative "
                "verification; use the native model executor for GLM MTP on "
                "a cp_size=1 Decode instance"));

  options.speculative_algorithm("DSpark");
  EXPECT_EQ(validate_model_cp(options,
                              EngineType::SSM,
                              "glm_moe_dsa",
                              /*global_world_size=*/16),
            std::optional<std::string>(
                "Current model-side CP does not support aux-hidden-capture "
                "speculative algorithms (Eagle3/DFlash/DSpark); run "
                "speculative decoding on a cp_size=1 Decode instance."));

  options.cp_size(1).instance_role(InstanceRole::DECODE).enable_disagg_pd(true);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm_moe_dsa",
                                 /*global_world_size=*/16)
                   .has_value());
  options.cp_size(4)
      .instance_role(InstanceRole::PREFILL)
      .enable_disagg_pd(false)
      .speculative_algorithm("MTP");

  // EP topology validation belongs to model construction, not this Python CP
  // capability gate.
  options.ep_size(2);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm_moe_dsa",
                                 /*global_world_size=*/16)
                   .has_value());

  parallel_config.kv_split_size(2);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "qwen3",
                                 /*global_world_size=*/16)
                   .has_value());
  EXPECT_EQ(validate_model_cp(options,
                              EngineType::LLM,
                              "glm_moe_dsa",
                              /*global_world_size=*/16),
            std::optional<std::string>(
                "Python GLM CP with kv_split_size > 1 requires "
                "disaggregated PD with the PREFILL role; set "
                "enable_disagg_pd=true and instance_role=PREFILL"));

  options.enable_disagg_pd(true);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm_moe_dsa",
                                 /*global_world_size=*/16)
                   .has_value());

  options.instance_role(InstanceRole::DEFAULT);
  EXPECT_EQ(validate_model_cp(options,
                              EngineType::LLM,
                              "glm_moe_dsa",
                              /*global_world_size=*/16),
            std::optional<std::string>(
                "Python GLM CP with kv_split_size > 1 requires "
                "disaggregated PD with the PREFILL role; set "
                "enable_disagg_pd=true and instance_role=PREFILL"));
  options.instance_role(InstanceRole::PREFILL);

  parallel_config.kv_split_size(1);
  EXPECT_EQ(validate_model_cp(options,
                              EngineType::LLM,
                              "glm_moe_dsa_mtp",
                              /*global_world_size=*/16),
            std::optional<std::string>(
                "Python model-side CP does not support "
                "model_type=glm_moe_dsa_mtp; supported models are qwen3 and "
                "glm_moe_dsa."));

  parallel_config.kv_split_size(3);
  EXPECT_EQ(validate_model_cp(options,
                              EngineType::LLM,
                              "glm_moe_dsa",
                              /*global_world_size=*/16),
            std::optional<std::string>(
                "Python CP requires kv_split_size effective value to be a "
                "positive divisor of cp_size"));
}

}  // namespace
}  // namespace xllm
